import streamlit as st
import os
import zipfile
import subprocess
import shutil
from pathlib import Path
import cv2
import numpy as np
import time
from ultralytics import YOLO
import mss
import json
import albumentations as A
import yaml
import patoolib
from utility.vizualize_samples import draw_labels_on_image

st.set_page_config(layout="wide")


# Paths
VIDS_DIR = Path("./vids")
VIDS_DIR.mkdir(exist_ok=True)

st.title("Detection / Segmentation Data Manager")



import tempfile
# Step 1: Add Data
st.header("1. Add Data")

new_class_name = st.text_input("Enter the class name:")

uploaded_video = st.file_uploader(
    "Upload a video (only 1 at a time)", 
    type=["mp4", "avi", "mov", "mkv"]
)
uploaded_archive = st.file_uploader(
    "Or upload an image archive (ZIP, RAR, 7z, TAR)", 
    type=["zip", "rar", "7z", "tar"]
)
interval_ms = st.number_input("Frame extraction interval (ms)", value=1000, min_value=1)

def provide_download(dir_path: Path):
    """Create a ZIP of the given directory and provide a download button"""
    with tempfile.TemporaryDirectory() as tmp_zip_dir:
        zip_path = Path(tmp_zip_dir) / f"{dir_path.name}.zip"
        shutil.make_archive(str(zip_path).replace(".zip", ""), 'zip', root_dir=dir_path)
        with open(zip_path, "rb") as f:
            st.download_button(
                label="Download Processed Images",
                data=f,
                file_name=f"{dir_path.name}.zip",
                mime="application/zip"
            )

# === Video Processing ===
if uploaded_video:
    if not new_class_name:
        st.error("Please enter a class name before saving the video.")
    else:
        uploaded_video.seek(0, 2)
        size = uploaded_video.tell()
        uploaded_video.seek(0)
        if size == 0:
            st.error("Uploaded video is empty. Please upload a valid video file.")
        else:
            st.info("Saving the video...")
            video_path = VIDS_DIR / uploaded_video.name
            with open(video_path, "wb") as f:
                shutil.copyfileobj(uploaded_video, f)
            st.success(f"Video saved to {video_path} (previous video overwritten if existed)")

            if st.button("Start Frame Extraction"):
                if interval_ms < 1:
                    st.error("Please set a valid frame extraction interval (ms).")
                else:
                    img_dir = VIDS_DIR / f"{new_class_name}_data"
                    img_dir.mkdir(exist_ok=True)

                    st.info(f"Extracting frames for class: {new_class_name}")
                    try:
                        subprocess.run(
                            [
                                "python",
                                str(VIDS_DIR / "extract_frames.py"),
                                str(video_path),
                                str(img_dir),
                                str(interval_ms),
                                str(new_class_name)
                            ],
                            check=True
                        )
                        if img_dir.exists() and any(img_dir.iterdir()):
                            st.success(f"Frames saved to {img_dir}")
                            st.markdown(
                                f"Now go to [makesense.ai](https://www.makesense.ai/), label your images, "
                                f"and download them as a ZIP."
                            )
                            # Provide download button
                            provide_download(img_dir)
                        else:
                            st.error("Frame extraction completed but no output images found.")
                    except subprocess.CalledProcessError as e:
                        st.error(f"Error extracting frames: {e}")

# === Archive Processing ===
elif uploaded_archive:
    if not new_class_name:
        st.error("Please enter a class name before saving the images.")
    else:
        if st.button("Process Archive"):
            st.info("Processing archive...")
            tmp_dir = VIDS_DIR / f"{new_class_name}_tmp"
            tmp_dir.mkdir(exist_ok=True)

            archive_path = tmp_dir / uploaded_archive.name
            with open(archive_path, "wb") as f:
                shutil.copyfileobj(uploaded_archive, f)

            try:
                patoolib.extract_archive(str(archive_path), outdir=str(tmp_dir))

                images = sorted([p for p in tmp_dir.glob("**/*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
                if not images:
                    st.error("No images found in the archive.")
                else:
                    dest_dir = VIDS_DIR / f"{new_class_name}_data"
                    dest_dir.mkdir(exist_ok=True)

                    for idx, img_path in enumerate(images):
                        new_name = f"{new_class_name}_{idx:05d}{img_path.suffix.lower()}"
                        dest_path = dest_dir / new_name
                        shutil.move(str(img_path), dest_path)

                    if dest_dir.exists() and any(dest_dir.iterdir()):
                        st.success(f"{len(images)} images saved to {dest_dir}")
                        st.markdown(
                            f"Now go to [makesense.ai](https://www.makesense.ai/), label your images, "
                            f"and download them as a ZIP."
                        )
                        # Provide download button
                        provide_download(dest_dir)
                    else:
                        st.error("Image processing completed but no output images found.")

            except Exception as e:
                st.error(f"Error extracting archive: {e}")
            finally:
                if archive_path.exists():
                    archive_path.unlink()
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)



# -------------------------------------------------------------------------------------------------------------------------
st.divider()




# Step 2: Upload labeled ZIP
st.header("2. Upload Labeled Data")

train_split = st.slider(
    "Select training split fraction",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.01,
    help="Fraction of data to use for training (rest will be validation)"
)

# Let user upload any common archive type
labeled_archive = st.file_uploader(
    "Upload your labeled data archive (ZIP, RAR, 7z, TAR, etc.)",
    type=["zip", "rar", "7z", "tar", "gz"]
)

if labeled_archive:
    if st.button("Start Processing Labeled Data"):
        if not new_class_name:
            st.error("Please enter a class name first.")
        else:
            img_dir = VIDS_DIR / f"{new_class_name}_data"
            img_dir.mkdir(exist_ok=True)

            # Use a temporary directory for the uploaded archive
            with tempfile.TemporaryDirectory() as tmp:
                archive_path = Path(tmp) / labeled_archive.name
                # Save uploaded archive safely
                archive_bytes = labeled_archive.read()
                with open(archive_path, "wb") as f:
                    f.write(archive_bytes)

                try:
                    # Extract with patoolib
                    patoolib.extract_archive(str(archive_path), outdir=str(img_dir))

                    # Flatten subfolders and remove duplicates
                    for root, _, files in os.walk(img_dir):
                        for file in files:
                            src = Path(root) / file
                            dst = img_dir / file
                            if src != dst:
                                if dst.exists():
                                    dst.unlink()
                                src.replace(dst)
                    for subdir in img_dir.iterdir():
                        if subdir.is_dir():
                            shutil.rmtree(subdir)

                    st.success(f"Labeled data extracted into {img_dir}")

                    # Rename files
                    st.info("Renaming images...")
                    # Count actual number of image files in img_dir
                    image_files = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
                    max_files = len(image_files)

                    subprocess.run(
                        [
                            "python",
                            str(Path("./utility/rename_file_base.py").resolve()),
                            "--dir", str(img_dir),
                            "--old_prefix", "frame",
                            "--new_prefix", new_class_name,
                            "--limit", str(max_files)
                        ],
                        check=True
                    )
                    st.success(f"{max_files} images renamed successfully!")

                except Exception as e:
                    st.error(f"Error extracting or renaming archive: {e}")

                # Run dataset setup
                st.info("Setting up labeled data...")
                try:
                    subprocess.run(
                        [
                            "python",
                            str(VIDS_DIR / "setup_new_data.py"),
                            str(img_dir),
                            str(new_class_name),
                            "--train-split", str(train_split)
                        ],
                        check=True
                    )
                    st.success("New data setup complete!")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error setting up new data: {e}")




# -------------------------------------------------------------------------------------------------------------------------
st.divider()




# Step 3: Setup data for training
st.header("3. Setup Training")

steps = {
    "clear_datasets": st.checkbox("Clear datasets", True),
    "renew_datasets": st.checkbox("Renew datasets from storage", True),
    "update_yaml": st.checkbox("Update data yaml", True),
    "segment": st.checkbox("Segment boxes", True),
}

# Make tiling only available if segmentation is selected
if steps["segment"]:
    steps["tile_and_mask"] = st.checkbox("Tile images and masks", False)
else:
    steps["tile_and_mask"] = False  # auto-disable if segmentation is off

steps["augment"] = st.checkbox("Create augmentations", True)


# Get all available classes from folder names under ./my_dataset/images/storage/train
storage_train_dir = "./my_dataset/images/storage/train"
available_classes = sorted(
    [d for d in os.listdir(storage_train_dir) if os.path.isdir(os.path.join(storage_train_dir, d))]
)

# Multiselect for classes with "all" option
selected_classes = st.multiselect(
    "Select classes to process (choose 'all' to process all valid classes):",
    options=["all"] + available_classes,
    default=["all"]
)

if steps["segment"]:
    # Strategy labels
    strategy_labels = {
        0: "Convex Hull (simple, smooth shapes with few edges)",
        1: "Largest Morphological Merge (detailed, complex or broken shapes)",
    }

    # Show strategy selection for each chosen class
    class_strategies = {}
    if selected_classes:
        classes_for_strategy = available_classes if "all" in selected_classes else selected_classes
        st.subheader("Assign Segmentation Strategies")
        for cls in classes_for_strategy:
            class_strategies[cls] = st.selectbox(
                f"Strategy for '{cls}':",
                options=list(strategy_labels.keys()),
                format_func=lambda k: strategy_labels[k],
                key=f"strategy_{cls}"
            )

if steps["augment"]:
    # Numeric input for number of augmentations
    num_augs = st.number_input("Number of augmentations per sample:", min_value=1, max_value=100, value=6)

    # Custom augmentation code input
    st.subheader("Custom Augmentation (Albumentations code)")
    st.markdown(
        "Write your Albumentations here. It must return an `A.Compose([...])` object. "
        "Example:\n```python\nA.Compose([\n    A.HorizontalFlip(p=0.5),\n    A.RandomBrightnessContrast(p=0.3)\n])\n```"
    )
    aug_cfg_path = "./tmp_aug_config.json"

    try:
        with open(aug_cfg_path, "r") as f:
            aug_cfg = json.load(f)
            default_code = aug_cfg.get("code", "")
    except FileNotFoundError:
        default_code = ""

    if not default_code.strip():
        default_code = """A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3)
    ])"""

    user_aug_code = st.text_area(
        "Albumentations code:",
        default_code,
        height=350
    )

    # Validate augmentation code before running
    aug_is_valid = False
    if user_aug_code.strip():
        # Restricted evaluation: give access to necessary modules
        allowed_globals = {
            "A": A,
            "cv2": cv2,
            "np": np,
        }

        try:
            transform = eval(user_aug_code, allowed_globals)
            if isinstance(transform, (A.BasicTransform, A.BaseCompose)):
                aug_is_valid = True
                st.success("✅ Augmentation code is valid.")
            else:
                st.warning("⚠ The code did not return a valid Albumentations transform.")
        except Exception as e:
            st.error(f"❌ Error parsing augmentation code: {e}")


# Run setup pipeline
if st.button("Run Setup Pipeline"):
    if len(selected_classes) == 0:
        st.error("Must select at least one class.")
    elif len(available_classes) == 0:
        st.warning("There are currently no classes in the storage.")
    elif not aug_is_valid and steps["augment"]:
        st.error("Augmentation code is invalid. Please fix before running.")
    else:
        st.info("🚀 Starting setup pipeline... please wait.")
        log_area = st.empty()
        log_lines = []
        MAX_LINES = 16

        # Prepare CLI arguments
        if "all" in selected_classes:
            classes_arg = ",".join(available_classes)
        else:
            classes_arg = ",".join(selected_classes)

        if steps["segment"]:
            strategies_arg = ",".join(f"{cls}:{class_strategies[cls]}" for cls in class_strategies)

        if steps["augment"]:
            # Save augmentation code to a temp JSON so setup.py can load it
            aug_config_path = "./tmp_aug_config.json"
            os.makedirs(os.path.dirname(aug_config_path), exist_ok=True)
            with open(aug_config_path, "w") as f:
                json.dump({"code": user_aug_code}, f)

        steps_arg = "".join(f"{key}_{str(int(val))}|" for key,val in steps.items())

        process = subprocess.Popen(
            [
                "python", "-u", "./yolo/setup.py",
                # classes, strats, aug_cfg, aug_n
                steps_arg,
                classes_arg,
                strategies_arg if steps["segment"] else "",
                aug_config_path if steps["augment"] else "",
                str(num_augs) if steps["augment"] else "",
                
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            encoding="utf-8",
        )

        for line in process.stdout:
            log_lines.append(line.rstrip())
            if len(log_lines) > MAX_LINES:
                log_lines = log_lines[-MAX_LINES:]
            log_area.text_area("Pipeline Output (latest logs)", "\n".join(log_lines), height=400)
        log_area = st.empty()

        process.stdout.close()
        retcode = process.wait()

        if retcode == 0:
            st.success("✅ Setup pipeline completed successfully.")
        else:
            st.error(f"❌ Pipeline exited with code {retcode}")




# -------------------------------------------------------------------------------------------------------------------------
st.divider()



# --- Streamlit App ---
st.title("3.1. YOLO Training Data Viewer")

# Paths
image_dir = "./my_dataset/images/train"
label_dir = "./my_dataset/labels/train"

# Load class names
with open("./my_dataset/data.yaml", "r") as f:
    class_names = yaml.safe_load(f)["names"]

# Get image files
image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
num_images = len(image_files)

if num_images == 0:
    st.warning("No images found in the dataset!")
else:
    # Persistent state for current index
    if "img_idx" not in st.session_state:
        st.session_state.img_idx = 0

    # Navigation buttons
    col_prev, col_next, _ = st.columns([0.12, 0.12, 0.76])
    with col_prev:
        if st.button("⬅ Previous", use_container_width=True):
            st.session_state.img_idx = max(0, st.session_state.img_idx - 1)
            st.rerun()  # refresh immediately

    with col_next:
        if st.button("Next ➡", use_container_width=True):
            st.session_state.img_idx = min(num_images - 1, st.session_state.img_idx + 1)
            st.rerun()

    # Display current image
    idx = st.session_state.img_idx
    img_file = image_files[idx]
    image_path = os.path.join(image_dir, img_file)
    label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

    st.write(f"**{idx+1}/{num_images}: {img_file}**")
    img = draw_labels_on_image(image_path, label_path, class_names)
    MAX_DISPLAY = 800 
    h, w = img.shape[:2]
    scale = min(MAX_DISPLAY / w, MAX_DISPLAY / h, 1.0)  # never scale up
    disp_img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    st.image(disp_img)




# -------------------------------------------------------------------------------------------------------------------------
st.divider()



import glob

st.header("4. Start Training")

epochs = st.number_input("Number of epochs:", min_value=1, max_value=1000, value=10)
training_imgsz = st.number_input("Training Image size (imgsz):", min_value=32, max_value=2048, step=32, value=960)

def labels_are_detection_only():
    label_files = glob.glob("./my_dataset/labels/train/*.txt")
    for lf in label_files:
        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # If any line has more than 5 elements, it's segmentation
                if len(parts) > 5:
                    return False
    return True

if st.button("Start Training"):
    st.info("⚙️ Starting training... please wait.")
    log_area = st.empty()
    log_lines = []
    MAX_LINES = 10

    # Decide which script to run
    if labels_are_detection_only():
        train_script = "./yolo/detection/train.py"
    else:
        train_script = "./yolo/segmentation/train.py"

    process = subprocess.Popen(
        ["python", "-u", train_script, str(epochs), str(training_imgsz)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        encoding="utf-8",
    )

    for line in process.stdout:
        log_lines.append(line.rstrip())
        if len(log_lines) > MAX_LINES:
            log_lines = log_lines[-MAX_LINES:]
        log_area.text_area("Training Output (latest logs)", "\n".join(log_lines), height=400)
    log_area = st.empty()

    process.stdout.close()
    retcode = process.wait()

    if retcode == 0:
        st.success("✅ Training completed successfully.")
    else:
        st.error(f"❌ Training exited with code {retcode}")




# -------------------------------------------------------------------------------------------------------------------------
import shutil

st.divider()

st.title("5. Real-Time YOLO Detection / Segmentation")

mode = st.radio("Select mode:", ["Detection", "Segmentation"])

model_dir = "runs/detect" if mode == "Detection" else "runs/segment"
available_models = []
if os.path.exists(model_dir):
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".pt"):
                available_models.append(os.path.join(root, file))

if not available_models:
    st.error(f"No models found in {model_dir}")
    #st.stop()

model_path = st.selectbox("Select model:", available_models)

if st.button("🗑 Delete Selected Model"):
    parent_dir = os.path.dirname(model_path)
    try:
        shutil.rmtree(parent_dir)
        st.success(f"Deleted model directory: {parent_dir}")
        st.rerun()
    except Exception as e:
        st.error(f"Error deleting directory: {e}")

# Image size
inference_imgsz = st.number_input("Inference Image size (imgsz):", min_value=32, max_value=2048, step=32, value=960)

# Confidence
pred_conf = st.number_input("Prediction confidence: ", min_value=0.01, max_value=1.0, value=0.45)

# Screen region
st.subheader("Screen Region")
top = st.number_input("Top", value=450)
left = st.number_input("Left", value=800)
width = st.number_input("Width", value=960)
height = st.number_input("Height", value=540)

# --- State management ---
if "run_inference" not in st.session_state:
    st.session_state.run_inference = False
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# Start/Stop buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶ Start Real-Time Inference"):
        if model_path == None:
            st.error("Please select a valid model.")
        else:
            if not st.session_state.model_loaded:
                st.info(f"Loading model from {model_path} ...")
                st.session_state.model = YOLO(model_path)
                st.session_state.model_loaded = True
            st.session_state.run_inference = True
with col2:
    if st.button("⏹ Stop"):
        st.session_state.run_inference = False

# --- Run inference (one frame per Streamlit cycle) ---
if st.session_state.run_inference and st.session_state.model_loaded:
    sct = mss.mss()
    screen_region = {'top': top, 'left': left, 'width': width, 'height': height}
    frame_display = st.empty()

    MAX_DISPLAY = 800

    while st.session_state.run_inference:
        screen = np.array(sct.grab(screen_region))
        frame_rgb = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)

        results = st.session_state.model(frame_rgb[..., ::-1], imgsz=inference_imgsz, conf=pred_conf)

        img_with_boxes_bgr = results[0].plot()
        img_with_boxes_rgb = cv2.cvtColor(img_with_boxes_bgr, cv2.COLOR_BGR2RGB)

        # --- Scale for display ---
        h, w = img_with_boxes_rgb.shape[:2]
        scale = min(MAX_DISPLAY / w, MAX_DISPLAY / h, 1.0)  # never scale up
        disp_img = cv2.resize(img_with_boxes_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        frame_display.image(disp_img, channels="RGB")
        time.sleep(0.001)




st.divider()


st.header("⚠️ Delete a Class")

del_selected_classes = st.multiselect(
    "Select classes you want to delete:",
    options=available_classes,
)

if st.button("Delete Class"):
    if not del_selected_classes:
        st.error("Please enter a class name.")
    else:
        base_dirs = [
            Path("my_dataset/images/storage/train"),
            Path("my_dataset/images/storage/val"),
            Path("my_dataset/labels/storage/train"),
            Path("my_dataset/labels/storage/val"),
        ]

        for delete_class_name in del_selected_classes:
            deleted_any = False
            for base in base_dirs:
                class_dir = base / delete_class_name
                if class_dir.exists() and class_dir.is_dir():
                    shutil.rmtree(class_dir)
                    st.write(f"Deleted: {class_dir}")
                    deleted_any = True

            if deleted_any:
                st.success(f"Class '{delete_class_name}' deleted from dataset.")
            else:
                st.warning(f"No folders found for class '{delete_class_name}'.")





from yolo.segmentation.train_utility import restore_labels, tmp_dirs
import atexit

def cleanup():
    print("App is shutting down, cleaning up...")

    for tmp_dir in tmp_dirs:
        if os.path.exists(tmp_dir) and os.listdir(tmp_dir):
            print(f" Restoring from existing temp folder: {tmp_dir}")
            restore_labels()
            break

atexit.register(cleanup)