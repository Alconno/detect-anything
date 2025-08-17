import os
import shutil
import cv2
import glob
import yaml
import sys
import time
import numpy as np
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from segmentation.segment_boxes import segment_boxes
from augmentation.segment_augmentation import segment_aug
from augmentation.box_augmentation import box_aug
from utility.cantor_pair import reverse_pairing_function, pairing_function

image_dirs = ["./my_dataset/images/train", "./my_dataset/images/val"]
label_dirs = ["./my_dataset/labels/train", "./my_dataset/labels/val"]
mask_dirs = ["./my_dataset/masks/train", "./my_dataset/masks/val"]
tmp_dirs = ["./my_dataset/labels/ttmp", "./my_dataset/labels/vtmp"]





def renew_from_storage(classes):
    """
    Restore label/image pairs from storage for the given classes.

    Steps:
        1. Copies label + image files from `my_dataset/labels/storage/*` and 
           `my_dataset/images/storage/*` into the working dataset.
        2. Ensures every requested class has its files renewed.
        3. Images are only copied if missing in the dataset.
        4. After copying, label files are remapped so all class IDs 
           start from 0 (currently hardcoded to `0`).

    Args:
        classes (list[str]): List of class names to restore 
                             (case-insensitive, must match subdir names).

    Raises:
        ValueError: If `classes` is empty or contains invalid names.

    Side Effects:
        - Creates missing dataset directories if not present.
        - Overwrites label files in train/val dirs.
        - Prints summary of renewed labels and copied images.
    """
    
    if not classes or any(cls.strip() == "" for cls in classes):
        raise ValueError("You must provide at least one valid class name.")

    classes = [cls.lower() for cls in classes]
    class_to_index = {cls: i for i, cls in enumerate(classes)}

    base_pairs = [
        ("./my_dataset/labels/train", "./my_dataset/labels/storage/train", "./my_dataset/images/train", "./my_dataset/images/storage/train"),
        ("./my_dataset/labels/val", "./my_dataset/labels/storage/val", "./my_dataset/images/val", "./my_dataset/images/storage/val"),
    ]

    renewed_labels = 0
    copied_images = 0

    # Step 1: Copy label and image files as-is from storage
    for label_dir, label_storage_dir, image_dir, image_storage_dir in base_pairs:
        subdirs = [
            d for d in os.listdir(label_storage_dir)
            if d.lower() in classes and os.path.isdir(os.path.join(label_storage_dir, d))
        ]

        for sub in subdirs:
            label_subdir = os.path.join(label_storage_dir, sub)
            image_subdir = os.path.join(image_storage_dir, sub)

            for fname in os.listdir(label_subdir):
                if not fname.endswith(".txt"):
                    continue

                storage_file = os.path.join(label_subdir, fname)
                label_file = os.path.join(label_dir, fname)
                os.makedirs(label_dir, exist_ok=True)
                shutil.copy2(storage_file, label_file)
                renewed_labels += 1

                # Copy image if missing
                base_name = os.path.splitext(fname)[0]
                found_img = False
                for ext in [".jpg", ".jpeg", ".png"]:
                    img_src = os.path.join(image_subdir, base_name + ext)
                    img_dst = os.path.join(image_dir, base_name + ext)

                    if os.path.isfile(img_src):
                        found_img = True
                        os.makedirs(image_dir, exist_ok=True)
                        if not os.path.exists(img_dst):
                            shutil.copy2(img_src, img_dst)
                            copied_images += 1
                        break

                if not found_img:
                    print(f"⚠️ No image found for label {fname} in {image_subdir}")

    print(f"✅ Renewed {renewed_labels} label files from storage ({classes}).")
    print(f"✅ Copied {copied_images} new images (skipped existing).")

    # Step 2: Update copied label files to remap class IDs uniquely
    print("🛠️ Updating copied label files to remap class IDs...")

    for label_dir in ["./my_dataset/labels/train", "./my_dataset/labels/val"]:
        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue

            label_path = os.path.join(label_dir, fname)
            with open(label_path, "r") as f:
                lines = f.readlines()

            remapped_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Ignore old class id and always map to 0
                    parts[0] = "0"  # Always class 0 (or implement logic for multiple classes)
                    remapped_lines.append(" ".join(parts))

            with open(label_path, "w") as f:
                f.write("\n".join(remapped_lines))

    print("✅ Finished remapping class IDs in copied label files.")





def clear_dataset_dirs():
    """
    Completely clear dataset directories for images, labels, and masks.

    Targeted directories:
        - ./my_dataset/labels/train
        - ./my_dataset/labels/val
        - ./my_dataset/images/train
        - ./my_dataset/images/val
        - ./my_dataset/masks/train
        - ./my_dataset/masks/val

    Behavior:
        - Removes all files and subdirectories.
        - If a directory is missing, prints a warning.
        - Uses safe deletion (files, symlinks, and nested dirs).

    Side Effects:
        - Destroys any existing dataset content in the above folders.
    """

    dirs_to_clear = [
        "./my_dataset/labels/train", 
        "./my_dataset/labels/val",
        "./my_dataset/images/train", 
        "./my_dataset/images/val",
        "./my_dataset/masks/train",
        "./my_dataset/masks/val",
    ]

    for dir_path in dirs_to_clear:
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
        else:
            print(f"Directory not found: {dir_path}")





def update_data_yaml_classes(classes):
    """
    Update `my_dataset/data.yaml` with the provided classes.

    Steps:
        1. If "all" is passed, auto-detect classes by scanning storage dirs 
           (only keeping classes that exist in BOTH images and labels storage).
        2. Loads `data.yaml` if present; otherwise creates a default config.
        3. Updates:
            - `nc`: number of classes
            - `names`: class name list
        4. Writes back to disk.

    Args:
        classes (list[str]): List of classes, or ["all"] to auto-detect.

    Side Effects:
        - Modifies or creates `my_dataset/data.yaml`.
        - Prints confirmation with updated class list.
    """

    yaml_path = "./my_dataset/data.yaml"
    img_storage_train = "./my_dataset/images/storage/train"
    label_storage_train = "./my_dataset/labels/storage/train"

    # Resolve classes to use
    if "all" in [c.lower() for c in classes]:
        img_classes = set(d for d in os.listdir(img_storage_train) if os.path.isdir(os.path.join(img_storage_train, d)))
        label_classes = set(d for d in os.listdir(label_storage_train) if os.path.isdir(os.path.join(label_storage_train, d)))

        # Only keep classes present in both dirs
        filtered_classes = sorted(list(img_classes.intersection(label_classes)))
    else:
        filtered_classes = classes

    # Load existing YAML (or create default if missing)
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
    else:
        data = {
            "path": "my_dataset",
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": ["humanoid"],
            "task": "segment"
        }

    # Update classes count and names
    data["nc"] = len(filtered_classes)
    data["names"] = filtered_classes

    # Write back updated YAML
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"✅ Updated {yaml_path} with classes: {filtered_classes} (nc={data['nc']})")





def tile_images_and_masks(tile_size=960, overlap=0.15):
    """
    Tile large images and masks into smaller square patches.

    Steps:
        1. Iterates over training and validation image/mask directories.
        2. Splits each pair into overlapping tiles of `tile_size`.
        3. Ensures instance IDs are re-mapped to local contiguous IDs 
           per class within each tile.
        4. Pads tiles if smaller than `tile_size`.
        5. Deletes original image + mask after tiling.
        6. Moves generated tiles back to the original dirs.

    Args:
        tile_size (int, optional): Size of output square tiles. Default = 960.
        overlap (float, optional): Overlap ratio between tiles (0–1). Default = 0.15.

    Side Effects:
        - Removes original image/mask pairs.
        - Creates new tiled images and masks.
        - Prints progress for each directory.
    """
      
    stride = int(tile_size * (1 - overlap))

    for img_dir, mask_dir in zip(image_dirs, mask_dirs):
        print(f"🧹 Processing {img_dir} and {mask_dir}...")

        # Get sorted file lists
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.*")))

        # Create temp directories to store tiles
        temp_img_dir = img_dir + "_temp"
        temp_mask_dir = mask_dir + "_temp"
        os.makedirs(temp_img_dir, exist_ok=True)
        os.makedirs(temp_mask_dir, exist_ok=True)

        # Process each pair of mask + image
        for mask_path, img_path in zip(mask_files, img_files):
            fname = os.path.basename(img_path)

            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            img_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            if mask_img is None or img_img is None:
                print(f"⚠️ Skipping {fname}, missing image or mask.")
                continue

            h, w = mask_img.shape[:2]
            tile_id = 0

            # Create tiles
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    x_end = min(x + tile_size, w)
                    y_end = min(y + tile_size, h)

                    mask_tile = mask_img[y:y_end, x:x_end]
                    img_tile = img_img[y:y_end, x:x_end]

                    unique_paired_ids = np.unique(mask_tile)
                    unique_paired_ids = unique_paired_ids[unique_paired_ids != 0]  # exclude background

                    # Decode all paired ids in the tile
                    decoded = [reverse_pairing_function(pid) for pid in unique_paired_ids]

                    # Map from (cls_id, old_inst_id) to new_inst_id per cls_id
                    new_instance_ids = {}
                    instance_counters = {}

                    for cls_id, old_inst_id in decoded:
                        if cls_id not in instance_counters:
                            instance_counters[cls_id] = 1

                        new_inst_id = instance_counters[cls_id]
                        new_instance_ids[(cls_id, old_inst_id)] = new_inst_id
                        instance_counters[cls_id] += 1

                    # Create remapped mask tile
                    remapped_mask_tile = np.zeros_like(mask_tile)

                    for old_paired_id in unique_paired_ids:
                        cls_id, old_inst_id = reverse_pairing_function(old_paired_id)
                        new_inst_id = new_instance_ids[(cls_id, old_inst_id)]
                        new_paired_id = pairing_function(cls_id, new_inst_id)
                        remapped_mask_tile[mask_tile == old_paired_id] = new_paired_id

                    mask_tile = remapped_mask_tile

                    # Pad tiles if smaller than tile size
                    if mask_tile.shape[0] < tile_size or mask_tile.shape[1] < tile_size:
                        mask_tile = cv2.copyMakeBorder(
                            mask_tile,
                            0, tile_size - mask_tile.shape[0],
                            0, tile_size - mask_tile.shape[1],
                            cv2.BORDER_CONSTANT, value=0
                        )
                        img_tile = cv2.copyMakeBorder(
                            img_tile,
                            0, tile_size - img_tile.shape[0],
                            0, tile_size - img_tile.shape[1],
                            cv2.BORDER_CONSTANT, value=(0, 0, 0)
                        )

                    if np.any(mask_tile > 0):
                        tile_name = f"{os.path.splitext(fname)[0]}_{tile_id:04d}.png"
                        cv2.imwrite(os.path.join(temp_mask_dir, tile_name), mask_tile)
                        cv2.imwrite(os.path.join(temp_img_dir, tile_name), img_tile)
                        tile_id += 1

            # After tiling this pair, delete the original mask and image
            try:
                os.remove(mask_path)
                os.remove(img_path)
                print(f"🗑️ Deleted original files: {fname}")
            except Exception as e:
                print(f"❌ Failed to delete original files {fname}: {e}")

        # Move all tiles from temp folders into the original directories
        for f in glob.glob(os.path.join(temp_img_dir, "*")):
            shutil.move(f, img_dir)
        for f in glob.glob(os.path.join(temp_mask_dir, "*")):
            shutil.move(f, mask_dir)

        # Remove temporary directories
        os.rmdir(temp_img_dir)
        os.rmdir(temp_mask_dir)

        print(f"✅ Finished tiling {img_dir} & {mask_dir} — originals removed.")



def masks_to_labels():
    """
    Convert mask tiles into YOLO segmentation label files.

    Steps:
        1. Clears all existing label directories (but keeps images).
        2. For each mask-image pair:
            - Reads mask values (encoded via pairing_function).
            - Extracts polygons per instance (via OpenCV contours).
            - Writes YOLO `.txt` labels containing:
                class_instance_id, bbox center (x,y), bbox size (w,h), polygon points.
        3. Saves generated labels into train/val label dirs.

    Format:
        Each line in label file = 
        "<cls_id>_<instance_id> x_center y_center width height x1 y1 x2 y2 ..."

    Side Effects:
        - Deletes and regenerates all label files.
        - Skips masks without a corresponding image.
        - Prints summary after processing each split.
    """

    # Step 1: Clear original label folders only (no images)
    for ld in label_dirs:
        if os.path.exists(ld):
            for f in os.listdir(ld):
                path = os.path.join(ld, f)
                try:
                    if os.path.isfile(path) or os.path.islink(path):
                        os.remove(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                except Exception as e:
                    print(f"Failed to delete {path}: {e}")
        else:
            print(f"Label directory not found (skipping clear): {ld}")

    # Step 2: Generate new labels from masks and images
    for mask_dir, img_dir, label_dir in zip(mask_dirs, image_dirs, label_dirs):
        os.makedirs(label_dir, exist_ok=True)
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        for fname in mask_files:
            mask_path = os.path.join(mask_dir, fname)
            img_path = os.path.join(img_dir, fname)

            if not os.path.exists(img_path):
                print(f"⚠️ No matching image for {fname}, skipping...")
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"⚠️ Failed to read mask {fname}, skipping...")
                continue

            h, w = mask.shape

            label_lines = []

            unique_vals = np.unique(mask)
        
            instance_id = 0
            for val in unique_vals:
                if val == 0:
                    continue
              
                cls_id, _ = reverse_pairing_function(val)

                binary_mask = (mask == val).astype(np.uint8) * 255

                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if len(contour) < 3:
                        continue

                    instance_id += 1
                    contour = contour.reshape(-1, 2)

                    poly = []
                    x_min, y_min = float('inf'), float('inf')
                    x_max, y_max = float('-inf'), float('-inf')

                    for x, y in contour:
                        poly.append(x / w)
                        poly.append(y / h)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)
                    x_c = ((x_min + x_max) / 2) / w
                    y_c = ((y_min + y_max) / 2) / h
                    bw = (x_max - x_min) / w
                    bh = (y_max - y_min) / h

                    label_line = (
                        f"{cls_id}_{instance_id} "
                        f"{x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f} "
                        + " ".join(f"{p:.6f}" for p in poly)
                    )
                    label_lines.append(label_line)


                

            label_path = os.path.join(label_dir, os.path.splitext(fname)[0] + ".txt")
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

        print(f"✅ Generated YOLO labels for {label_dir}: {len(os.listdir(label_dir))} files")



# Allow special characters to be streamed to streamlit
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdout.reconfigure(line_buffering=True)

def setup_pipe(steps_map, classes, strategy_map, aug_transform, num_augs=5):
    """
    Execute the full preprocessing pipeline step-by-step.

    Steps (controlled by `steps_map` flags):
        - clear_datasets : wipe existing dataset dirs.
        - renew_datasets : restore data from storage.
        - update_yaml    : update data.yaml with given classes.
        - segment        : run segmentation strategy (`segment_boxes`).
        - tile_and_mask  : split into tiles + regenerate YOLO labels.
        - augment        : apply augmentations (segment_aug or box_aug).

    Args:
        steps_map (dict[str, int]): Step execution map (1 = enabled, 0 = skipped).
        classes (list[str]): List of classes to process.
        strategy_map (dict[str, int]): Per-class segmentation strategy.
        aug_transform (callable | None): Albumentations transform pipeline.
        num_augs (int, optional): Number of augmentations to generate. Default = 5.

    Side Effects:
        - Modifies dataset folders.
        - Writes YAML, masks, labels, images.
        - Calls augmentation and segmentation functions.
        - Prints step progress and completion.
    """

    step_idx = 1

    if steps_map["clear_datasets"]:
        print("\n" + "="*60)
        print(f"🧹  STEP {step_idx}: REMOVING OLD FILES")
        print("="*60 + "\n")
        clear_dataset_dirs() # Clear all so we can prepare requested data from the storage
        #remove_all_aug()
        time.sleep(0.2)
        
    if steps_map["renew_datasets"]:
        print("\n" + "="*60)
        print(f"📦  STEP {step_idx}: RENEWING LABELS + IMAGES FROM STORAGE")
        print("="*60 + "\n")
        renew_from_storage(classes)
        time.sleep(0.2)
        step_idx += 1

    if steps_map["update_yaml"]:
        print("\n" + "="*60)
        print(f"📦  STEP {step_idx}: UPDATING DATA.YAML")
        print("="*60 + "\n")
        update_data_yaml_classes(classes)
        time.sleep(0.2)
        step_idx += 1

    if steps_map["segment"]:
        print("\n" + "="*60)
        print(f"🧠  STEP {step_idx}: SEGMENTING BOXES (MASK GENERATION)")
        print("="*60 + "\n")
        segment_boxes(strategy_map)
        time.sleep(0.2)
        step_idx += 1

    if steps_map["tile_and_mask"]:
        print("\n" + "="*60)
        print(f"🪓  STEP {step_idx}: TILING MASKS + IMAGES")
        print("="*60 + "\n")
        tile_images_and_masks(tile_size=960, overlap=0.15)
        time.sleep(0.2)
        step_idx += 1

        print("\n" + "="*60)
        print(f"📝  STEP {step_idx}: GENERATE UPDATED YOLO LABELS FROM MASK TILES")
        print("="*60 + "\n")
        masks_to_labels()
        time.sleep(0.2)
        step_idx += 1
    
    if steps_map["augment"]:
        print("\n" + "="*60)
        print(f"🧪  STEP {step_idx}: GENERATING AUGMENTATIONS")
        print("="*60 + "\n")
        if steps_map["segment"]:
            segment_aug(strategy_map, aug_transform, num_augs)
        else:
            box_aug(aug_transform, num_augs)
        time.sleep(0.2)
        step_idx += 1

    print("\n" + "="*60)
    print("✅  SETUP COMPLETE")
    print("="*60 + "\n")




def parse_classes_arg(arg):
    """
    Parse comma-separated classes argument from CLI input.

    Args:
        arg (str): String of classes (e.g. "fire,rhino,tree").

    Returns:
        list[str]: Cleaned list of class names.

    Raises:
        ValueError: If no valid class names are provided.
    """

    classes = [cls.strip() for cls in arg.split(",") if cls.strip()]
    if not classes:
        raise ValueError("You must provide at least one class name.")
    return classes





def main():
    import sys, multiprocessing, json, albumentations as A, cv2, numpy as np
    multiprocessing.freeze_support()

    # Expected args:
    # 0: steps
    # 1: classes (comma-separated)
    # 2: strategies (optional, e.g., "fire:0,rhino:1")
    # 3: aug_cfg_path (optional)
    # 4: num_augs (optional)

    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit("❌ Usage: setup.py <steps> <classes> [strategies] [aug_cfg_path] [num_augs]")

    steps = args[0] # format "1_1_1_1_0_1_"
    classes = parse_classes_arg(args[1])
    strategies_arg = args[2] if len(args) > 2 and ":" in args[2] else ""
    aug_cfg_path = args[3] if len(args) > 3 or (len(args) > 2 and ":" not in args[2]) else ""
    num_augs = int(args[4]) if len(args) > 4 else 5

    # Load augmentation config if provided
    aug_transform = None
    if aug_cfg_path:
        with open(aug_cfg_path, "r") as f:
            aug_cfg = json.load(f)
        local_env = {"A": A, "cv2": cv2, "np": np}
        aug_transform = eval(aug_cfg["code"], local_env)

    # Build strategy map
    if strategies_arg:
        strategy_map = {}
        for pair in strategies_arg.split(","):
            cls, strat = pair.split(":")
            strategy_map[cls] = int(strat)
    else:
        strategy_map = {cls: 0 for cls in classes}

    steps_map = {k: int(v) for k, v in (pair.rsplit("_", 1) for pair in steps.strip("|").split("|"))}

    setup_pipe(steps_map, classes, strategy_map, aug_transform, num_augs)


if __name__ == "__main__":
    main()