import os
import cv2
import numpy as np
import yaml

# ==== Paths ====
label_dirs = ["./my_dataset/labels/train", "./my_dataset/labels/val"]
mask_dirs = ["./my_dataset/masks/train", "./my_dataset/masks/val"]
image_dirs = ["./my_dataset/images/train", "./my_dataset/images/val"]

# Load class names from data.yaml (optional)
with open("./my_dataset/data.yaml", "r") as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg["names"]

for label_dir, mask_dir, image_dir in zip(label_dirs, mask_dirs, image_dirs):
    """
    Generates instance masks from YOLO-format polygon labels.

    For each label file:
    1. Finds the corresponding image in the `image_dir`.
    2. Reads segmentation polygons from the label file (after the first 5 values).
    3. Converts normalized coordinates to pixel coordinates.
    4. Fills a blank mask with each polygon using the instance ID as pixel value.
    5. Saves the resulting single-channel mask as a PNG in `mask_dir`.

    Notes:
    ------
    - Only labels with polygons (length > 5) are processed.
    - Background is 0; instance IDs are used as pixel values.
    - Handles class_instance IDs in the format `class_inst` or just `class`.
    - Useful for preparing segmentation masks for training or visualization.
    """

    os.makedirs(mask_dir, exist_ok=True)

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        img_name = os.path.splitext(label_file)[0]
        label_path = os.path.join(label_dir, label_file)
        image_path = None

        for ext in [".jpg", ".png", ".jpeg"]:
            possible_path = os.path.join(image_dir, img_name + ext)
            if os.path.exists(possible_path):
                image_path = possible_path
                break

        if image_path is None:
            print(f"⚠ No image found for {label_file}, skipping.")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠ Could not read {image_path}, skipping.")
            continue
        h, w = image.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) <= 5:
                continue  # skip non-segmentations

            try:
                class_instance = parts[0]
                if "_" in class_instance:
                    cls_str, inst_str = class_instance.split("_")
                    cls_id = int(cls_str)
                    inst_str = int(inst_str)
                else:
                    cls_id = int(float(class_instance))
            except:
                print(f"⚠ Invalid class_instance format in {label_file}: {parts[0]}")
                continue

            poly_coords = list(map(float, parts[5:]))

            pts = np.array([
                [int(poly_coords[i] * w), int(poly_coords[i + 1] * h)]
                for i in range(0, len(poly_coords), 2)
            ], dtype=np.int32)

            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], color=inst_str)

        mask_path = os.path.join(mask_dir, img_name + ".png")
        cv2.imwrite(mask_path, mask)
        print(f"✅ Mask saved: {mask_path}")
