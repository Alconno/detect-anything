import os
import cv2
import albumentations as A
import numpy as np
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
from utility.cantor_pair import pairing_function, reverse_pairing_function

import yaml
with open("./my_dataset/data.yaml", "r") as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg["names"]

image_dir = "./my_dataset/images/train"
mask_dir = "./my_dataset/masks/train"
label_dir = "./my_dataset/labels/train"

label_files = [f for f in os.listdir(label_dir) if f.lower().endswith((".txt"))]


def segment_aug(strategy_map, aug_transform, num_augs=5):
    """
    Applies Albumentations augmentations to training images and their corresponding 
    instance masks, generating multiple augmented image-mask-label sets for segmentation.

    Each image is paired with:
      - A combined mask representing all object instances
      - YOLO-format labels that include bounding boxes and polygons for each instance

    Parameters:
    ----------
    strategy_map : dict
        Maps class names to augmentation strategies (0 = full mask, 1 = per-instance).

    aug_transform : albumentations.core.composition.Compose
        Albumentations Compose object defining image and mask augmentations.

    num_augs : int, default=5
        Number of augmented versions to generate per image.

    Notes:
    ------
    - Bounding boxes and polygon coordinates are recalculated to match the augmented masks.
    - Images are resized to their original dimensions after augmentation.
    - Instances without valid labels are discarded to avoid corrupt data.
    - All augmented files are saved in the corresponding dataset folders.
    """
    
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, os.path.splitext(filename)[0] + ".png")

        if not os.path.exists(mask_path):
            print(f"⚠ Missing mask: {mask_path}")
            continue

        # Map class and instance ids
        instance_class_map = {}
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                raw_id = parts[0]

                if "_" in raw_id:
                    cls_id_str, inst_id_str = raw_id.split("_", 1)
                    cls_id = int(cls_id_str)
                    inst_id = int(inst_id_str)
                    paired_id = pairing_function(cls_id, inst_id)

                    instance_class_map[paired_id] = cls_id
                else:
                    instance_class_map = None
                    break

        if instance_class_map is None or instance_class_map == {}:
            continue


        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if image is None or mask is None:
            continue

        h, w = image.shape[:2]
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]  # exclude background

        # Prepare list of binary masks (one per instance)
        list_masks = [(mask == obj_id).astype(np.uint8) for obj_id in instance_ids]

        # Build additional_targets dict for all masks
        additional_targets = {f"mask{i}": "mask" for i in range(len(list_masks))}

        transform = A.Compose(
            aug_transform.transforms + [
                A.Resize(height=h, width=w, always_apply=True),
            ],
            additional_targets=additional_targets
        )

        """
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.15,
                rotate_limit=30,
                border_mode=cv2.BORDER_CONSTANT,
                interpolation=cv2.INTER_NEAREST,
                p=0.6
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.1,
                p=0.3
            ),
            # Commenting out HueSaturationValue to avoid color shift
            # A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0, p=0.0),
            A.Resize(height=h, width=w, always_apply=True),
        ], additional_targets=additional_targets)
        """

        for i_aug in range(num_augs):
            inputs = {"image": image}
            for i, msk in enumerate(list_masks):
                inputs[f"mask{i}"] = msk

            augmented = transform(**inputs)
            aug_img = augmented["image"]

            # Rebuild combined mask from augmented instance masks
            aug_mask = np.zeros((h, w), dtype=np.uint8)
            for i, obj_id in enumerate(instance_ids):
                cls_id = instance_class_map.get(obj_id, None)
                if cls_id is None:
                    print(f"⚠ Warning: cls_id not found for obj_id {obj_id}, skipping this instance")
                    continue
                
                class_name = class_names[cls_id]

                if class_name not in strategy_map:
                    print(f"⚠ Warning: class_name '{class_name}' not in strategy_map keys {list(strategy_map.keys())}, skipping")
                    continue

                single_aug_mask = augmented[f"mask{i}"]
                if single_aug_mask.ndim == 3:
                    single_aug_mask = single_aug_mask[:, :, 0]
                single_aug_mask = (single_aug_mask > 0).astype(np.uint8)

                if strategy_map[class_name] == 0:
                    aug_mask[single_aug_mask == 1] = obj_id
                elif strategy_map[class_name] == 1:
                    min_area = 100
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(single_aug_mask)

                    for label_id in range(1, num_labels):  # skip background
                        area = stats[label_id, cv2.CC_STAT_AREA]
                        if area < min_area:
                            continue

                        instance_region = (labels == label_id).astype(np.uint8)
                        aug_mask[instance_region == 1] = obj_id

            # Save augmented image
            base_name = os.path.splitext(filename)[0]
            new_img_name = f"{base_name}_aug{i_aug+1}.png"
            cv2.imwrite(os.path.join(image_dir, new_img_name), aug_img)

            # Save augmented mask
            new_mask_name = f"{base_name}_aug{i_aug+1}.png"
            cv2.imwrite(os.path.join(mask_dir, new_mask_name), aug_mask)

            # Save labels
            label_path = os.path.join(label_dir, f"{base_name}_aug{i_aug+1}.txt")
            label_lines_written = 0
            with open(label_path, "w") as f:
                for id_pair in np.unique(aug_mask):
                    cls_id, inst_id = reverse_pairing_function(id_pair)
                    if inst_id == 0:  # bg
                        continue

                    instance_mask = (aug_mask == id_pair).astype(np.uint8)
                    contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if len(cnt) < 3:
                            continue
                        poly = []
                        x_min, y_min = float("inf"), float("inf")
                        x_max, y_max = float("-inf"), float("-inf")
                        for p in cnt:
                            x, y = p[0]
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
                        f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f} " +
                                " ".join(f"{p:.6f}" for p in poly) + "\n")
                        label_lines_written += 1

            if label_lines_written == 0:
                # No labels written, remove all augmented files
                os.remove(label_path)
                os.remove(os.path.join(image_dir, new_img_name))
                os.remove(os.path.join(mask_dir, new_mask_name))
                print(f"⚠️ No labels generated, deleted {new_img_name}, {new_mask_name}, and label file.")
            else:
                print(f"✅ Saved augmentation {new_img_name}, {new_mask_name}, labels")


from multiprocessing import freeze_support

def main():
    freeze_support()


    full_strategy_map = {cls: 1 for cls in class_names}
    segment_aug(full_strategy_map)

if __name__ == '__main__':
    main()
