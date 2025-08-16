import os
import cv2
import albumentations as A

# ==== Paths ====
image_dir = "./my_dataset/images/train"
label_dir = "./my_dataset/labels/train"
 

# ==== Augment ====
def box_aug(aug_transform, num_augs=5):
    """
    Applies a given Albumentations augmentation pipeline to all images 
    and their YOLO-format bounding boxes, generating multiple 
    augmented image-label pairs.

    Parameters:
    ----------
    aug_transform : albumentations.core.composition.Compose
        An Albumentations Compose object defining the augmentation pipeline.
    
    num_augs : int, default=5
        Number of augmented versions to generate per original image.

    Notes:
    ------
    - Bounding boxes remain consistent with augmentations.
    - All saved boxes are in YOLO normalized format.
    - Images are not resized; original size is preserved.
    - Augmentations causing bbox errors are skipped.
    """
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

        if not os.path.exists(label_path):
            print(f"⚠ Missing label: {label_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠ Could not read {image_path}")
            continue
        h,w = image.shape[:2]

        # Read YOLO bboxes (ignore polygon coords if present)
        bboxes, class_labels = [], []
        with open(label_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 5:
                    print(f"⚠ Skipping malformed line in {label_path}: {line.strip()}")
                    continue
                cls_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])  # only first 4 numbers after cls_id
                bboxes.append([xc, yc, bw, bh])
                class_labels.append(cls_id)

        if not bboxes:
            print(f"⚠ No valid boxes in {label_path}")
            continue

        transform = A.Compose(
            aug_transform.transforms,
            bbox_params=A.BboxParams(
                format='yolo', 
                label_fields=['class_labels'],
            )
        )


        # Augment multiple times
        for i in range(num_augs):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            except Exception as e:
                print(f"⚠ Skipping {filename} due to augmentation error: {e}")
                continue

            aug_img = augmented['image']
            base_name = os.path.splitext(filename)[0]
            new_filename = f"{base_name}_aug{i+1}.png"

            # Save augmented image
            cv2.imwrite(os.path.join(image_dir, new_filename), aug_img)

            # Save augmented labels (YOLO format)
            with open(os.path.join(label_dir, os.path.splitext(new_filename)[0] + ".txt"), 'w') as f:
                for bbox, cls_id in zip(augmented['bboxes'], augmented['class_labels']):
                    x, y, bw, bh = bbox
                    f.write(f"{str(int(cls_id))} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

            print(f"✅ Augmented: {new_filename}")

from multiprocessing import freeze_support

def main():
    freeze_support()
    
    box_aug()

if __name__ == '__main__':
    main()
