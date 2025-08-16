import cv2
import numpy as np
import os

mask_dirs = ["./my_dataset/masks/train", "./my_dataset/masks/val"]
label_dirs = ["./my_dataset/labels/train", "./my_dataset/labels/val"]

for mask_dir, label_dir in zip(mask_dirs, label_dirs):
    """
    Converts instance segmentation masks into YOLO-format polygon labels.

    For each mask file:
    1. Loads the mask as grayscale.
    2. Finds connected components (each unique instance).
    3. Extracts the external contour for each instance.
    4. Normalizes contour coordinates to [0,1] relative to image size.
    5. Computes the bounding box (x_center, y_center, width, height) in YOLO format.
    6. Writes a label file with one line per instance:
    `class_id x_center y_center width height x1 y1 x2 y2 ...`  

    Notes:
    ------
    - Background is assumed to be 0.
    - Class ID is set to 0 by default.
    - Each instance gets its own polygon.
    - Skips masks without valid contours.
    """

    os.makedirs(label_dir, exist_ok=True)

    for mask_file in os.listdir(mask_dir):
        if not mask_file.endswith(".png"):
            continue
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        num_labels, labels = cv2.connectedComponents(mask)

        h, w = mask.shape
        label_lines = []

        for i in range(1, num_labels):  # skip background (label 0)
            component_mask = (labels == i).astype(np.uint8) * 255

            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = contours[0]

            poly = []
            for point in cnt:
                x, y = point[0]
                poly.append(x / w)
                poly.append(y / h)

            x, y, bw, bh = cv2.boundingRect(cnt)
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            box_w = bw / w
            box_h = bh / h

            class_id = 0
            label_lines.append(f"{class_id} {x_center} {y_center} {box_w} {box_h} " + " ".join(map(str, poly)))

        with open(os.path.join(label_dir, mask_file.replace(".png", ".txt")), "w") as f:
            f.write("\n".join(label_lines))
