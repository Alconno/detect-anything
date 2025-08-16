import os
import cv2
import numpy as np
import torch
import sys
import shutil
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
from utility.cantor_pair import pairing_function





sys.path.append("./segmentation")
from labeling_strats import mask_to_yolo_polygon

# SAM imports
sys.path.append("./segmentation/segment_anything")
from segment_anything import sam_model_registry, SamPredictor




# ==== Paths ====
img_dirs = ["./my_dataset/images/train", "./my_dataset/images/val"]
label_dirs = ["./my_dataset/labels/train", "./my_dataset/labels/val"]
mask_dirs = ["./my_dataset/masks/train", "./my_dataset/masks/val"]

for mask_dir in mask_dirs:
    os.makedirs(mask_dir, exist_ok=True)



# ==== Load SAM ====
checkpoint_path = "./segmentation/sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)



# ==== Helpers ====
def yolo_to_xyxy(box_line, img_w, img_h):
    """Convert YOLO normalized bbox to pixel xyxy"""
    cls, x, y, w, h = map(float, box_line.strip().split())
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return int(cls), [x1, y1, x2, y2]


def sort_clockwise(points):
    """
    Sorts polygon points clockwise around centroid.
    Input: list of (x, y)
    Output: list of (x, y)
    """
    pts = np.array(points)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_idx = np.argsort(angles)
    return pts[sorted_idx].tolist()






def segment_boxes(strategy_map):
    """
    Generates instance segmentation masks and polygon labels from YOLO 
    bounding boxes using the Segment Anything Model (SAM).

    For each image:
    1. Loads the image and corresponding YOLO label file.
    2. Skips images whose labels already contain polygon data.
    3. Converts YOLO normalized boxes to pixel coordinates.
    4. Uses SAM to predict masks for each bounding box.
    5. Filters and merges predicted masks, retaining only high-confidence regions.
    6. Converts masks to polygons, sorts points clockwise, and recalculates 
       bounding boxes in YOLO format (x_center, y_center, width, height).
    7. Saves:
       - Combined mask image for all instances
       - Updated label file with instance IDs and polygon coordinates
    8. Fills in missing or invalid masks with the original bounding box.

    Parameters:
    ----------
    strategy_map : dict
        Maps class names or IDs to segmentation strategies or post-processing 
        behaviors for polygons and masks.

    Notes:
    ------
    - Each instance is assigned a unique ID using a pairing function.
    - Polygons are normalized relative to image dimensions.
    - Images without valid masks fallback to their original bounding boxes.
    - Progress is printed to the console as a simple progress bar.
    """
    
    # ==== Main Loop ====
    for img_dir, label_dir, mask_dir in zip(img_dirs, label_dirs, mask_dirs):
        img_names = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        total = len(img_names)

        for i, img_name in enumerate(img_names):
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
            mask_path = os.path.join(mask_dir, os.path.splitext(img_name)[0] + ".png")

            if not os.path.isfile(label_path):
                continue

            # Skip if label file already contains polygons
            already_converted = False
            with open(label_path, "r") as f:
                for line in f:
                    if len(line.strip().split()) != 5:
                        already_converted = True
                        break
            if already_converted:
                continue

            # Load image
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            predictor.set_image(image_rgb)

            combined_mask = np.zeros((h, w), dtype=np.uint8)
            output_polygons = []

            with open(label_path, "r") as f:
                lines = f.readlines()

            instance_id = 1

            for line in lines:
                cls_id, box = yolo_to_xyxy(line, w, h)
                box_np = np.array(box, dtype=np.float32)
                x1, y1, x2, y2 = map(int, box)

                masks, scores, logits = predictor.predict(
                    box=box_np,
                    multimask_output=True
                )

                score_thresh = 0.35
                good_masks = [m for m, s in zip(masks, scores) if s >= score_thresh]

                if not good_masks:
                    best_idx = int(np.argmax(scores))
                    good_masks = [masks[best_idx]]

                if len(good_masks) == 0:
                    rect_cnt = np.array([
                        [[x1, y1]],
                        [[x2, y1]],
                        [[x2, y2]],
                        [[x1, y2]]
                    ], dtype=np.int32)
                    output_polygons.append((cls_id, rect_cnt))
                    combined_mask[y1:y2, x1:x2] = cls_id + 1
                    continue

                merged_mask = np.any(good_masks, axis=0)

                largest_contour = mask_to_yolo_polygon(merged_mask, cls_id, strategy_map, simplification_factor=0.0)

                cv2.drawContours(combined_mask, [largest_contour], -1, color=pairing_function(cls_id, instance_id), thickness=-1)
                

                if largest_contour is None or cv2.contourArea(largest_contour) < 5:
                    rect_cnt = np.array([
                        [[x1, y1]],
                        [[x2, y1]],
                        [[x2, y2]],
                        [[x1, y2]]
                    ], dtype=np.int32)
                    output_polygons.append((cls_id, rect_cnt))
                else:
                    output_polygons.append((cls_id, instance_id, largest_contour))
                instance_id += 1

            cv2.imwrite(mask_path, combined_mask.astype(np.uint8))


            with open(label_path, "w") as f:
                for cls_id, instance_id, cnt in output_polygons:
                    points = [(point[0][0], point[0][1]) for point in cnt]
                    sorted_points = sort_clockwise(points)

                    poly = []
                    x_min, y_min = float("inf"), float("inf")
                    x_max, y_max = float("-inf"), float("-inf")

                    for x, y in sorted_points:
                        poly.append(x / w)
                        poly.append(y / h)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)

                    x_center = ((x_min + x_max) / 2) / w
                    y_center = ((y_min + y_max) / 2) / h
                    width = (x_max - x_min) / w
                    height = (y_max - y_min) / h

                    f.write(f"{cls_id}_{instance_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} " +
                            " ".join(f"{p:.6f}" for p in poly) + "\n")


            # Update progress bar (one line)
            progress = (i + 1) / total
            bar_length = 20
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            percent = int(progress * 100)
            sys.stdout.write(f"\rProgress: |{bar}| {percent}% ({i+1}/{total})")
            sys.stdout.flush()

        print()




from multiprocessing import freeze_support

def main():
    freeze_support()
    
    segment_boxes()

if __name__ == '__main__':
    main()
