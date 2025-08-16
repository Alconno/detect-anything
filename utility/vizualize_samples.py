import os
import cv2
import numpy as np
import yaml

image_dir = "./my_dataset/images/train"
label_dir = "./my_dataset/labels/train"
with open("./my_dataset/data.yaml", "r") as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg["names"]

def display_all_samples_cv2():
    """
    Display all images in the dataset with their corresponding YOLO-format labels.

    - Reads images from `image_dir` and label files from `label_dir`.
    - Supports both bounding boxes and polygonal segmentations.
    - Bounding boxes are drawn in green.
    - Polygons are drawn as filled red overlays with transparency.
    - If labels are missing or malformed, warnings are printed.
    - Press `ESC` to stop previewing before finishing all samples.

    Returns:
        None
    """

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + ".txt")

        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load {image_path}")
            continue
        h, w = img.shape[:2]

        if not os.path.exists(label_path):
            print(f"⚠️ No label found for {image_name}")
            continue

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                if "_" in parts[0]:
                    class_id, instance_id = parts[0].split("_",1)
                    class_id = int(float(class_id))
                    instance_id = int(float(instance_id))
                else:
                    class_id = int(float(parts[0]))
                    instance_id = None

                label_text = class_names[class_id] if class_id < len(class_names) else str(class_id)

                if len(parts) == 5:  
                    # BBOX only
                    xc, yc, bw, bh = map(float, parts[1:])
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                elif len(parts) > 5:  
                    # Polygon segmentation (with or without bbox first)
                    coords = list(map(float, parts[1:]))

                    if len(coords) % 2 != 0:
                        print(f"⚠️ Malformed polygon in {label_path}: {line.strip()}")
                        continue

                    # If first 4 coords look like bbox, treat them as bbox first
                    has_bbox = len(coords) >= 4 and all(0 <= c <= 1 for c in coords[:4])
                    if has_bbox:
                        xc, yc, bw, bh = coords[:4]
                        poly_coords = coords[4:]
                        x1 = int((xc - bw / 2) * w)
                        y1 = int((yc - bh / 2) * h)
                        x2 = int((xc + bw / 2) * w)
                        y2 = int((yc + bh / 2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, label_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        poly_coords = coords

                    # Draw polygon
                    pts = []
                    for i in range(0, len(poly_coords), 2):
                        x = int(poly_coords[i] * w)
                        y = int(poly_coords[i + 1] * h)
                        pts.append([x, y])
                    pts = np.array(pts, np.int32)
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [pts], color=(0,0,255))
                    alpha = 0.25
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                    #cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                    #cv2.putText(img, label_text, (pts[0][0], pts[0][1] - 10),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("YOLO Preview", img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to break early
            break

    cv2.destroyAllWindows()





def draw_labels_on_image(image_path, label_path, class_names):
    """
    Draw YOLO annotations on a single image.

    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the YOLO label file.
        class_names (list[str]): List of class names from dataset config.

    - Handles both bounding boxes and polygon segmentation.
    - Bounding boxes are drawn in green with class labels.
    - Polygons are filled red with transparency overlay.
    - Skips malformed or missing labels with a warning.

    Returns:
        np.ndarray: The image with labels drawn (in RGB format).
    """

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load {image_path}")
        return
    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"⚠️ No label found for {label_path}")
        return

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if "_" in parts[0]:
                class_id, instance_id = parts[0].split("_",1)
                class_id = int(float(class_id))
                instance_id = int(float(instance_id))
            else:
                class_id = int(float(parts[0]))
                instance_id = None

            label_text = class_names[class_id] if class_id < len(class_names) else str(class_id)

            if len(parts) == 5:  
                # BBOX only
                xc, yc, bw, bh = map(float, parts[1:])
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            elif len(parts) > 5:  
                # Polygon segmentation (with or without bbox first)
                coords = list(map(float, parts[1:]))

                if len(coords) % 2 != 0:
                    print(f"⚠️ Malformed polygon in {label_path}: {line.strip()}")
                    continue

                # If first 4 coords look like bbox, treat them as bbox first
                has_bbox = len(coords) >= 4 and all(0 <= c <= 1 for c in coords[:4])
                if has_bbox:
                    xc, yc, bw, bh = coords[:4]
                    poly_coords = coords[4:]
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    poly_coords = coords

                # Draw polygon
                pts = []
                for i in range(0, len(poly_coords), 2):
                    x = int(poly_coords[i] * w)
                    y = int(poly_coords[i + 1] * h)
                    pts.append([x, y])
                pts = np.array(pts, np.int32)
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color=(0,0,255))
                alpha = 0.25
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)




def main():
    import multiprocessing, json, albumentations as A, cv2
    multiprocessing.freeze_support()

    display_all_samples_cv2()


if __name__ == "__main__":
    main()