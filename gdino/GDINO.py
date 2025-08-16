import os
import cv2
import argparse
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology


def main(image_dir, output_dir, label_dir):
    # Define the ontology
    ontology = CaptionOntology({
        "orange fire": "fire",
    })

    # Load the model
    model = GroundingDINO(ontology=ontology)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    class_names = {0: "fire"}

    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(image_dir, filename)
            print(f"Processing {img_path}")

            results = model.predict(img_path)
            boxes = results.xyxy
            confidences = results.confidence
            class_ids = results.class_id

            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue

            # Draw boxes
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = f"{class_names.get(cls_id, 'fire')} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save annotated image
            out_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
            cv2.imwrite(out_path, img)

            # Save YOLO-format label
            h, w = img.shape[:2]
            yolo_txt_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
            with open(yolo_txt_path, "w") as f:
                for box, cls_id in zip(boxes, class_ids):
                    x1, y1, x2, y2 = box
                    xc = ((x1 + x2) / 2) / w
                    yc = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print("\n✅ YOLO-format labels saved.")
    print(f"✅ All images processed and saved to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Grounding DINO and save YOLO labels + annotated images.")
    parser.add_argument("image_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory to save annotated images")
    parser.add_argument("label_dir", help="Directory to save YOLO-format labels")

    args = parser.parse_args()
    main(args.image_dir, args.output_dir, args.label_dir)
