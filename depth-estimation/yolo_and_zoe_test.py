import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline

# Load your image
image_path = "./my_dataset/images/storage/train/rhino/rhino_00000.png"
image_pil = Image.open(image_path).convert("RGB")
image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Load YOLOv8 model
model = YOLO('runs/segment/train/weights/best.pt')
results = model(image_cv2)[0]

# Load depth estimation pipeline
depth_estimator = pipeline(task="depth-estimation", model="Intel/zoedepth-nyu-kitti")
depth_result = depth_estimator(image_pil)
depth_pil = depth_result['depth']
depth_np = np.array(depth_pil)

# Normalize depth map for display
depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255.0
depth_uint8 = depth_norm.astype(np.uint8)
depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

# Estimate depth for each detected object
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])

    # Extract bottom part of bounding box from depth map
    box_height = y2 - y1
    bottom_start = y2 - int(0.2 * box_height)
    bottom_start = max(bottom_start, 0)
    x1 = max(x1, 0)
    x2 = min(x2, depth_np.shape[1])
    y2 = min(y2, depth_np.shape[0])
    bottom_depth = depth_np[bottom_start:y2, x1:x2]
    avg_depth = np.mean(bottom_depth)

    # Draw bounding box and depth
    cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{model.names[cls_id]} {conf:.2f} | depth: {avg_depth:.1f}"
    cv2.putText(image_cv2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# Show results
cv2.imshow("Object Detection", image_cv2)
cv2.imshow("Depth Map", depth_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()

