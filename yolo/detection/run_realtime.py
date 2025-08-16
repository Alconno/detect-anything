import cv2
import numpy as np
import time
from ultralytics import YOLO
import mss

# Load YOLOv8 model
model = YOLO('runs/detect/train/weights/best.pt')

# Define screen region to capture (left, top, width, height)
screen_region = {'top': 450, 'left': 800, 'width': 960, 'height': 540} # middle for 2k

# Use MSS for fast screen grabbing
sct = mss.mss()

while True:
    start_time = time.time()

    # Grab screenshot
    screen = np.array(sct.grab(screen_region))

    # Convert BGRA to RGB
    frame_rgb = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)

    # Run YOLO inference
    results = model(frame_rgb[..., ::-1], imgsz=960, conf=0.35)  # BGR → RGB manually


    # Get image with boxes
    img_with_boxes = results[0].plot()

    # Show the image (resize if needed)
    cv2.imshow("Real-Time YOLO Detection", img_with_boxes)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Print FPS
    fps = 1 / (time.time() - start_time)
    print(f"FPS: {fps:.2f}")
    time.sleep(0.001)

cv2.destroyAllWindows()
