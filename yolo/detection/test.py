from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/train/weights/best.pt')

results = model('./fires12344.png', imgsz=960, conf=0.5)

results[0].show()  # optional: display image window

img_with_boxes = results[0].plot()
cv2.imwrite('output.jpg', cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
