from transformers import pipeline
from PIL import Image
import requests
import numpy as np

# Load depth estimation pipeline
depth_estimator = pipeline(task="depth-estimation", model="Intel/zoedepth-nyu-kitti")

# Load image (from URL or local)
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

image = "./fires12344.png"

# Run inference
outputs = depth_estimator(image)

# Access depth map (PIL Image)
depth = outputs['depth']  # or outputs['predicted_depth'] depending on model

# Convert to displayable format
depth_np = np.array(depth)
depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255.0
depth_uint8 = depth_normalized.astype(np.uint8)

# Visualize using OpenCV
import cv2
depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
cv2.imshow("Depth Map", depth_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()
