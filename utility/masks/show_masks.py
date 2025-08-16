import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

img_dirs = ["./my_dataset/images/train", "./my_dataset/images/val"]
mask_dirs = ["./my_dataset/masks/train", "./my_dataset/masks/val"]

# Base palette (index 0 = background)
base_colors = [
    [0, 0, 0],          # background
    [255, 0, 0],        # red
    [0, 255, 0],        # green
    [0, 0, 255],        # blue
    [255, 255, 0],      # yellow
    [255, 0, 255],      # magenta
    [0, 255, 255],      # cyan
    [128, 0, 0],        # maroon
    [0, 128, 0],        # dark green
    [0, 0, 128],        # navy
    [128, 128, 0],      # olive
    [128, 0, 128],      # purple
    [0, 128, 128],      # teal
    [192, 192, 192],    # silver
    [128, 128, 128],    # gray
    [255, 165, 0],      # orange
    [255, 192, 203],    # pink
    [173, 216, 230],    # light blue
    [0, 255, 127],      # spring green
    [255, 20, 147],     # deep pink
    [75, 0, 130],       # indigo
    [240, 230, 140],    # khaki
    [0, 191, 255],      # deep sky blue
    [139, 69, 19],      # saddle brown
    [255, 140, 0],      # dark orange
    [144, 238, 144],    # light green
    [220, 20, 60],      # crimson
    [0, 255, 255],      # aqua
    [0, 100, 0],        # dark green
    [210, 105, 30],     # chocolate
]

for img_dir, mask_dir in zip(img_dirs, mask_dirs):
    """
    Displays RGB images overlaid with their instance segmentation masks.

    For each mask file:
    1. Loads the corresponding image and grayscale mask.
    2. Assigns a distinct color to each unique instance ID:
    - Uses a fixed palette for the first IDs.
    - Generates random colors for higher IDs.
    3. Blends the colored mask with the original image using alpha transparency.
    4. Displays the overlay using matplotlib with no axes.

    Notes:
    ------
    - Background (mask ID 0) is ignored.
    - Overlay transparency is set by `alpha = 0.5`.
    - Random colors are deterministic per instance ID (same ID → same color).
    - Useful for visually inspecting segmentation masks.
    """

    for filename in sorted(os.listdir(mask_dir)):
        if not filename.lower().endswith(".png"):
            continue

        img_path = os.path.join(img_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        if not os.path.isfile(img_path):
            print(f"⚠️ Image not found for mask: {filename}, skipping...")
            continue

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        overlay = image.copy()
        alpha = 0.5

        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids != 0]

        id_to_color = {}
        for uid in unique_ids:
            if uid < len(base_colors):
                id_to_color[uid] = base_colors[uid]
            else:
                random.seed(int(uid))  # ensure it's a Python int
                id_to_color[uid] = [random.randint(0, 255) for _ in range(3)]

        for uid, color in id_to_color.items():
            class_mask = (mask == uid)
            overlay[class_mask] = (
                (1 - alpha) * overlay[class_mask] + alpha * np.array(color)
            ).astype(np.uint8)

        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.title(f"Image and Mask Overlay: {filename} ({os.path.basename(img_dir)})")
        plt.axis("off")
        plt.show()
