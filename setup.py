import os
import subprocess
import urllib.request

from utility.ensure_dataset_structure import ensure_dataset_structure

# Step 1: Ensure dataset folders
ensure_dataset_structure("my_dataset")


# Step 2: Ensure SAM repository exists
SAM_REPO_URL = "https://github.com/facebookresearch/segment-anything.git"
SEGMENTATION_DIR = os.path.join("segmentation", "segment_anything")

if not os.path.exists(SEGMENTATION_DIR):
    print(f"📦 Cloning Segment Anything repository into {SEGMENTATION_DIR} ...")
    os.makedirs(os.path.dirname(SEGMENTATION_DIR), exist_ok=True)
    subprocess.run(["git", "clone", SAM_REPO_URL, SEGMENTATION_DIR])
else:
    print(f"✅ Segment Anything repo already exists at {SEGMENTATION_DIR}")


# Step 3: Ensure SAM model weights exist
SAM_MODEL_PATH = os.path.join("segmentation", "sam_vit_h_4b8939.pth")
SAM_MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

if not os.path.exists(SAM_MODEL_PATH):
    print(f"📥 Downloading SAM model weights to {SAM_MODEL_PATH} ...")
    os.makedirs(os.path.dirname(SAM_MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(SAM_MODEL_URL, SAM_MODEL_PATH)
    print(f"✅ Downloaded SAM model weights.")
else:
    print(f"✅ SAM model weights already exist at {SAM_MODEL_PATH}")
