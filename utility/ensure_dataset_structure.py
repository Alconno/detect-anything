import os

def ensure_dataset_structure(base_path="my_dataset"):
    """
    Ensure the dataset folder structure exists for images, labels, and masks.
    
    This will create the following folders if missing:
    - images/storage/train
    - images/storage/val
    - images/train
    - images/val
    - labels/storage/train
    - labels/storage/val
    - labels/train
    - labels/val
    - masks/train
    - masks/val
    
    Class-specific folders (fire, human, etc.) are NOT created.
    """
    folders_to_ensure = [
        # Images
        os.path.join(base_path, "images", "storage", "train"),
        os.path.join(base_path, "images", "storage", "val"),
        os.path.join(base_path, "images", "train"),
        os.path.join(base_path, "images", "val"),
        
        # Labels
        os.path.join(base_path, "labels", "storage", "train"),
        os.path.join(base_path, "labels", "storage", "val"),
        os.path.join(base_path, "labels", "train"),
        os.path.join(base_path, "labels", "val"),
        
        # Masks
        os.path.join(base_path, "masks", "train"),
        os.path.join(base_path, "masks", "val"),
    ]

    for folder in folders_to_ensure:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Ensured folder exists: {folder}")