import os
import re

images_dir = r".\my_dataset\images\storage\train\human"
labels_dir = r".\my_dataset\labels\storage\train\human"

prefix = "human_"
image_ext = None
label_ext = None

def get_extension(files, exts):
    """Return the first matching extension from exts found in files."""
    for ext in exts:
        if any(f.endswith(ext) for f in files):
            return ext
    return None

def get_number(filename, prefix):
    """Extract the number from filename like human_00001.jpg or .txt."""
    pattern = re.escape(prefix) + r"(\d+)"
    m = re.search(pattern, filename)
    if m:
        return int(m.group(1))
    else:
        return None

def main():
    """
    Cleans up and renames image-label pairs in a dataset folder.

    Steps:
    ------
    1. Detects the image and label file extensions if not specified.
    2. Deletes images that do not have a matching label file.
    3. Sorts remaining images and labels by the numeric part of their filename.
    4. Renames files with continuous numbering using the specified prefix
    (e.g., human_00001.jpg, human_00001.txt).

    Notes:
    ------
    - Only files starting with the given prefix are processed.
    - Works with common image formats (.jpg, .jpeg, .png) and .txt labels.
    - Ensures that every image has a corresponding label and the numbering is sequential.
    """
    global image_ext, label_ext

    images = os.listdir(images_dir)
    labels = os.listdir(labels_dir)

    # Detect image and label extensions dynamically if not set
    image_ext = get_extension(images, ['.jpg', '.jpeg', '.png'])
    label_ext = get_extension(labels, ['.txt'])

    if image_ext is None:
        print("No images found with .jpg, .jpeg, or .png extensions in images folder!")
        return
    if label_ext is None:
        print("No label files found with .txt extension in labels folder!")
        return

    print(f"Using image extension: {image_ext}")
    print(f"Using label extension: {label_ext}")

    # Filter only relevant files with correct extensions and prefix
    images = [f for f in images if f.endswith(image_ext) and f.startswith(prefix)]
    labels = [f for f in labels if f.endswith(label_ext) and f.startswith(prefix)]

    labels_bases = set(os.path.splitext(f)[0] for f in labels)

    # Step 1: Delete images without matching labels
    deleted_count = 0
    for img_file in images:
        base = os.path.splitext(img_file)[0]
        if base not in labels_bases:
            img_path = os.path.join(images_dir, img_file)
            os.remove(img_path)
            deleted_count += 1
            print(f"Deleted image without label: {img_file}")

    print(f"\nDeleted {deleted_count} images without labels.")

    # Step 2: Refresh list of images and labels after deletion
    images = [f for f in os.listdir(images_dir) if f.endswith(image_ext) and f.startswith(prefix)]
    labels = [f for f in os.listdir(labels_dir) if f.endswith(label_ext) and f.startswith(prefix)]

    # Sort files by their number part
    images_sorted = sorted(images, key=lambda f: get_number(f, prefix))
    labels_sorted = sorted(labels, key=lambda f: get_number(f, prefix))

    # Step 3: Rename files with continuous numbering
    for new_idx, (img_file, lbl_file) in enumerate(zip(images_sorted, labels_sorted)):
        new_num_str = f"{new_idx:05d}"
        new_img_name = f"{prefix}{new_num_str}{image_ext}"
        new_lbl_name = f"{prefix}{new_num_str}{label_ext}"

        old_img_path = os.path.join(images_dir, img_file)
        new_img_path = os.path.join(images_dir, new_img_name)

        if old_img_path != new_img_path:
            os.rename(old_img_path, new_img_path)
            print(f"Renamed image: {img_file} -> {new_img_name}")

        old_lbl_path = os.path.join(labels_dir, lbl_file)
        new_lbl_path = os.path.join(labels_dir, new_lbl_name)

        if old_lbl_path != new_lbl_path:
            os.rename(old_lbl_path, new_lbl_path)
            print(f"Renamed label: {lbl_file} -> {new_lbl_name}")

    print("\n✅ Cleanup and renaming complete.")

if __name__ == "__main__":
    main()
