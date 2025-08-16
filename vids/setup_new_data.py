#!/usr/bin/env python3
import os
import sys
import shutil
import random
import argparse
import re
from pathlib import Path
import uuid

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
LABEL_EXT = '.txt'
IMAGES_STORAGE = Path("./my_dataset/images/storage")
LABELS_STORAGE = Path("./my_dataset/labels/storage")
TRAIN_SPLIT = 0.7
PAD = 5  # zero pad width


def natural_key(s):
    """
    natural_key(s) -> tuple

    Purpose:
        Provide a sorting key that sorts strings containing integers in human-friendly
        (natural) order. For example, 'img2' comes before 'img10'.

    How it works:
        Splits the input string `s` into parts that are either digit sequences or
        non-digit text. Numeric substrings are converted to integers and returned
        as integers in the tuple; non-numeric substrings are lowercased strings.
        This tuple can then be used by Python's sorted(..., key=natural_key).

    Example:
        sorted(["img1", "img10", "img2"], key=natural_key) -> ['img1', 'img2', 'img10']
    """
    parts = re.split(r'(\d+)', s)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def discover_maps(folder: Path):
    """
    discover_maps(folder: Path) -> (images_map, labels_map)

    Purpose:
        Scan `folder` and return two dictionaries mapping basenames (stems) to
        actual filenames for images and labels respectively.

    Behavior / Notes:
        - Images are recognized by suffixes in IMAGE_EXTS (case-insensitive).
        - Labels are recognized by the extension LABEL_EXT ('.txt').
        - If multiple image files share the same stem but different extensions,
          the first encountered image extension for that stem is kept.
        - Only files (not directories) are considered.
        - The returned maps map 'stem' (filename without extension) -> 'filename with extension'.

    Returns:
        images_map (dict): e.g. {'0001': '0001.jpg', 'frame_1': 'frame_1.png', ...}
        labels_map (dict): e.g. {'0001': '0001.txt', 'frame_1': 'frame_1.txt', ...}
    """
    images_map = {}
    labels_map = {}
    for f in folder.iterdir():
        if not f.is_file():
            continue
        suf = f.suffix.lower()
        stem = f.stem
        if suf in IMAGE_EXTS:
            # prefer first image ext encountered for a basename
            if stem not in images_map:
                images_map[stem] = f.name
        elif suf == LABEL_EXT:
            if stem not in labels_map:
                labels_map[stem] = f.name
    return images_map, labels_map


def safe_two_phase_rename(folder: Path, pairs_ordered, object_name, dry_run=False):
    """
    safe_two_phase_rename(folder, pairs_ordered, object_name, dry_run=False)

    Purpose:
        Rename matched image/label pairs in `folder` into a sequential, collision-free
        naming scheme of the form:
            object_name_00000.ext
            object_name_00000.txt

        The function performs a two-phase rename to avoid filename collisions:
            1. Rename each original pair to a unique temporary name (__tmp_<uid>_<i>...).
            2. Rename all temporary names to the final sequential names.

    Parameters:
        folder (Path): Directory containing the matched image/label files.
        pairs_ordered (list): Ordered list of basenames (stems) to process. These
                              should correspond to keys in the global images_map and labels_map.
        object_name (str): Prefix for final filenames (e.g., 'humanoid').
        dry_run (bool): If True, only print planned operations without performing them.

    Important notes:
        - This function expects `images_map` and `labels_map` to be available in the
          module/global scope (the script sets them in main via `discover_maps` and
          declares them as global). It looks up filenames as `images_map[stem]` and
          `labels_map[stem]`.
        - Using a two-phase rename avoids accidental overwrites if the final target
          names already exist in the same folder.
        - Uses module-level PAD to determine zero-padding width for the sequence numbers.

    Example flow:
        safe_two_phase_rename(Path("./frames"), ['1','2','3'], 'humanoid')
    """
    uid = uuid.uuid4().hex[:8]
    tmp_names = []

    # Phase 1: rename to unique tmp names
    for idx, stem in enumerate(pairs_ordered):
        img_name = images_map[stem]
        lbl_name = labels_map[stem]
        img_ext = Path(img_name).suffix
        tmp_img = f"__tmp_{uid}_{idx}{img_ext}"
        tmp_lbl = f"__tmp_{uid}_{idx}.txt"
        src_img = folder / img_name
        src_lbl = folder / lbl_name
        print(f"[RENAME-TMP] {src_img.name} -> {tmp_img}")
        print(f"[RENAME-TMP] {src_lbl.name} -> {tmp_lbl}")
        if not dry_run:
            src_img.rename(folder / tmp_img)
            src_lbl.rename(folder / tmp_lbl)
        tmp_names.append((tmp_img, tmp_lbl))

    # Phase 2: rename tmp -> final sequential object_name_00000
    for idx, (tmp_img, tmp_lbl) in enumerate(tmp_names):
        new_idx = str(idx).zfill(PAD)
        tmp_img_path = folder / tmp_img
        tmp_lbl_path = folder / tmp_lbl
        final_img = f"{object_name}_{new_idx}{Path(tmp_img).suffix}"
        final_lbl = f"{object_name}_{new_idx}.txt"
        print(f"[RENAME-FINAL] {tmp_img} -> {final_img}")
        print(f"[RENAME-FINAL] {tmp_lbl} -> {final_lbl}")
        if not dry_run:
            tmp_img_path.rename(folder / final_img)
            tmp_lbl_path.rename(folder / final_lbl)


def move_and_reindex(src_folder: Path, pairs, dest_img_folder: Path, dest_lbl_folder: Path, object_name, dry_run=False):
    """
    move_and_reindex(src_folder, pairs, dest_img_folder, dest_lbl_folder, object_name, dry_run=False)

    Purpose:
        Copy (not move) image/label pairs from `src_folder` into destination folders,
        renaming them into a new zero-padded sequence based on `object_name`.

    Behavior / Steps:
        - Ensures destination directories exist (creates them if necessary).
        - For each (image_filename, label_filename) in `pairs`, produces:
            dest_img_folder/object_name_XXXXX<ext>
            dest_lbl_folder/object_name_XXXXX.txt
          where XXXXX is a zero-padded index (width determined by PAD).
        - Uses shutil.copy2 to copy files (copies metadata too).
        - Prints each operation; respects `dry_run` to only print without copying.

    Parameters:
        src_folder (Path): Directory where source files currently live.
        pairs (list of tuples): Each tuple is (image_filename, label_filename).
        dest_img_folder (Path): Destination folder for images.
        dest_lbl_folder (Path): Destination folder for label .txt files.
        object_name (str): Prefix for renamed files (e.g., 'humanoid').
        dry_run (bool): If True, only print planned copy operations without performing them.

    Important:
        - This function **copies** files (shutil.copy2) leaving the source files intact.
        - It will overwrite existing files in destination paths if they share the same name.
          If you want overwrite protection, add checks for dst_img.exists() / dst_lbl.exists()
          before copying and handle according to your policy.
    """
    dest_img_folder.mkdir(parents=True, exist_ok=True)
    dest_lbl_folder.mkdir(parents=True, exist_ok=True)

    for new_idx, (img_name, lbl_name) in enumerate(pairs):
        num = str(new_idx).zfill(PAD)
        new_img = f"{object_name}_{num}{Path(img_name).suffix}"
        new_lbl = f"{object_name}_{num}.txt"

        src_img = src_folder / img_name
        src_lbl = src_folder / lbl_name
        dst_img = dest_img_folder / new_img
        dst_lbl = dest_lbl_folder / new_lbl

        print(f"[COPY] {src_img.name} -> {dst_img}")
        print(f"[COPY] {src_lbl.name} -> {dst_lbl}")

        if not dry_run:
            shutil.copy2(str(src_img), str(dst_img))
            shutil.copy2(str(src_lbl), str(dst_lbl))


def main():
    """
    main()

    Purpose:
        Orchestrates the preparation of a folder of extracted video frames + labels
        into the `my_dataset` storage layout used by the rest of the pipeline.

    Command-line usage:
        python setup_new_data.py ./vids/humanoid_imgs [--dry-run] [--seed 42]

    Workflow:
        1. Validate the provided source folder path.
        2. Derive object_name from folder name (removes a trailing '_imgs' if present).
        3. Discover image/label filename maps within the source folder.
        4. Delete any images that do not have a corresponding label (orphan images).
        5. Re-discover maps and compute matched pairs (images & labels).
        6. Rename all matched pairs safely into a sequential naming scheme using
           safe_two_phase_rename to avoid collisions.
        7. After renaming, construct ordered lists of images and labels, pair them,
           shuffle and split into train/validation according to TRAIN_SPLIT.
        8. Copy (move_and_reindex) train & val sets into the storage folders:
           - ./my_dataset/images/storage/train/<object_name>/
           - ./my_dataset/images/storage/val/<object_name>/
           - ./my_dataset/labels/storage/train/<object_name>/
           - ./my_dataset/labels/storage/val/<object_name>/
        9. Print a final summary of where train/val images were copied.

    Notes:
        - The function uses a global `images_map` and `labels_map` name; `discover_maps`
          returns those maps and main declares them global so they are available to the
          renaming helper.
        - Use the --dry-run flag to preview actions before making any filesystem changes.
        - The function currently copies files to the destination (doesn't delete sources).
    """
    parser = argparse.ArgumentParser(description="Prepare vid frames folder into my_dataset storage (rename, split, move).")
    parser.add_argument("source_folder", help="Exact path to folder, e.g. ./vids/humanoid_imgs")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without deleting/moving files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    src = Path(args.source_folder).resolve()
    if not src.exists() or not src.is_dir():
        print(f"[ERROR] Folder does not exist: {src}")
        sys.exit(1)

    folder_name = src.name
    object_name = folder_name[:-5] if folder_name.endswith("_imgs") else folder_name
    print(f"[INFO] Source folder: {src}")
    print(f"[INFO] Object name: {object_name}")

    global images_map, labels_map
    images_map, labels_map = discover_maps(src)
    print(f"[INFO] Found {len(images_map)} images and {len(labels_map)} labels in source folder.")

    # Delete images that don't have a label
    orphan_images = set(images_map.keys()) - set(labels_map.keys())
    if orphan_images:
        print(f"[INFO] Deleting {len(orphan_images)} images without labels...")
    for stem in sorted(orphan_images, key=natural_key):
        img_name = images_map[stem]
        print(f"[DELETE] {img_name}")
        if not args.dry_run:
            (src / img_name).unlink()
        del images_map[stem]

    # Update maps after deletion
    images_map, labels_map = discover_maps(src)
    paired_basenames = sorted(set(images_map.keys()) & set(labels_map.keys()), key=natural_key)
    if not paired_basenames:
        print("[ERROR] No matching image/label pairs found. Exiting.")
        return

    print(f"[INFO] {len(paired_basenames)} valid pairs found and will be processed.")

    # Rename all matched pairs to object_name_00000... safely
    safe_two_phase_rename(src, paired_basenames, object_name, dry_run=args.dry_run)

    # After renaming, build final pairs list from files
    all_images = sorted([p.name for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.name.startswith(object_name + "_")], key=natural_key)
    all_labels = sorted([p.name for p in src.iterdir() if p.is_file() and p.suffix.lower() == LABEL_EXT and p.name.startswith(object_name + "_")], key=natural_key)
    # pair by sorted order (they should align)
    pairs = list(zip(all_images, all_labels))
    print(f"[INFO] {len(pairs)} pairs after renaming.")

    # Shuffle & split
    random.seed(args.seed)
    random.shuffle(pairs)
    n_train = int(len(pairs) * TRAIN_SPLIT)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]
    print(f"[INFO] Split -> train: {len(train_pairs)}, val: {len(val_pairs)} (ratio {TRAIN_SPLIT})")

    # Dest folders
    train_img_dest = IMAGES_STORAGE / "train" / object_name
    val_img_dest = IMAGES_STORAGE / "val" / object_name
    train_lbl_dest = LABELS_STORAGE / "train" / object_name
    val_lbl_dest = LABELS_STORAGE / "val" / object_name

    # Move and reindex per set
    print("[INFO] Moving & reindexing train set...")
    move_and_reindex(src, train_pairs, train_img_dest, train_lbl_dest, object_name, dry_run=args.dry_run)

    print("[INFO] Moving & reindexing val set...")
    move_and_reindex(src, val_pairs, val_img_dest, val_lbl_dest, object_name, dry_run=args.dry_run)

    print("[DONE] Finished. Train:", train_img_dest, "Val:", val_img_dest)


if __name__ == "__main__":
    main()
