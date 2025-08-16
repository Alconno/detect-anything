"""
📝 SCRIPT OVERVIEW: rename_file_base.py

This script renames files in a given directory from one base name format to another.

🔧 USE CASE:
You have a directory of files named in the format:
    frame_00000.jpg, frame_00001.jpg, ..., frame_00017.jpg
And you want to rename them to:
    rhino_00000.jpg, rhino_00001.jpg, ..., rhino_00017.jpg

You provide:
- The directory containing the files
- The old prefix (e.g., "frame")
- The new prefix (e.g., "rhino")
- A limit for how many numbered files to process

🎯 WHAT IT DOES:
1. Iterates from 0 up to `limit` (exclusive).
2. For each number `i`, generates a filename like:
      old_name = frame_00000, frame_00001, ...
      new_name = rhino_00000, rhino_00001, ...
3. Searches the directory for a file whose base name matches `old_name` (regardless of extension).
4. If found, renames that file to `new_name` while preserving the original file extension.
5. Stops once it processes `limit` number of filenames (not necessarily renames—only if matching files are found).

📦 EXAMPLE USAGE:

    python rename_file_base.py \\
        --dir ./vids/rhino_imgs \\
        --old_prefix frame \\
        --new_prefix rhino \\
        --limit 18

This would attempt to rename:
    frame_00000 → rhino_00000
    frame_00001 → rhino_00001
    ...
    frame_00017 → rhino_00017

✅ NOTES:
- Files are matched by base name only, so this works with any extension (.jpg, .png, etc.).
- Only files that exist and match the pattern are renamed.
- Useful for renaming images, frames, labels, etc. in sequence.
"""

import os
import argparse


def rename_files(directory, old_prefix, new_prefix, limit):
    renamed = 0
    for i in range(limit):
        old_name = f"{old_prefix}_{i:05d}"
        new_name = f"{new_prefix}_{i:05d}"
        
        # Find file with matching base name and any extension
        for file in os.listdir(directory):
            base, ext = os.path.splitext(file)
            if base == old_name:
                src = os.path.join(directory, file)
                dst = os.path.join(directory, new_name + ext)
                os.rename(src, dst)
                renamed += 1
                break
    print(f"Renamed {renamed} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename files in format word_00000 to new prefix.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing the files.")
    parser.add_argument("--old_prefix", type=str, required=True, help="Old prefix to match (e.g., 'frame').")
    parser.add_argument("--new_prefix", type=str, required=True, help="New prefix to use (e.g., 'rhino').")
    parser.add_argument("--limit", type=int, required=True, help="Number of files to process.")

    args = parser.parse_args()
    rename_files(args.dir, args.old_prefix, args.new_prefix, args.limit)
