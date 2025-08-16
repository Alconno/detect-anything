"""
📝 SCRIPT OVERVIEW: relabel_by_prefix.py

This script is used to update YOLO-format `.txt` label files by changing the **class index**
(i.e., the first number in each line) for files with a specific prefix in their filename.

🎯 USE CASE:
Suppose you have labeled files like:
    rhino_00000.txt
    rhino_00001.txt
And each line in those files looks like:
    0 0.123 0.456 0.789 0.101

This `0` is the class index. If `rhino` should be class `1` instead of `0`, you can run:

    python relabel_by_prefix.py --dir ./labels --prefix rhino --new_index 1

This will change the labels to:
    1 0.123 0.456 0.789 0.101

✅ It updates only files that:
- Start with the given prefix (`rhino_`)
- End with `.txt`
- Have lines that follow YOLO format: `class x_center y_center width height`

📦 EXAMPLE USAGE:

    python relabel_by_prefix.py \\
        --dir ./my_dataset/labels/train \\
        --prefix rhino \\
        --new_index 1

This will change the class index in all `rhino_*.txt` files inside `./my_dataset/labels/train`.

📌 NOTES:
- Files are matched by filename prefix (e.g., "rhino"), not by content.
- Only the **first number** on each line (the class index) is changed.
- Malformed lines (with fewer than 5 tokens) are left untouched.
- This is helpful if you generated labels automatically and now need to fix class mismatches.
"""
import os
import argparse



def relabel_files(dir_path, prefix, new_index):
    changed_files = 0
    for fname in os.listdir(dir_path):
        if fname.startswith(prefix) and fname.endswith(".txt"):
            file_path = os.path.join(dir_path, fname)
            with open(file_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[0] = str(new_index)
                    new_lines.append(" ".join(parts))
                else:
                    new_lines.append(line.strip())  # leave invalid lines unchanged

            with open(file_path, "w") as f:
                f.write("\n".join(new_lines) + "\n")
            changed_files += 1

    print(f"Updated {changed_files} label files starting with '{prefix}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relabel YOLO .txt files based on filename prefix.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing .txt label files.")
    parser.add_argument("--prefix", type=str, required=True, help="File prefix to match (e.g., 'rhino').")
    parser.add_argument("--new_index", type=int, required=True, help="New class index to assign.")

    args = parser.parse_args()
    relabel_files(args.dir, args.prefix, args.new_index)
