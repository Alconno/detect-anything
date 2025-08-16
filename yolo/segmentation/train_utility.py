import os
import shutil



label_dirs = ["./my_dataset/labels/train", "./my_dataset/labels/val"]
tmp_dirs = ["./my_dataset/labels/ttmp", "./my_dataset/labels/vtmp"]

def prepare_labels():
    for label_dir, tmp_dir in zip(label_dirs, tmp_dirs):
        os.makedirs(tmp_dir, exist_ok=True)

        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue

            src_file = os.path.join(label_dir, fname)
            tmp_file = os.path.join(tmp_dir, fname)

            # Move original to tmp
            shutil.move(src_file, tmp_file)

            with open(tmp_file, "r") as f:
                lines = f.readlines()

            cleaned_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) <= 1:
                    continue

                raw_cls = parts[0]
                coords = parts[1:]
                polygon = coords[4:]  # Skip bbox (cx, cy, w, h)

                # For non-augmented files: remove instance ID
                if "aug" not in fname:
                    if "_" in raw_cls:
                        cls = raw_cls.split("_")[0]
                    else:
                        cls = raw_cls
                else:
                    cls = raw_cls  # Keep instance ID

                cleaned_lines.append(f"{cls} {' '.join(polygon)}\n")

            with open(src_file, "w") as f:
                f.writelines(cleaned_lines)

    print("✅ Labels prepared for training.")


def restore_labels():
    for label_dir, tmp_dir in zip(label_dirs, tmp_dirs):
        if not os.path.exists(tmp_dir):
            print(f"⚠️ Temp folder '{tmp_dir}' doesn't exist — skipping restore.")
            continue

        for fname in os.listdir(tmp_dir):
            tmp_file = os.path.join(tmp_dir, fname)
            original_file = os.path.join(label_dir, fname)

            if os.path.exists(original_file):
                os.remove(original_file)

            shutil.move(tmp_file, original_file)

        shutil.rmtree(tmp_dir)

    print("✅ Original labels restored.")


