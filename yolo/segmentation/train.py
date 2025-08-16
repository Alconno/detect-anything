from ultralytics import YOLO
from multiprocessing import freeze_support
import os
import sys

from train_utility import restore_labels, prepare_labels, tmp_dirs


def main():
    freeze_support()
    
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 960

    # 🛑 Check if previous temp exists → restore first
    for tmp_dir in tmp_dirs:
        if os.path.exists(tmp_dir) and os.listdir(tmp_dir):
            print(f" Restoring from existing temp folder: {tmp_dir}")
            restore_labels()
            break  # Only need to restore once, then prepare again

    prepare_labels()

    try:
        print("\n" + "="*60)
        print("STARTING FINE-TUNING")
        print("="*60 + "\n")

        model = YOLO('./yolo/yolo11n-seg.pt')
        model.train(data='./my_dataset/data.yaml', epochs=epochs, imgsz=imgsz, batch=16)
    finally:
        restore_labels()


if __name__ == '__main__':
    main()





