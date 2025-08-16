from ultralytics import YOLO
from multiprocessing import freeze_support
import os
import sys



def main():
    freeze_support()

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 960

    print("\n" + "="*60)
    print("🧠   STARTING FINE-TUNING (Detection)")
    print("="*60 + "\n")

    model = YOLO('./yolo/yolo11n.pt')
    model.train(data='./my_dataset/data.yaml', epochs=epochs, imgsz=imgsz, batch=16)



if __name__ == '__main__':
    main()
