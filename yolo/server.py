import cv2
import numpy as np
import base64
import json
from fastapi import FastAPI, WebSocket
from ultralytics import YOLO
import uvicorn

# Load models once
segment_model = YOLO("runs/segment/train2/weights/best.pt")
detect_model = YOLO("runs/detect/train/weights/best.pt")

app = FastAPI()

def infer_image(image: np.ndarray, model: YOLO, imgsz: int = 960, conf: float = 0.35):
    results = model(image, imgsz=imgsz, conf=conf)
    output = []
    for r in results:
        for i, box in enumerate(r.boxes):
            obj = {
                "class": int(box.cls[0].item()),
                "confidence": float(box.conf[0].item()),
                "bbox": box.xyxy[0].tolist()
            }
            if r.masks is not None:
                obj["mask"] = r.masks.xy[i].tolist()
            output.append(obj)
    return output

@app.websocket("/ws/segment")
async def ws_segment(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            message = await websocket.receive_text()
            data = json.loads(message)
            img_bytes = base64.b64decode(data["image"])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            imgsz = data.get("imgsz", 960)
            conf = data.get("conf", 0.45)

            results = infer_image(image, segment_model, imgsz, conf)
            await websocket.send_text(json.dumps({"results": results}))

        except Exception as e:
            print("WebSocket error:", e)
            await websocket.close()
            break

@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            message = await websocket.receive_text()
            data = json.loads(message)
            img_bytes = base64.b64decode(data["image"])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            imgsz = data.get("imgsz", 960)
            conf = data.get("conf", 0.45)

            results = infer_image(image, detect_model, imgsz, conf)
            await websocket.send_text(json.dumps({"results": results}))

        except Exception as e:
            print("WebSocket error:", e)
            await websocket.close()
            break

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
