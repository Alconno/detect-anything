import numpy as np
import time
import mss
import base64
import json
import asyncio
import websockets
import cv2  # for encoding only

SCREEN_REGION = {'top': 450, 'left': 800, 'width': 960, 'height': 540}  # same as server ROI

async def run_ws_stream(uri="ws://localhost:8000/ws/segment", imgsz=960, conf=0.45):
    async with websockets.connect(uri) as websocket:
        sct = mss.mss()
        while True:
            start_time = time.time()

            # Capture screen
            screen = np.array(sct.grab(SCREEN_REGION))

            frame_rgb = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)
            frame_bgr = frame_rgb[..., ::-1] # RGB -> BGR for model
            _, buffer = cv2.imencode(".jpg", frame_bgr)
            img_b64 = base64.b64encode(buffer).decode("utf-8")

            # Send frame to server
            message = {
                "image": img_b64,
                "imgsz": imgsz,
                "conf": conf,
                }
            await websocket.send(json.dumps(message))

            # Receive JSON result
            response = await websocket.recv()
            data = json.loads(response)

            # Print results per frame
            fps = 1 / (time.time() - start_time)
            print(f"FPS: {fps:.2f}, Results:", data)

            # Small sleep to avoid flooding
            await asyncio.sleep(0.005)

if __name__ == "__main__":
    asyncio.run(run_ws_stream())
