"""
📝 SCRIPT OVERVIEW: extract_frames.py

This script extracts frames from a video at a fixed time interval and saves them as image files.

🎯 USE CASE:
You have a video file (e.g., `fires.mp4`) and you want to extract still frames every N milliseconds
(e.g., every 1000ms = 1 second) for use in datasets, object detection, training, etc.

🛠️ WHAT IT DOES:
1. Opens the video using OpenCV.
2. Calculates how many frames to skip based on the video's FPS and your interval in milliseconds.
3. Iterates through the video frame by frame.
4. Saves a frame only if it matches the interval (based on frame count).
5. Stores frames in the specified output directory, using filenames like:
       frame_00000.png
       frame_00001.png
       ...

📦 EXAMPLE USAGE:

    python extract_frames.py ./vids/fires_clip.mp4 ./vids/output_imgs 1000

This will:
- Load `fires_clip.mp4`
- Save one frame every 1000 milliseconds
- Output frames as PNG files to `./vids/output_imgs/`

📌 NOTES:
- Output directory will be created automatically if it doesn’t exist.
- Output filenames are zero-padded to 5 digits (e.g., frame_00012.png).
- Interval is based on **wall time**, not frame count, so it's FPS-aware.
"""

import cv2
import os
import argparse


def extract_frames(video_path, output_dir, interval_ms):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int((interval_ms / 1000) * fps)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:05d}.png")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video every N milliseconds.")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("output_dir", help="Directory to save extracted frames")
    parser.add_argument("interval_ms", type=int, help="Interval in milliseconds between frames")

    args = parser.parse_args()
    extract_frames(args.video_path, args.output_dir, args.interval_ms)
