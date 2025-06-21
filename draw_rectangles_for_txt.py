"""
Region Annotation Tool for Vehicle Detection
-------------------------------------------
This script allows the user to draw two rectangular regions on the first frame of each video in every subfolder of VIDEO_DIR.
The coordinates of these regions are saved to a .txt file for each video, which will be used for vehicle counting and direction analysis in downstream processing (e.g., cardetection.py).

Features:
- Loads the first frame of each video in all subfolders.
- Lets the user interactively draw two rectangles (regions of interest).
- Saves the coordinates to a .txt file named after the video file.
- The .txt file format: 8 comma-separated integers (x1,y1,x2,y2,x3,y3,x4,y4).

Author: [Ikkear99]
Date: 2025-06-21
"""


import os
import cv2
from dotenv import load_dotenv

load_dotenv()

VIDEO_DIR = os.environ.get("VIDEO_DIR", "video/")

# Get all subfolders in VIDEO_DIR
subfolders = [f.path for f in os.scandir(VIDEO_DIR) if f.is_dir()]

for folder in subfolders:
    video_files = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.mp3'))]
    if not video_files:
        print(f"No video files found in {folder}")
        continue
    video_file = video_files[0]
    video_path = os.path.join(folder, video_file)
    video_name = os.path.splitext(video_file)[0]
    region_txt_path = os.path.join(folder, f"{video_name}.txt")

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Cannot read the first frame of the video: {video_path}.")
        continue

    frame_copy = frame.copy()
    rectangles = []
    drawing = False
    pt1 = None

    print(f"Draw 2 rectangles (regions) on the video frame for video: {video_file}. Left click and drag to draw. Press 'q' to quit after drawing.")

    def draw_rectangle(event, x, y, flags, param):
        global pt1, drawing, rectangles, frame_copy
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            pt1 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            pt2 = (x, y)
            rectangles.append((pt1, pt2))
            cv2.rectangle(frame_copy, pt1, pt2, (0, 255, 0), 2)
            cv2.imshow("Draw Rectangles", frame_copy)

    cv2.imshow("Draw Rectangles", frame_copy)
    cv2.setMouseCallback("Draw Rectangles", draw_rectangle)

    while True:
        cv2.imshow("Draw Rectangles", frame_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(rectangles) == 2:
            break
    cv2.destroyAllWindows()

    if len(rectangles) != 2:
        print("You must draw exactly 2 rectangles.")
        continue

    # Save rectangles to <video_name>.txt in the same folder as the video
    def save_rectangles(rects):
        if len(rects) == 2:
            coords = []
            for rect in rects:
                (x1, y1), (x2, y2) = rect
                coords.extend([x1, y1, x2, y2])
            with open(region_txt_path, "w", encoding="utf-8") as f:
                f.write(",".join(map(str, coords)))
            print(f"Saved region coordinates to {region_txt_path}")
        else:
            print("You must draw exactly 2 rectangles to save.")
    save_rectangles(rectangles)

print("\nREGION1 and REGION2 have been saved to region.txt.")
