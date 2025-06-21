import os
import cv2
from dotenv import load_dotenv

load_dotenv()

VIDEO_DIR = os.environ.get("VIDEO_DIR", "video/（主）高崎渋川線（バイパス） ふれあい歩道橋/")

# Get the first video in the VIDEO_DIR directory
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.mp3'))]
if not video_files:
    print(f"No video files found in {VIDEO_DIR}")
    exit(1)
video_path = os.path.join(VIDEO_DIR, video_files[0])

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()
if not ret:
    print("Cannot read the first frame of the video.")
    exit(1)

frame_copy = frame.copy()
rectangles = []
drawing = False
pt1 = None

print("Draw 2 rectangles (regions) on the video frame. Left click and drag to draw. Press 'q' to quit after drawing.")

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
    exit(1)

# Convert to x1,y1,x2,y2 format
region1 = f"{rectangles[0][0][0]},{rectangles[0][0][1]},{rectangles[0][1][0]},{rectangles[0][1][1]}"
region2 = f"{rectangles[1][0][0]},{rectangles[1][0][1]},{rectangles[1][1][0]},{rectangles[1][1][1]}"

# Suggest writing to .env file
with open('.env', 'a', encoding='utf-8') as f:
    f.write(f"\nREGION1={region1}\n")
    f.write(f"REGION2={region2}\n")

print("\nREGION1 and REGION2 have been written to the .env file. You can copy them to the server if needed.")
print(f"REGION1={region1}")
print(f"REGION2={region2}")
