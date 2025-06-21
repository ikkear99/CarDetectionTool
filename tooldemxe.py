# main.py
import os

import cv2
import pandas as pd
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

VIDEO_DIR = os.environ.get("VIDEO_DIR", "video/（主）高崎渋川線（バイパス） ふれあい歩道橋/")
REGION_PATH = os.environ.get("REGION_PATH", "output/region.txt")
OUTPUT_EXCEL = os.environ.get("OUTPUT_EXCEL", "output/results.xlsx")

VIDEO_PATHS = [os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.endswith('.mp3')]

os.makedirs("output", exist_ok=True)
region = []
car_count = 0
object_state = {}
region_lines = []

model = YOLO("yolov8m.pt")  # You can use yolov8s.pt or yolov8m.pt for higher accuracy
model.to("cuda")  # Move model to GPU
print(f"[INFO] YOLO model is using device: {model.device}")


def draw_rectangle(event, x, y, flags, param):
    global region_rectangles
    if event == cv2.EVENT_LBUTTONDOWN:
        region_rectangles.append([(x, y)])
    elif event == cv2.EVENT_LBUTTONUP:
        region_rectangles[-1].append((x, y))
        cv2.rectangle(frame_copy, region_rectangles[-1][0], region_rectangles[-1][1], (0, 255, 0), 2)
        cv2.imshow("Select 2 Rectangles", frame_copy)
        if len(region_rectangles) == 2:
            with open(REGION_PATH, "w") as f:
                # save 2 rectangles
                coord = ",".join(map(str, sum(region_rectangles, [])))  # flatten list
                f.write(coord)
            print(f"[INFO] 2 rectangles saved to {REGION_PATH}")
            cv2.waitKey(500)
            cv2.destroyAllWindows()


# Step 1: Select Line A -> B
VIDEO_PATH = VIDEO_PATHS[0]
if not os.path.exists(VIDEO_PATH):
    print(f"[ERROR] Video not found at {VIDEO_PATH}")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print("[ERROR] Cannot read the first frame of the video.")
    exit()

frame_copy = frame.copy()

# --- Select 2 Rectangles ---
region_rectangles = []
if not os.path.exists(REGION_PATH):
    cv2.imshow("Select 2 Rectangles", frame_copy)
    cv2.setMouseCallback("Select 2 Rectangles", draw_rectangle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Load rectangles from file if needed ---
if not region_rectangles:
    with open(REGION_PATH, "r") as f:
        coords = list(map(int, f.read().strip().split(",")))
        region_rectangles = [
            [(coords[0], coords[1]), (coords[2], coords[3])],
            [(coords[4], coords[5]), (coords[6], coords[7])]
        ]


def is_inside_rectangle(point, rect_top_left, rect_bottom_right):
    x, y = point
    x1, y1 = rect_top_left
    x2, y2 = rect_bottom_right
    return x1 <= x <= x2 and y1 <= y <= y2


def save_counted_cars(counted_cars, save_root, period):
    minute_folder = os.path.join(save_root, f"{period:03d}min")
    os.makedirs(minute_folder, exist_ok=True)
    for car_info in counted_cars:
        x1, y1, x2, y2, car_count, track_id, fidx, frame_img = car_info
        car_img = frame_img[y1:y2, x1:x2]
        img_name = f"car_{car_count}_id{track_id}_f{fidx}.jpg"
        img_path = os.path.join(minute_folder, img_name)
        cv2.imwrite(img_path, car_img)


# --- Main loop ---
for VIDEO_PATH in VIDEO_PATHS:
    print(f"[INFO] Processing video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = 0
    car_count_down = 0  # Downward direction (region 2 -> region 1)
    car_count_up = 0  # Upward direction (region 1 -> region 2)
    data = []
    car_ids_down = set()
    car_ids_up = set()
    object_state = {}
    video_base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    # Change output path to under output/高崎渋川線（バイパス） ふれあい歩道橋
    output_base = os.path.join('output', '高崎渋川線（バイパス） ふれあい歩道橋')
    os.makedirs(output_base, exist_ok=True)
    save_root_down = os.path.join(output_base, video_base, 'down')
    save_root_up = os.path.join(output_base, video_base, 'up')
    os.makedirs(save_root_down, exist_ok=True)
    os.makedirs(save_root_up, exist_ok=True)

    # Get fps to calculate minutes
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Change here
    counted_cars_in_period_down = []
    counted_cars_in_period_up = []
    current_period = 0
    period_length_min = 10
    period_length_frame = int(fps * 60 * period_length_min)

    print("[INFO] Starting vehicle detection and counting...")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Save remaining cars at the end of the video
            if counted_cars_in_period_down:
                save_counted_cars(counted_cars_in_period_down, save_root_down, current_period)
            if counted_cars_in_period_up:
                save_counted_cars(counted_cars_in_period_up, save_root_up, current_period)
            break

        results = model.track(frame, persist=True)[0]
        for box in results.boxes:
            cls = int(box.cls)
            # Filter for specific classes: 1 (car), 2 (truck), 3 (bus), 5 (motorcycle), 7 (bicycle)
            if cls in [1, 2, 3, 5, 7] and box.id is not None:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                track_id = int(box.id)

                # Draw bounding box and center
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Track state for both directions, with separate flags for up/down
                if track_id not in object_state:
                    object_state[track_id] = {
                        "entered_rect1": False, "entered_rect2": False,
                        "counted_up": False, "counted_down": False,
                        "first_entered": None
                    }
                state = object_state[track_id]

                # Check for entering region 1 (bottom)
                if not state["entered_rect1"] and is_inside_rectangle((cx, cy), region_rectangles[0][0],
                                                                      region_rectangles[0][1]):
                    state["entered_rect1"] = True
                    if state["first_entered"] is None:
                        state["first_entered"] = "rect1"
                # Check for entering region 2 (top)
                if not state["entered_rect2"] and is_inside_rectangle((cx, cy), region_rectangles[1][0],
                                                                      region_rectangles[1][1]):
                    state["entered_rect2"] = True
                    if state["first_entered"] is None:
                        state["first_entered"] = "rect2"

                # Down: region 1 -> region 2
                if (
                        state["entered_rect1"] and state["entered_rect2"] and
                        not state["counted_down"] and state["first_entered"] == "rect1"
                ):
                    if track_id not in car_ids_down:
                        car_ids_down.add(track_id)
                        car_count_down += 1
                        data.append({
                            "frame": frame_idx,
                            "car_id": track_id,
                            "car_count_down": car_count_down,
                            "direction": "down"
                        })
                        counted_cars_in_period_down.append(
                            (x1, y1, x2, y2, car_count_down, track_id, frame_idx, frame.copy()))
                        state["counted_down"] = True
                # Up: region 2 -> region 1
                if (
                        state["entered_rect1"] and state["entered_rect2"] and
                        not state["counted_up"] and state["first_entered"] == "rect2"
                ):
                    if track_id not in car_ids_up:
                        car_ids_up.add(track_id)
                        car_count_up += 1
                        data.append({
                            "frame": frame_idx,
                            "car_id": track_id,
                            "car_count_up": car_count_up,
                            "direction": "up"
                        })
                        counted_cars_in_period_up.append(
                            (x1, y1, x2, y2, car_count_up, track_id, frame_idx, frame.copy()))
                        state["counted_up"] = True

        # Draw rectangles
        for rect in region_rectangles:
            cv2.rectangle(frame, rect[0], rect[1], (0, 255, 255), 2)

        frame_idx += 1
        # Show both counts with new positions and highlight colors
        # Up: top-left, color: (0, 0, 255) - Red
        cv2.putText(frame, f"Up: {car_count_up}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        # Down: bottom-right, color: (0, 255, 255) - Yellow
        h, w = frame.shape[:2]
        down_text = f"Down: {car_count_down}"
        (down_text_width, down_text_height), _ = cv2.getTextSize(down_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(frame, down_text, (w - down_text_width - 20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255),
                    3)
        # Display FPS at the top right corner
        text = f"FPS: {fps:.2f}"
        y = 30
        cv2.putText(frame, text, (frame.shape[1] - 200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        # Display real-time video time (seconds)
        video_time_sec = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        sec_text = f"Time: {video_time_sec}s"
        (sec_text_width, sec_text_height), _ = cv2.getTextSize(sec_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        sec_x = frame.shape[1] - sec_text_width - 20
        sec_y = y + 20
        cv2.putText(frame, sec_text, (sec_x, sec_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Change here: divide period by 600 real video seconds
        period_idx = video_time_sec // 600
        if period_idx != current_period:
            if counted_cars_in_period_down:
                save_counted_cars(counted_cars_in_period_down, save_root_down, current_period)
            if counted_cars_in_period_up:
                save_counted_cars(counted_cars_in_period_up, save_root_up, current_period)
            counted_cars_in_period_down = []
            counted_cars_in_period_up = []
            current_period = period_idx
            car_ids_down = set()
            car_ids_up = set()
            object_state = {}
            car_count_down = 0
            car_count_up = 0

    cap.release()
    cv2.destroyAllWindows()

# Step 3: Save results
# Đảm bảo các biến luôn được khởi tạo để tránh warning
car_count_down = len(car_ids_down) if 'car_ids_down' in locals() else 0
car_count_up = len(car_ids_up) if 'car_ids_up' in locals() else 0
print(f"[INFO] Total cars counted (down): {car_count_down}")
print(f"[INFO] Total cars counted (up): {car_count_up}")
df = pd.DataFrame(data) if 'data' in locals() else pd.DataFrame()
df.to_excel(OUTPUT_EXCEL, index=False)
print(f"[INFO] Results saved to {OUTPUT_EXCEL}")

# Check if the REGION_PATH file exists and delete it
if os.path.exists(REGION_PATH):
    os.remove(REGION_PATH)
print(f"[INFO] Existing file at {REGION_PATH} has been deleted.")
