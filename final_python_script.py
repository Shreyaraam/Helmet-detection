import sys
import os
from datetime import datetime
import traceback

# ---------------- PATH SETUP ----------------
base_path = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)

log_file_path = os.path.join(base_path, "helmet_monitor_log.txt")

# Create violation folder
violation_folder = os.path.join(base_path, "violations")
if not os.path.exists(violation_folder):
    os.makedirs(violation_folder)

def log_to_file(message):
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# ---------------- RUN HEADER ----------------
run_start_time = datetime.now()
log_to_file("\n" + "="*60)
log_to_file(f"Run Started At: {run_start_time}")
log_to_file("="*60)

# ---------------- IMPORTS ----------------
from ultralytics import YOLO
import cv2
import yagmail
import time
import json

ALERT_COOLDOWN = 10

last_detections = []
IOU_THRESHOLD = 0.3
TIME_WINDOW = 10
DISTANCE_THRESHOLD = 100

alerted_ids = set()

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# ---------------- CONFIG LOAD ----------------
config_path = os.path.join(base_path, "config.json")

if not os.path.exists(config_path):
    raise Exception("config.json file not found!")

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

model_path = config["model_path"]
video_path = config["video_path"]

senders = config["senders"]
receivers = config["receivers"]
cc_list = config.get("cc", [])

# Track config changes
last_config_modified_time = os.path.getmtime(config_path)
current_model_path = model_path
current_video_path = video_path

# ---------------- INITIALIZE MODEL ----------------
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

last_alert_time = 0

print("Monitoring started...")

# ---------------- CONFIG RELOAD FUNCTION ----------------
def load_config():
    global senders, receivers, cc_list
    global last_config_modified_time
    global model, cap
    global current_model_path, current_video_path

    modified_time = os.path.getmtime(config_path)

    if modified_time != last_config_modified_time:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            senders = config["senders"]
            receivers = config["receivers"]
            cc_list = config.get("cc", [])

            new_model_path = config["model_path"]
            if new_model_path != current_model_path:
                print("Reloading model...")
                model = YOLO(new_model_path)
                current_model_path = new_model_path
                print("Model reloaded.")

            new_video_path = config["video_path"]
            if new_video_path != current_video_path:
                print("Reloading video source...")
                cap.release()
                cap = cv2.VideoCapture(new_video_path)
                current_video_path = new_video_path
                print("Video source updated.")

            last_config_modified_time = modified_time
            print("Configuration updated dynamically.")

        except Exception as e:
            print("Error loading config.json:", e)

# ---------------- MAIN LOOP ----------------
try:
    while True:

        load_config()

        ret, frame = cap.read()
        if not ret:
            cap.release()

            while True:
                cap = cv2.VideoCapture(current_video_path)
                if cap.isOpened():
                    break
                time.sleep(5)

        results = model.track(frame, persist=True, tracker="botsort.yaml")

        new_violation_detected = False
        bike_boxes = []

        for r in results:

            if r.boxes is None:
                continue

            for box in r.boxes:

                track_id = int(box.id[0]) if box.id is not None else None
                cls = int(box.cls[0])
                label = model.names[cls].lower()

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if "bike" in label:
                    bike_boxes.append((x1, y1, x2, y2))

                if "no" in label:

                    #  NEW: SKIP IF ALREADY ALERTED
                    
                    current_box = (x1, y1, x2, y2)

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # -------- BIKE PROXIMITY --------
                    is_near_bike = False
                    for bx1, by1, bx2, by2 in bike_boxes:
                        bcx = (bx1 + bx2) // 2
                        bcy = (by1 + by2) // 2

                        distance_bike = ((center_x - bcx)**2 + (center_y - bcy)**2) ** 0.5

                        if distance_bike < 200:
                            is_near_bike = True
                            break

                    if not is_near_bike:
                        continue

                    # -------- DUPLICATE FILTER --------
                    is_duplicate = False

                    for prev_box, prev_time in last_detections:

                        px1, py1, px2, py2 = prev_box

                        prev_center_x = (px1 + px2) // 2
                        prev_center_y = (py1 + py2) // 2

                        distance = ((center_x - prev_center_x)**2 + (center_y - prev_center_y)**2) ** 0.5
                        iou = compute_iou(current_box, prev_box)

                        if (
                            iou > IOU_THRESHOLD and
                            distance < DISTANCE_THRESHOLD and
                            (time.time() - prev_time) < TIME_WINDOW
                        ):
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        last_detections.append((current_box, time.time()))
                          #  MARK AS ALERTED
                        new_violation_detected = True

        # clean old detections
        last_detections = [
            (box, t) for box, t in last_detections
            if time.time() - t < TIME_WINDOW
        ]

        if new_violation_detected:

            current_time = time.time()

            if current_time - last_alert_time > ALERT_COOLDOWN:

                print("No Helmet Detected")
                log_to_file(f"{datetime.now()} - No Helmet Detected")

                screenshot_name = os.path.join(
                    violation_folder,
                    f"violation_{int(current_time)}.jpg"
                )

                cv2.imwrite(screenshot_name, frame)

                for sender in senders:
                    try:
                        yag = yagmail.SMTP(sender["email"], sender["app_password"])

                        yag.send(
                            to=receivers,
                            cc=cc_list,
                            subject="Helmet Violation Detected",
                            contents="No helmet detected. See attached screenshot.",
                            attachments=screenshot_name
                        )

                        yag.close()

                        print("Mail Sent")
                        log_to_file(f"{datetime.now()} - Mail Sent from {sender['email']}")

                    except Exception as mail_error:
                        error_message = f"{datetime.now()} - ERROR SENDING MAIL from {sender['email']}: {str(mail_error)}"
                        print(error_message)
                        log_to_file(error_message)
                        log_to_file(traceback.format_exc())

                last_alert_time = current_time

except Exception as e:
    error_message = f"{datetime.now()} - ERROR: {str(e)}"
    print(error_message)
    log_to_file(error_message)
    log_to_file(traceback.format_exc())

cap.release()

log_to_file(f"Run Ended At: {datetime.now()}")
log_to_file("="*60)
