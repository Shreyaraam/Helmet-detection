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

# -------- TRACKING STORAGE (NEW) --------
alerted_ids = set()

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

            # Reload model if changed
            new_model_path = config["model_path"]
            if new_model_path != current_model_path:
                print("Reloading model...")
                model = YOLO(new_model_path)
                current_model_path = new_model_path
                print("Model reloaded.")

            # Reload video source if changed
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

        # -------- YOLO TRACKING (NEW) --------
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        new_violation_detected = False

        for r in results:

            if r.boxes.id is None:
                continue

            for box in r.boxes:

                track_id = int(box.id[0])
                cls = int(box.cls[0])
                label = model.names[cls].lower()

                if "no" in label and track_id not in alerted_ids:

                    alerted_ids.add(track_id)
                    new_violation_detected = True

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
