from ultralytics import YOLO
import cv2
import yagmail
import time

# ---------------- SETTINGS ----------------
MODEL_PATH = "best.pt"
VIDEO_PATH = "test_image.jpg"

EMAIL_SENDER = "shreyarambly@gmail.com"
EMAIL_PASSWORD = "xznd tahx pvgs lmoe"
EMAIL_RECEIVER = "ruchithab.1si22et020@gmail.com"

ALERT_COOLDOWN = 30  # seconds between alerts
# -----------------------------------------

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

yag = yagmail.SMTP(EMAIL_SENDER, EMAIL_PASSWORD)
last_alert_time = 0

print("Monitoring started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    no_helmet_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower()

            if "no" in label:   # matches "no helmet"
                no_helmet_detected = True

    # Send alert if detected
    if no_helmet_detected:
        current_time = time.time()

        if current_time - last_alert_time > ALERT_COOLDOWN:

            screenshot_name = f"violation_{int(current_time)}.jpg"
            cv2.imwrite(screenshot_name, frame)

            yag.send(
                to=EMAIL_RECEIVER,
                subject="Helmet Violation Detected",
                contents="No helmet detected. See attached screenshot.",
                attachments=screenshot_name
            )

            print("ALERT SENT: No Helmet Detected")
            last_alert_time = current_time

    annotated_frame = results[0].plot()
    cv2.imshow("Helmet Monitoring", annotated_frame)
from ultralytics import YOLO
import cv2
import yagmail
import time

# ---------------- SETTINGS ----------------
MODEL_PATH = "best16.pt"   # your final trained model
VIDEO_PATH = "test_image.jpg"    # test video

EMAIL_SENDER = "shreyarambly@gmail.com"
EMAIL_PASSWORD = "xznd tahx pvgs lmoe"
EMAIL_RECEIVER = "ruchithab.1si22et020@gmail.com"

ALERT_COOLDOWN = 30  # seconds between alerts
# -----------------------------------------

# load model
model = YOLO(MODEL_PATH)

# video source
cap = cv2.VideoCapture(VIDEO_PATH)
# cap = cv2.VideoCapture(0)  # webcam (optional)

# email setup
yag = yagmail.SMTP(EMAIL_SENDER, EMAIL_PASSWORD)
last_alert_time = 0

print("Monitoring started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run detection
    results = model(frame)

    no_helmet_detected = False

    # check detection results
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower()

            if "no" in label:   # matches "no helmet"
                no_helmet_detected = True

    # send alert if violation detected
    if no_helmet_detected:
        current_time = time.time()

        if current_time - last_alert_time > ALERT_COOLDOWN:

            screenshot_name = f"violation_{int(current_time)}.jpg"
            cv2.imwrite(screenshot_name, frame)

            yag.send(
                to=EMAIL_RECEIVER,
                subject="Helmet Violation Detected",
                contents="No helmet detected. See attached screenshot.",
                attachments=screenshot_name
            )

            print("ALERT SENT: No Helmet Detected")
            last_alert_time = current_time

    # draw bounding boxes + labels
    annotated_frame = results[0].plot()

    # show live video detection
    cv2.imshow("Helmet Monitoring", annotated_frame)

    # press ESC to stop
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
