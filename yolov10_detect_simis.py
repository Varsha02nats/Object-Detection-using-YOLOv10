import random
import cv2
import numpy as np
from ultralytics import YOLOv10
import time

# Load the trained YOLOv10 model
model = YOLOv10("simis.pt")

# Retrieve class names from the model
class_list = model.names

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Vals to resize video frames | small frame optimizes the run
frame_wid = 1280
frame_hyt = 720

# Open webcam
cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

prev_frame_time = 0
new_frame_time = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Resize the frame | small frame optimizes the run
    frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on the image
    results = model(frame)

    if len(results) > 0:
        for result in results:
            boxes = result.boxes
            for box in boxes:
                clsID = int(box.cls[0])
                conf = float(box.conf[0])
                bb = box.xyxy[0].cpu().numpy().astype(int)

                cv2.rectangle(
                    frame,
                    (bb[0], bb[1]),
                    (bb[2], bb[3]),
                    detection_colors[clsID],
                    3,
                )

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    class_list[clsID] + " " + str(round(conf * 100, 2)) + "%",
                    (bb[0], bb[1] - 10),
                    font,
                    1,
                    (0, 0, 0),  # Set text color to black
                    2,
                )

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Display FPS on frame
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
