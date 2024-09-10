import cv2
import supervision as sv
from ultralytics import YOLOv10
import typer
import time

# Load the model
model = YOLOv10("yolov10/weights/yolov10n.pt")
app = typer.Typer()

category_dict = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# Define real-world widths for some classes (in cm)
real_widths = {
    'person': 40,
    'bicycle': 60,
    'car': 180,
    'motorcycle': 100,
    'airplane': 2000,
    'bus': 250,
    'train': 300,
    'truck': 250,
    'boat': 300,
    'traffic light': 30,
    'fire hydrant': 50,
    'stop sign': 75,
    'parking meter': 20,
    'bench': 150,
    'bird': 30,
    'cat': 25,
    'dog': 40,
    'horse': 150,
    'sheep': 60,
    'cow': 180,
    'elephant': 250,
    'bear': 200,
    'zebra': 150,
    'giraffe': 250,
    'backpack': 30,
    'umbrella': 100,
    'handbag': 30,
    'tie': 10,
    'suitcase': 50,
    'frisbee': 25,
    'skis': 180,
    'snowboard': 150,
    'sports ball': 22,
    'kite': 100,
    'baseball bat': 80,
    'baseball glove': 30,
    'skateboard': 80,
    'surfboard': 200,
    'tennis racket': 70,
    'bottle': 8,
    'wine glass': 7,
    'cup': 10,
    'fork': 20,
    'knife': 20,
    'spoon': 20,
    'bowl': 15,
    'banana': 20,
    'apple': 10,
    'sandwich': 15,
    'orange': 8,
    'broccoli': 10,
    'carrot': 5,
    'hot dog': 20,
    'pizza': 30,
    'donut': 10,
    'cake': 20,
    'chair': 50,
    'couch': 200,
    'potted plant': 30,
    'bed': 200,
    'dining table': 150,
    'toilet': 50,
    'tv': 100,
    'laptop': 35,
    'mouse': 6,
    'remote': 15,
    'keyboard': 40,
    'cell phone': 8,
    'microwave': 45,
    'oven': 60,
    'toaster': 30,
    'sink': 50,
    'refrigerator': 70,
    'book': 5,
    'clock': 30,
    'vase': 20,
    'scissors': 15,
    'teddy bear': 30,
    'hair drier': 20,
    'toothbrush': 3,
}

# Assume a focal length for the camera (in pixels)
focal_length = 800  # This needs to be calibrated for your specific camera

def process_webcam():

    # Vals to resize video frames | small frame optimizes the run
    frame_wid = 1280
    frame_hyt = 720

    cap = cv2.VideoCapture(0)  # 0 is typically the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Track summary statistics only for detected classes
    total_detections = {}
    max_detections = {}
    frames_with_detections = {}

    prev_frame_time = 0

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame | small frame optimizes the run
        frame = cv2.resize(frame, (frame_wid, frame_hyt))

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        class_detections_in_frame = {}

        for box, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
            class_name = category_dict[class_id]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Calculate distance
            real_width = real_widths.get(class_name, 50)  # Default to 50 cm if not found
            object_width_in_pixels = x2 - x1
            distance = (real_width * focal_length) / object_width_in_pixels

            # Update class detection count
            class_detections_in_frame[class_name] = class_detections_in_frame.get(class_name, 0) + 1

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}, {distance:.2f} cm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Update summary statistics
        for class_name, count in class_detections_in_frame.items():
            if class_name not in total_detections:
                total_detections[class_name] = 0
                max_detections[class_name] = 0
                frames_with_detections[class_name] = 0

            total_detections[class_name] += count
            frames_with_detections[class_name] += 1
            max_detections[class_name] = max(max_detections[class_name], count)

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

        frame_count += 1

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

@app.command()
def webcam():
    typer.echo("Starting webcam processing...")
    process_webcam()

if __name__ == "__main__":
    app()
