"""
detector.py
Wraps the YOLO model. Only job: take a frame, return a list of
car bounding boxes. This is basically the same detection loop
I already wrote before, just pulled into a function so main.py
stays readable.
"""

from ultralytics import YOLO
import config

# load the model once when this module is imported (not every frame,
# that would be very slow)
model = YOLO(config.YOLO_MODEL_PATH)


def detect_cars(frame):
    """
    Runs YOLO on a single frame and returns only the car boxes.

    Returns: list of (x1, y1, x2, y2, confidence) tuples
    """
    car_boxes = []

    results = model(frame, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id == config.CAR_CLASS_ID and confidence >= config.CAR_CONF_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                car_boxes.append((x1, y1, x2, y2, confidence))

    return car_boxes
