"""
config.py
All the fixed values used across the project live here.
Keeping them in one place means if I need to tune something
(like how much of the car box counts as "plate area"), I only
change it once instead of hunting through every file.
"""

import os

# path to the pretrained YOLO model (COCO classes -> we only care about class 2 = "car")
YOLO_MODEL_PATH = "yolo-weights/yolov8n.pt"

# COCO class id for "car". (0=person, 2=car, 3=motorcycle, 5=bus, 7=truck)
CAR_CLASS_ID = 2

# minimum confidence for YOLO to count a detection as a real car
# lowered temporarily for debugging - screen/photo detections score lower
# than real cars. put this back to 0.5 once things are working.
CAR_CONF_THRESHOLD = 0.3

# --- plate region heuristic ---
# I'm not using a plate-trained model yet, so instead I estimate where
# the plate usually sits inside a detected car box: bottom-center area.
PLATE_Y_START_RATIO = 0.65   # start cropping from 65% down the car box
PLATE_Y_END_RATIO = 0.95     # end near the bottom of the car box
PLATE_X_START_RATIO = 0.20   # skip the left 20% of the car box
PLATE_X_END_RATIO = 0.80     # skip the right 20% of the car box

# minimum length a cleaned plate string must have to be considered
# real. This blocks single-character OCR "noise" (e.g. it once read
# a lone digit "2" at high confidence and logged it as a plate).
# Most real plates are 6+ characters once cleaned.
MIN_PLATE_LENGTH = 6

# minimum OCR confidence to accept a reading as valid.
# 0.1 was too low - it let garbage readings through. 0.3 filters out
# the worst noise while still accepting real plates read at an angle.
OCR_CONF_THRESHOLD = 0.3

# where logs get written
LOG_FILE = "data/vehicle_log.csv"

# don't log the same plate again within this many seconds
# (otherwise every frame the car is on screen would create a new log row)
# 20s: a car sitting in view won't spam the log, but a genuine second
# entry/exit within the same session still gets recorded quickly enough.
LOG_COOLDOWN_SECONDS = 20

# webcam index (0 = default webcam). Can also be a video file path.
VIDEO_SOURCE = 0

os.makedirs("data", exist_ok=True)

# --- debug: save plate crops to disk so you can see what OCR is receiving ---
# Set to True to dump every plate crop as a PNG into data/debug_crops/.
# Useful for tuning PLATE_*_RATIO values when OCR reads nothing.
# Set back to False once the crop region looks correct.
DEBUG_SAVE_CROPS = False
if DEBUG_SAVE_CROPS:
    os.makedirs("data/debug_crops", exist_ok=True)
