"""
plate_extractor.py
Given a car's bounding box, this crops out the region where the
plate is most likely to be. This is the same crop logic from my
crop-tool script (img[y1:y2, x1:x2]) — just now the coordinates
are calculated from ratios instead of a mouse click.

NOTE: this is a heuristic, not a trained plate detector. It works
because plates usually sit bottom-center of the car's bounding box.
A proper next step would be training YOLO on an actual plate dataset,
but this proves the rest of the pipeline (OCR + logging) works first.
"""

import config


def extract_plate_region(frame, car_box):
    """
    car_box: (x1, y1, x2, y2, confidence) from detector.py
    Returns: cropped plate image (numpy array), or None if the crop is invalid
    """
    x1, y1, x2, y2, _ = car_box

    box_width = x2 - x1
    box_height = y2 - y1

    plate_x1 = x1 + int(box_width * config.PLATE_X_START_RATIO)
    plate_x2 = x1 + int(box_width * config.PLATE_X_END_RATIO)
    plate_y1 = y1 + int(box_height * config.PLATE_Y_START_RATIO)
    plate_y2 = y1 + int(box_height * config.PLATE_Y_END_RATIO)

    plate_img = frame[plate_y1:plate_y2, plate_x1:plate_x2]

    # guard against empty crops (can happen if the car box is right at
    # the edge of the frame)
    if plate_img.size == 0:
        return None, None

    plate_coords = (plate_x1, plate_y1, plate_x2, plate_y2)
    return plate_img, plate_coords
