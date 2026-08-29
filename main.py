"""
main.py
Entry point. Same webcam-loop pattern I already used before
(cap.read() -> process -> imshow -> waitKey), just with each
processing step pulled out into its own module so the loop
stays readable.

Pipeline per frame:
1. Grab frame from webcam
2. detector.py    -> find car bounding boxes
3. plate_extractor.py -> crop the likely plate region from each car box
4. ocr_reader.py  -> read text off that crop
5. logger.py      -> log IN/OUT to CSV if this is a new/confident reading
6. draw boxes + plate text on the frame and show it
"""

import cv2
import cvzone
import time
import config
from detector import detect_cars
from plate_extractor import extract_plate_region
from ocr_reader import read_plate_text, _preprocess
from logger import log_plate

_debug_frame_count = 0  # used to give each saved crop a unique filename


def main():
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)

    if not cap.isOpened():
        print("Could not open video source:", config.VIDEO_SOURCE)
        return

    # rolling FPS measurement - so I have a real number to quote instead
    # of guessing. Averaged over the last 30 frames so it doesn't jump
    # around wildly frame to frame.
    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        fps = 1 / (now - prev_time) if now != prev_time else fps
        prev_time = now

        car_boxes = detect_cars(frame)
        print(f"[DEBUG] cars detected this frame: {len(car_boxes)}")  # DEBUG

        for car_box in car_boxes:
            x1, y1, x2, y2, conf = car_box
            print(f"[DEBUG] car box={x1,y1,x2,y2} conf={conf:.2f}")  # DEBUG

            # draw the car box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

            plate_img, plate_coords = extract_plate_region(frame, car_box)
            if plate_img is None:
                print("[DEBUG] plate crop was empty, skipping")  # DEBUG
                continue

            # save the raw crop + preprocessed version so we can
            # visually check the heuristic is landing on the plate
            if config.DEBUG_SAVE_CROPS:
                global _debug_frame_count
                _debug_frame_count += 1
                crop_path = f"data/debug_crops/crop_{_debug_frame_count:05d}_raw.png"
                proc_path = f"data/debug_crops/crop_{_debug_frame_count:05d}_proc.png"
                cv2.imwrite(crop_path, plate_img)
                cv2.imwrite(proc_path, _preprocess(plate_img))

            # draw the plate region box too, so I can visually check
            # the heuristic crop is landing in a sane place
            px1, py1, px2, py2 = plate_coords
            cv2.rectangle(frame, (px1, py1), (px2, py2), color=(0, 255, 255), thickness=2)

            plate_text, ocr_conf = read_plate_text(plate_img)
            print(f"[DEBUG] OCR result: text={plate_text} conf={ocr_conf}")  # DEBUG

            if plate_text:
                status = log_plate(plate_text)
                print(f"[DEBUG] logger status: {status}")  # DEBUG
                label = f"{plate_text} ({ocr_conf:.2f})"
                if status:
                    label += f" - {status}"
                cvzone.putTextRect(frame, label, (px1, max(0, py1 - 10)), scale=1, thickness=1)

        cvzone.putTextRect(frame, f'FPS: {int(fps)}', (10, 30), scale=1, thickness=1)
        cv2.imshow("ANPR", frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
