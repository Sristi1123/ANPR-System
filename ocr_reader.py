"""
ocr_reader.py
Takes a cropped plate image and returns the text on it.
Using EasyOCR because it's a 3-line setup and I can explain
exactly what it's doing: it detects text regions in an image
and runs recognition on each one, giving back (text, confidence).

PREPROCESSING NOTE:
Raw BGR webcam frames are often too low-contrast for EasyOCR to find
text reliably. We apply three steps before passing the crop in:
  1. Grayscale  – removes colour noise, text is high-contrast by nature
  2. CLAHE      – local contrast normalisation; makes faint characters
                  pop even when the plate is in partial shadow or glare
  3. 2× resize  – EasyOCR accuracy drops sharply on crops smaller than
                  ~80px tall; upscaling recovers that detail
"""

import cv2
import easyocr
import config

# 'en' = English. gpu=False so this also works on a laptop with no GPU.
# loaded once, not per-frame (loading the model is slow).
reader = easyocr.Reader(['en'], gpu=False)

# CLAHE object – reused across frames (cheap to keep alive)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _preprocess(plate_img):
    """
    Converts a BGR plate crop into a high-contrast grayscale image
    that EasyOCR can read much more reliably.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    enhanced = _clahe.apply(gray)
    # upscale if the crop is tiny (common when car is still far away)
    h, w = enhanced.shape
    if h < 80:
        scale = 80 / h
        enhanced = cv2.resize(enhanced, (int(w * scale), 80),
                              interpolation=cv2.INTER_CUBIC)
    return enhanced


def read_plate_text(plate_img):
    """
    plate_img: cropped numpy image of the plate region
    Returns: (cleaned_text, confidence) or (None, 0) if nothing usable was read
    """
    if plate_img is None:
        return None, 0

    processed = _preprocess(plate_img)
    results = reader.readtext(processed)

    if not results:
        return None, 0

    # a plate crop can contain multiple separate text fragments - e.g.
    # Indian HSRP plates print a small high-contrast "IND" country-code
    # sticker next to the actual registration number. Originally I only
    # kept the single highest-confidence fragment, which meant "IND"
    # (very easy to read, often 90-100% confidence) kept winning over
    # the real plate number sitting right next to it and getting
    # discarded before it was ever checked.
    #
    # Fix: sort all fragments by confidence, then walk through them and
    # use the first one that actually looks like a plate (passes the
    # length + letter+digit checks) instead of blindly trusting whichever
    # fragment scored highest.
    results_sorted = sorted(results, key=lambda r: r[2], reverse=True)

    for candidate in results_sorted:
        raw_text = candidate[1]
        confidence = candidate[2]
        print(f"[DEBUG] raw OCR candidate: '{raw_text}' conf={confidence:.2f}")  # DEBUG

        if confidence < config.OCR_CONF_THRESHOLD:
            continue

        cleaned_text = clean_plate_text(raw_text)

        if len(cleaned_text) < config.MIN_PLATE_LENGTH:
            print(f"[DEBUG] rejected '{cleaned_text}' - too short to be a plate")  # DEBUG
            continue

        has_digit = any(ch.isdigit() for ch in cleaned_text)
        has_letter = any(ch.isalpha() for ch in cleaned_text)
        if not (has_digit and has_letter):
            print(f"[DEBUG] rejected '{cleaned_text}' - not a plausible plate (no letter+digit mix)")  # DEBUG
            continue

        # first fragment that survives all the checks - use it
        return cleaned_text, confidence

    # nothing in this crop passed all the checks
    return None, 0


def clean_plate_text(text):
    """
    Basic cleanup: uppercase, strip spaces, drop characters that
    plates never contain (OCR sometimes picks up junk like '.' or '_').
    """
    text = text.upper().strip()
    text = "".join(ch for ch in text if ch.isalnum())
    return text