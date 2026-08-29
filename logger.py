"""
logger.py
Handles the "database" part of the project — kept as a CSV on
purpose. The goal of this project is proving detection + OCR +
entry/exit logic works end to end, not building a backend. A CSV
is honest about that and is easy to explain in an interview.

CSV columns: plate_number, status (IN/OUT), timestamp

Logic:
- If a plate has never been seen -> log it as IN
- If the last record for that plate was IN -> next time we see it, log OUT
- If the last record was OUT -> next time we see it, log IN again
- A cooldown prevents logging the same plate every single frame while
  the car is sitting in view of the camera

FUZZY MATCHING NOTE:
OCR doesn't read the exact same string every frame — the same plate
can come out as "KA01AG1034", "KA01AG10345", "KA01AG1034E" across
different frames. If we compared strings exactly, each near-miss
reading would count as a brand new plate and get logged as a fresh
"IN". To fix that, before logging we check recently-seen plates for
a close string match (using difflib, which is Python's built-in
sequence-similarity tool) and treat a close-enough match as the
same car instead of a new one.
"""

import csv
import os
import time
import difflib
from datetime import datetime
import config

# tracks the last time (in seconds) each plate was logged,
# purely in-memory so we don't need to re-read the CSV every frame
_last_logged_time = {}

# how similar two plate strings need to be (0-1) to be treated as the
# same plate. Lowered to 0.65 (was 0.75) to better absorb OCR drift
# (e.g. 'KA01AB1034' vs 'KA01AB1034E'). Tradeoff: slightly higher risk
# of merging two genuinely different plates. If false merges appear,
# raise back toward 0.75.
SIMILARITY_THRESHOLD = 0.65   # was 0.75


def _find_matching_recent_plate(plate_number):
    """
    Checks plates we've logged recently (still within cooldown) for one
    that's a close text match to plate_number. Returns that plate's key
    if found, otherwise returns plate_number itself (treated as new).
    """
    best_match = None
    best_score = 0.0

    for known_plate in _last_logged_time:
        score = difflib.SequenceMatcher(None, plate_number, known_plate).ratio()
        if score > best_score:
            best_score = score
            best_match = known_plate

    if best_match and best_score >= SIMILARITY_THRESHOLD:
        return best_match
    return plate_number


def _init_log_file():
    """
    Creates the log file with a proper header if it doesn't exist yet.
    Also checks the header is actually what we expect - if the CSV was
    ever opened and re-saved in Excel, Excel can silently switch the
    delimiter (comma -> semicolon depending on Windows region settings),
    which breaks column names without the file looking obviously wrong.
    If that's happened, we back up the broken file and start a fresh one
    instead of crashing every time we try to read it.
    """
    if not os.path.exists(config.LOG_FILE):
        with open(config.LOG_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["plate_number", "status", "timestamp"])
        return

    with open(config.LOG_FILE, mode="r") as f:
        reader = csv.DictReader(f)
        header_ok = reader.fieldnames == ["plate_number", "status", "timestamp"]

    if not header_ok:
        backup_path = config.LOG_FILE + ".broken"
        print(f"[WARNING] log file header looks wrong, backing up to {backup_path} and starting fresh")
        os.replace(config.LOG_FILE, backup_path)
        with open(config.LOG_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["plate_number", "status", "timestamp"])


def _get_last_status(plate_number):
    """Reads the CSV and returns the last logged status for this plate, or None."""
    if not os.path.exists(config.LOG_FILE):
        return None

    last_status = None
    with open(config.LOG_FILE, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # .get() instead of row["plate_number"] so a stray malformed
            # row (e.g. a half-written line if the script was killed
            # mid-write) gets skipped instead of crashing the whole run
            if row.get("plate_number") == plate_number:
                last_status = row.get("status")
    return last_status


def log_plate(plate_number):
    """
    Decides IN or OUT for this plate and appends a row to the CSV.
    Returns the status that was logged, or None if skipped due to cooldown.
    """
    _init_log_file()

    # collapse near-duplicate OCR readings onto whichever plate string
    # we already started tracking, so "KA01AG1034" and "KA01AG10345"
    # count as the same car
    plate_number = _find_matching_recent_plate(plate_number)

    now = time.time()
    last_seen = _last_logged_time.get(plate_number, 0)

    if now - last_seen < config.LOG_COOLDOWN_SECONDS:
        # seen this plate (or a close match of it) too recently, skip
        return None

    last_status = _get_last_status(plate_number)
    new_status = "OUT" if last_status == "IN" else "IN"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(config.LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([plate_number, new_status, timestamp])

    _last_logged_time[plate_number] = now
    return new_status