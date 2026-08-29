# ─── Stage 1: builder ────────────────────────────────────────────────────────
# Install all Python deps into a clean prefix so we can copy just that
# into the final image without pip's cache or build tools.
FROM python:3.11-slim AS builder

WORKDIR /install

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install/pkgs -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu


# ─── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim

# OpenCV needs these system libraries (libGL for cv2, libGLib for EasyOCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install/pkgs /usr/local

WORKDIR /app

# Copy source code (weights and venv are excluded via .dockerignore)
COPY . .

# Pre-download YOLO weights into yolo-weights/ at build time so the
# container doesn't hit the internet on every first run
RUN mkdir -p yolo-weights data && \
    python -c "\
from ultralytics import YOLO; \
import shutil, os; \
m = YOLO('yolov8n.pt'); \
os.makedirs('yolo-weights', exist_ok=True); \
shutil.move('yolov8n.pt', 'yolo-weights/yolov8n.pt') \
"

# Pre-download EasyOCR English model weights
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# data/ is a volume so logs persist across container restarts
VOLUME ["/app/data"]

# Default: headless mode processing webcam 0.
# Override --source with a video file path for Docker use:
#   docker run -v $(pwd)/video.mp4:/app/input.mp4 sristi1/anpr-system \
#              python main.py --source /app/input.mp4 --headless
ENTRYPOINT ["python", "main.py"]
CMD ["--headless", "--source", "0"]
