"""
app.py
Streamlit Web Application for ANPR - Automatic Number Plate Recognition & Vehicle Entry/Exit Logger
Wraps detector.py, plate_extractor.py, ocr_reader.py, and logger.py for live cloud deployment.
"""

import os
import cv2
import time
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

import config
from detector import detect_cars
from plate_extractor import extract_plate_region
from ocr_reader import read_plate_text, _preprocess
from logger import log_plate

# Page Configuration
st.set_page_config(
    page_title="ANPR System | Automatic Number Plate Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .stTable {
        font-size: 0.95rem;
    }
    </style>
""", unsafe_allow_html=True)


def process_frame(frame, draw_boxes=True):
    """
    Processes a single frame through the ANPR pipeline:
    Detection -> Crop -> CLAHE Preprocess -> OCR -> Fuzzy IN/OUT Logging
    """
    car_boxes = detect_cars(frame)
    annotated_frame = frame.copy()
    detection_results = []

    for car_box in car_boxes:
        x1, y1, x2, y2, car_conf = car_box
        
        # Draw car box (Blue)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)

        # Extract plate crop
        plate_img, plate_coords = extract_plate_region(frame, car_box)
        if plate_img is None:
            continue

        px1, py1, px2, py2 = plate_coords
        # Draw plate region box (Cyan)
        cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), color=(0, 255, 255), thickness=2)

        # Perform CLAHE + EasyOCR
        plate_text, ocr_conf = read_plate_text(plate_img)

        status = None
        if plate_text:
            status = log_plate(plate_text)
            label = f"{plate_text} ({ocr_conf:.2f})"
            if status:
                label += f" - {status}"
            
            # Put label banner
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated_frame, (px1, max(0, py1 - 25)), (px1 + w, max(0, py1)), (0, 255, 255), -1)
            cv2.putText(annotated_frame, label, (px1, max(18, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        detection_results.append({
            "car_box": (x1, y1, x2, y2, car_conf),
            "plate_coords": (px1, py1, px2, py2),
            "plate_raw": plate_img,
            "plate_proc": _preprocess(plate_img),
            "plate_text": plate_text,
            "ocr_conf": ocr_conf,
            "status": status
        })

    return annotated_frame, detection_results


def main():
    st.markdown('<div class="main-header">🚗 ANPR — Automatic Number Plate Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time vehicle license plate detection, CLAHE contrast enhancement, EasyOCR text extraction & IN/OUT logging</div>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("⚙️ Settings & Pipeline Controls")
    
    input_type = st.sidebar.radio(
        "Select Input Mode:",
        ["🖼️ Image Upload", "🎥 Video File Upload", "📷 Live Browser Camera"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Pipeline Tuning")
    config.CAR_CONF_THRESHOLD = st.sidebar.slider("YOLO Car Confidence", 0.1, 0.9, float(config.CAR_CONF_THRESHOLD), 0.05)
    config.OCR_CONF_THRESHOLD = st.sidebar.slider("OCR Confidence Threshold", 0.1, 0.9, float(config.OCR_CONF_THRESHOLD), 0.05)
    config.LOG_COOLDOWN_SECONDS = st.sidebar.slider("Log Cooldown (seconds)", 5, 60, int(config.LOG_COOLDOWN_SECONDS), 5)

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **CV Engineering Pipeline:**\n\n1. **YOLOv8n**: Vehicle detection\n2. **Heuristic Cropping**: Bottom-center 65–95%\n3. **CLAHE Normalization**: Local contrast boost\n4. **EasyOCR**: Letter/Digit verification\n5. **Difflib Fuzzy Match**: OCR-drift deduplication to CSV")

    # Layout Columns
    col_main, col_logs = st.columns([2, 1])

    with col_main:
        if input_type == "🖼️ Image Upload":
            st.subheader("Upload Vehicle Image")
            uploaded_file = st.file_uploader("Choose a car image (JPG/PNG)", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                with st.spinner("Running YOLO Detection + CLAHE Preprocessing + EasyOCR..."):
                    start_t = time.time()
                    annotated_frame, detections = process_frame(frame)
                    proc_time = (time.time() - start_t) * 1000

                # Display annotated result
                res_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                st.image(res_image, caption=f"Pipeline Output — Processed in {proc_time:.1f} ms", use_column_width=True)

                if detections:
                    st.markdown("### 🔍 Extracted Plate Pipeline Inspection")
                    cols = st.columns(min(len(detections), 3))
                    for idx, det in enumerate(detections):
                        with cols[idx % 3]:
                            st.write(f"**Vehicle #{idx+1}**")
                            if det["plate_raw"] is not None:
                                st.image(cv2.cvtColor(det["plate_raw"], cv2.COLOR_BGR2RGB), caption="Raw Plate Crop", width=160)
                                st.image(det["plate_proc"], caption="CLAHE Enhanced", width=160)
                            if det["plate_text"]:
                                st.success(f"**Plate:** `{det['plate_text']}`\n\n**Conf:** {det['ocr_conf']:.2f}")
                                if det["status"]:
                                    st.info(f"**Logged as:** {det['status']}")
                            else:
                                st.warning("No plate reading passed filters")
                else:
                    st.warning("No vehicles detected matching confidence threshold.")

        elif input_type == "🎥 Video File Upload":
            st.subheader("Upload Vehicle Video")
            uploaded_video = st.file_uploader("Choose a video file (MP4/AVI/MOV)", type=["mp4", "avi", "mov"])

            if uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                tfile.close()

                cap = cv2.VideoCapture(tfile.name)
                st_frame = st.empty()
                stop_button = st.button("Stop Processing")

                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    annotated_frame, _ = process_frame(frame)
                    res_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                    st_frame.image(res_image, use_column_width=True)

                cap.release()
                os.unlink(tfile.name)

        elif input_type == "📷 Live Browser Camera":
            st.subheader("Live Browser Camera")
            camera_image = st.camera_input("Take a photo of a vehicle license plate")

            if camera_image is not None:
                image = Image.open(camera_image).convert("RGB")
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                with st.spinner("Processing camera snapshot..."):
                    annotated_frame, detections = process_frame(frame)

                res_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                st.image(res_image, caption="ANPR Pipeline Result", use_column_width=True)

    with col_logs:
        st.subheader("📋 Vehicle Entry/Exit Log")
        
        if os.path.exists(config.LOG_FILE):
            try:
                df = pd.read_csv(config.LOG_FILE)
                if not df.empty:
                    # Display metrics
                    in_count = len(df[df['status'] == 'IN'])
                    out_count = len(df[df['status'] == 'OUT'])
                    
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("Vehicles IN", in_count)
                    m_col2.metric("Vehicles OUT", out_count)

                    st.dataframe(df.tail(15), use_container_width=True)

                    # Download button
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download CSV Log",
                        data=csv_data,
                        file_name="vehicle_entry_exit_log.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("Log file is empty. Process an image/video to record entries.")
            except Exception as e:
                st.error(f"Error reading log file: {e}")
        else:
            st.info("No log file created yet.")


if __name__ == "__main__":
    main()
