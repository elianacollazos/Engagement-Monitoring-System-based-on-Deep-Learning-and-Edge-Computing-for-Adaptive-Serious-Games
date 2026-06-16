# ==========================================
# EMOTION ENGAGEMENT ANALYZER
# ==========================================

import streamlit as st
import cv2
import pandas as pd
import time

from models.model_loader import load_models
from vision.face_mesh import create_face_mesh
from vision.eye_tracking import *
from emotion.engagement import *
from emotion.emotion_predictor import predict_emotion_va
from report.report_generator import generate_report
from emotion.circumplex import create_circumplex_plot
from emotion.levels import compute_all_levels

# ==========================================
# CONFIG
# ==========================================

st.set_page_config(
    page_title="Engagement Level Estimation",
    layout="wide"
)

st.title("Engagement Level Estimation")
st.caption("Real-time estimation of cognitive and emotional engagement from facial affective signals.")

# ==========================================
# SESSION DATA
# ==========================================

st.subheader("Session Data")

colA, colB = st.columns(2)

with colA:
    participant_name = st.text_input("Participant")

with colB:
    session_id = st.text_input("Session ID")

# ==========================================
# SESSION STATE
# ==========================================

if "running" not in st.session_state:
    st.session_state.running = False

if "history" not in st.session_state:
    st.session_state.history = []

if "prev_eye_center" not in st.session_state:
    st.session_state.prev_eye_center = None

if "camera" not in st.session_state:
    st.session_state.camera = None

if "final_plot" not in st.session_state:
    st.session_state.final_plot = None

if "live_plot_counter" not in st.session_state:
    st.session_state.live_plot_counter = 0

if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = None

if "session_end_time" not in st.session_state:
    st.session_state.session_end_time = None

if "session_duration_seconds" not in st.session_state:
    st.session_state.session_duration_seconds = None

if "final_metrics" not in st.session_state:
    st.session_state.final_metrics = None

# ==========================================
# VISUAL HELPERS
# ==========================================

def get_level_color(level):
    colors = {
        "Fully Engaged": "#d62828",
        "Highly Engaged": "#ee964b",
        "Engaged": "#f4d35e",
        "Disengaged": "#bdbdbd"
    }
    return colors.get(level, "#ffffff")


def get_level_color_bgr(level):
    colors = {
        "Fully Engaged": (0, 0, 255),
        "Highly Engaged": (0, 165, 255),
        "Engaged": (0, 255, 255),
        "Disengaged": (180, 180, 180)
    }
    return colors.get(level, (255, 255, 255))


def get_main_level_from_emo_label(emo_label):
    if not emo_label:
        return "Engaged"

    stages = [
        "Disengaged",
        "Engaged",
        "Highly Engaged",
        "Fully Engaged"
    ]

    for stage in stages:
        if emo_label.startswith(stage):
            return stage

    return "Engaged"


def render_level_bar(title, level, subtitle=""):
    levels = ["Disengaged", "Engaged", "Highly Engaged", "Fully Engaged"]

    blocks = ""
    for lv in levels:
        opacity = "1.0" if lv == level else "0.22"
        border = "3px solid #222222" if lv == level else "1px solid #bbbbbb"

        blocks += f"""<div style="flex:1;text-align:center;padding:10px 0;border-radius:10px;background:{get_level_color(lv)};opacity:{opacity};border:{border};font-weight:700;color:#111111;">{lv}</div>"""

    return f"""<div style="margin-top:12px;padding:12px;border-radius:12px;background:#f7f7f7;border:1px solid #dddddd;"><div style="font-weight:700;font-size:16px;margin-bottom:4px;">{title}</div><div style="font-size:14px;margin-bottom:8px;">{subtitle}</div><div style="display:flex;gap:8px;">{blocks}</div></div>"""

def format_duration(seconds):
    if seconds is None:
        return "N/A"

    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

# ==========================================
# LOAD MODELS
# ==========================================

model_emotion, model_va = load_models()
face_mesh = create_face_mesh()

# ==========================================
# CONTROLS
# ==========================================

col1, col2 = st.columns(2)

with col1:
    if not st.session_state.running:
        if st.button("Start"):
            if st.session_state.camera is None:
                st.session_state.camera = cv2.VideoCapture(1)
                st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            st.session_state.running = True
            st.session_state.history = []
            st.session_state.final_plot = None
            st.session_state.prev_eye_center = None
            st.session_state.live_plot_counter = 0
            st.session_state.session_start_time = time.time()
            st.session_state.session_end_time = None
            st.session_state.session_duration_seconds = None


with col2:
    if st.session_state.running:
        if st.button("Stop"):
            st.session_state.running = False
            st.session_state.session_end_time = time.time()
            st.session_state.session_duration_seconds = st.session_state.session_end_time - st.session_state.session_start_time

            # Store final metrics
            import numpy as np

            if "frames_total" in st.session_state and st.session_state.frames_total > 0:
                lat_mean = np.mean(st.session_state.latencias)
                fps = 1 / lat_mean if lat_mean > 0 else 0
                cpu_mean = np.mean(st.session_state.cpu)
                ram_mean = np.mean(st.session_state.ram)
                failure_pct = (st.session_state.failures / st.session_state.frames_total) * 100

                st.session_state.final_metrics = {
                    "fps": fps,
                    "latency": lat_mean,
                    "cpu": cpu_mean,
                    "ram": ram_mean,
                    "failures": failure_pct
                }

            if st.session_state.camera is not None:
                st.session_state.camera.release()
                st.session_state.camera = None

            if len(st.session_state.history) > 0:
                st.session_state.final_plot = create_circumplex_plot(
                    st.session_state.history,
                    show_trajectory=False
                )

# ==========================================
# PLACEHOLDERS
# ==========================================

top_col1, top_col2 = st.columns([1.6, 1])

with top_col1:
    frame_placeholder = st.empty()

with top_col2:
    info_placeholder = st.empty()

performance_placeholder = st.empty()
plot_placeholder = st.empty()

# ==========================================
# VIDEO LOOP
# ==========================================

if st.session_state.running and st.session_state.camera is not None:
    cap = st.session_state.camera

    # Initialize performance metrics
    import psutil
    import numpy as np
    process = psutil.Process()

    if "latencias" not in st.session_state:
        st.session_state.latencias = []
    if "cpu" not in st.session_state:
        st.session_state.cpu = []
    if "ram" not in st.session_state:
        st.session_state.ram = []
    if "failures" not in st.session_state:
        st.session_state.failures = 0
    if "frames_total" not in st.session_state:
        st.session_state.frames_total = 0

    while st.session_state.running:

        # Start measurement
        start_frame = time.time()

        ret, frame = cap.read()

        if not ret:
            st.session_state.failures += 1
            st.error("Unable to access the camera")
            st.session_state.running = False
            break

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_mesh = face_mesh.process(rgb)

        # Failure when no face is detected
        if not results_mesh.multi_face_landmarks:
            st.session_state.failures += 1

        if results_mesh.multi_face_landmarks:
            landmarks = results_mesh.multi_face_landmarks[0].landmark

            x_coords = [int(p.x * w) for p in landmarks]
            y_coords = [int(p.y * h) for p in landmarks]

            x, y = min(x_coords), min(y_coords)
            bw = max(x_coords) - x
            bh = max(y_coords) - y

            margin = 0.25

            x1 = int(max(0, x - bw * margin))
            y1 = int(max(0, y - bh * margin))
            x2 = int(min(w, x + bw * (1 + margin)))
            y2 = int(min(h, y + bh * (1 + margin)))

            face = frame[y1:y2, x1:x2]

            left_eye = get_eye_points(landmarks, LEFT_EYE_IDX, w, h)
            right_eye = get_eye_points(landmarks, RIGHT_EYE_IDX, w, h)
            iris_left = get_iris_center(landmarks, LEFT_IRIS_IDX, w, h)
            iris_right = get_iris_center(landmarks, RIGHT_IRIS_IDX, w, h)

            (
                cognitive_engagement,
                eye_center,
                blink_score,
                gaze_score,
                head_pose_score,
                base_attention,
            ) = compute_cognitive_engagement(
                left_eye,
                right_eye,
                st.session_state.prev_eye_center,
                iris_left,
                iris_right,
                landmarks,
                w,
                h,
            )


            st.session_state.prev_eye_center = eye_center

            if face.size != 0:
                result = predict_emotion_va(face, model_emotion, model_va)

                # Model failure
                if result[0] is None:
                    st.session_state.failures += 1

                if result[0] is not None:
                    label, score, valence, arousal = result

                    emotional_engagement = compute_engagement_emotional(valence, arousal)

                    (
                        emo_base,
                        emo_percentage,
                        emo_label,
                        affective_intensity,
                        emo_color,
                        cog_level,
                        cog_percentage,
                        cog_label,
                        cog_color,
                    ) = compute_all_levels(
                        label,
                        valence,
                        arousal,
                        cognitive_engagement
                    )


                    emo_main_level = get_main_level_from_emo_label(emo_label)

                    st.session_state.history.append({
                        "timestamp": time.time(),
                        "emotion": label,
                        "score": score,
                        "valence": valence,
                        "arousal": arousal,
                        "cognitive_engagement": cognitive_engagement,
                        "emotional_engagement": emotional_engagement,
                        "blink_score": blink_score,
                        "gaze_score": gaze_score,
                        "head_pose_score": head_pose_score,
                        "base_attention": base_attention,
                        "emo_base": emo_base,
                        "emo_percentage": emo_percentage,
                        "emo_label": emo_label,
                        "affective_intensity": affective_intensity,
                        "emo_color": emo_color,
                        "cog_level": cog_level,
                        "cog_percentage": cog_percentage,
                        "cog_label": cog_label,
                        "cog_color": cog_color
                    })

                    text_y1 = max(30, y - 60)
                    text_y2 = max(55, y - 35)
                    text_y3 = max(80, y - 10)
                    text_y4 = min(h - 45, y + bh + 20)
                    text_y5 = min(h - 20, y + bh + 45)

                    cv2.putText(
                        frame,
                        f"Emotion: {label}",
                        (x, text_y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Valence: {valence:.2f}",
                        (x, text_y2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Arousal: {arousal:.2f}",
                        (x, text_y3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 200, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Emotional Engagement: {emo_label}",
                        (x, text_y4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        get_level_color_bgr(emo_main_level),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Cognitive Engagement: {cog_label}",
                        (x, text_y5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        get_level_color_bgr(cog_level),
                        2
                    )

                    info_html = (
                        f'<div style="padding:14px;border-radius:14px;background:#ffffff;border:1px solid #dddddd;">'
                        f'<div style="font-size:20px;font-weight:700;margin-bottom:10px;">Current Status</div>'
                        f'<div style="font-size:15px;margin-bottom:6px;"><b>Detected Emotion:</b> {label}</div>'
                        f'<div style="font-size:15px;margin-bottom:6px;"><b>Valence:</b> {valence:.2f}</div>'
                        f'<div style="font-size:15px;margin-bottom:6px;"><b>Arousal:</b> {arousal:.2f}</div>'
                        f'<div style="font-size:15px;margin-bottom:6px;"><b>Affective intensity:</b> {affective_intensity:.2f}</div>'
                        f'<div style="font-size:15px;margin-bottom:6px;"><b>Emotional Engagement Level:</b> {emo_label}</div>'
                        f'<div style="font-size:15px;margin-bottom:6px;"><b>Cognitive Engagement Level:</b> {cog_label}</div>'
                        f'{render_level_bar("Emotional Engagement", emo_main_level, emo_label)}'
                        f'{render_level_bar("Cognitive Engagement", cog_level, cog_label)}'
                        f'</div>'
                    )

                    info_placeholder.markdown(info_html, unsafe_allow_html=True)

        frame_placeholder.image(frame, channels="BGR", width="stretch")

        # End measurement
        end_frame = time.time()
        latencia = end_frame - start_frame

        st.session_state.latencias.append(latencia)

        cpu = process.cpu_percent(interval=None)
        ram = process.memory_info().rss / (1024**2)

        st.session_state.cpu.append(cpu)
        st.session_state.ram.append(ram)

        st.session_state.frames_total += 1

        # Show performance metrics
        if st.session_state.frames_total > 20:
            lat_mean = np.mean(st.session_state.latencias)
            fps = 1 / lat_mean if lat_mean > 0 else 0
            cpu_mean = np.mean(st.session_state.cpu)
            ram_mean = np.mean(st.session_state.ram)
            failure_pct = (st.session_state.failures / st.session_state.frames_total) * 100

            performance_placeholder.markdown(
                f"""
                <div style="padding:12px;border-radius:10px;background:#eef3ff;border:1px solid #99aaff;">
                <b>System Performance</b><br>
                FPS: {fps:.2f}<br>
                Latency: {lat_mean:.4f} s<br>
                CPU: {cpu_mean:.2f}%<br>
                RAM: {ram_mean:.2f} MB<br>
                Failures: {failure_pct:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        time.sleep(0.01)

    if cap is not None:
        cap.release()

# ==========================================
# FINAL CHART
# ==========================================

if st.session_state.final_plot is not None:
    st.subheader("Valence-Arousal Circumplex")
    st.plotly_chart(
        st.session_state.final_plot,
        use_container_width=True,
        key="final_plot",
        config={"displayModeBar": False}
    )

    # FINAL METRICS
    if st.session_state.final_metrics is not None:
        m = st.session_state.final_metrics

        st.markdown(
            f"""
            <div style="margin-top:20px;padding:16px;border-radius:12px;background:#eef3ff;border:1px solid #99aaff;">
            <div style="font-size:18px;font-weight:700;margin-bottom:10px;">System Performance Summary</div>
            FPS: {m['fps']:.2f}<br>
            Latency: {m['latency']:.4f} s<br>
            CPU: {m['cpu']:.2f}%<br>
            RAM: {m['ram']:.2f} MB<br>
            Failures: {m['failures']:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.session_state.session_duration_seconds is not None:
        st.caption(
            f"Session Duration: {format_duration(st.session_state.session_duration_seconds)}"
        )

# ==========================================
# REPORT
# ==========================================

if st.button("Generate Report"):
    if len(st.session_state.history) == 0:
        st.warning("No data available yet")
    else:
        df = pd.DataFrame(st.session_state.history)

        try:
            pdf_buffer = generate_report(
                df,
                participant_name,
                session_id,
                st.session_state.session_duration_seconds
            )

            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="engagement_report.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Error generating report: {e}")
