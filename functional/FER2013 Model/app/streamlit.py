import os
import sys
import time
import ctypes
import tracemalloc
from collections import Counter

import cv2
import numpy as np
import streamlit as st

try:
    import psutil
except ImportError:
    psutil = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import CONFIDENCE_THRESHOLD, MODEL_PATH, MODEL_TYPE, get_device
from app.generate_report import generate_report
from models import cognitive_eng as cog
from models.emotion_model import load_model, predict_emotion, preprocess_face


st.set_page_config(page_title="Engagement Level Estimation", layout="wide")
st.title("Engagement Level Estimation")

input_col1, input_col2 = st.columns(2)
with input_col1:
    user_name = st.text_input("User name", value="User")
with input_col2:
    session_id = st.text_input("Session ID", value="Session-001")

camera_index = 1
conf_thr = float(CONFIDENCE_THRESHOLD)
emotional_window_size = 20


device, _ = get_device()
model, _ = load_model(MODEL_PATH, MODEL_TYPE)
model.to(device)
model.eval()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

LEVEL_ORDER = ["Disengaged", "Engaged", "Highly Engaged", "Fully Engaged"]
LEVEL_MAP = {level: idx + 1 for idx, level in enumerate(LEVEL_ORDER)}
LEVEL_PERCENT_MAP = {
    "Disengaged": 25.0,
    "Engaged": 50.0,
    "Highly Engaged": 75.0,
    "Fully Engaged": 100.0,
}

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTIONAL_LEVEL_GROUPS = {
    "Fully Engaged": ["Surprise"],
    "Highly Engaged": ["Happy", "Angry", "Fear"],
    "Engaged": ["Neutral"],
    "Disengaged": ["Sad", "Disgust"],
}
EMOTION_TO_LEVEL = {
    emotion: level
    for level, emotions in EMOTIONAL_LEVEL_GROUPS.items()
    for emotion in emotions
}
LEVEL_IMPACT_VALUES = {
    level: 1.0 / len(emotions)
    for level, emotions in EMOTIONAL_LEVEL_GROUPS.items()
}

cognitive_estimator = cog.CognitiveEngagementEstimator(window_size=30, level_order=LEVEL_ORDER)

def get_emotional_level(emotion):
    return EMOTION_TO_LEVEL.get(emotion, "Disengaged")


def calculate_emotional_engagement(observed_emotions):
    scores = {level: 0.0 for level in LEVEL_ORDER}
    counts = {level: 0 for level in LEVEL_ORDER}
    total_observed = len(observed_emotions)

    if total_observed == 0:
        return "Disengaged", scores, counts

    level_counts = Counter(get_emotional_level(emotion) for emotion in observed_emotions)

    for level in LEVEL_ORDER:
        count = level_counts.get(level, 0)
        counts[level] = count
        if count == 0:
            continue

        impact_sum = count * LEVEL_IMPACT_VALUES[level]
        scores[level] = impact_sum * (count / total_observed)

    final_level = max(
        LEVEL_ORDER,
        key=lambda level: (scores[level], LEVEL_ORDER.index(level)),
    )
    return final_level, scores, counts


def get_cognitive_level(attention_pct):
    return cog.get_cognitive_level(attention_pct, LEVEL_ORDER)


def level_percent_value(level):
    return LEVEL_PERCENT_MAP.get(level, 25.0)


def calculate_emotional_engagement_percentage(observed_emotions):
    if not observed_emotions:
        return 0.0
    values = [level_percent_value(get_emotional_level(emotion)) for emotion in observed_emotions]
    return float(np.mean(values))


def collect_performance_sample(frame_latency, frame_failed):
    st.session_state.latencies.append(frame_latency)
    st.session_state.total_frames += 1
    if frame_failed:
        st.session_state.failures += 1

    if psutil is not None:
        process = st.session_state.get("process_monitor")
        if process is None:
            process = psutil.Process(os.getpid())
            process.cpu_percent(interval=None)
            st.session_state.process_monitor = process

        st.session_state.cpu.append(process.cpu_percent(interval=None))
        st.session_state.ram.append(get_process_memory_mb(process))
    else:
        wall_start = st.session_state.get("process_wall_start")
        cpu_start = st.session_state.get("process_cpu_start")
        if wall_start is not None and cpu_start is not None:
            wall_delta = max(time.time() - wall_start, 1e-6)
            cpu_delta = time.process_time() - cpu_start
            st.session_state.cpu.append((cpu_delta / wall_delta) * 100)
        st.session_state.ram.append(get_process_memory_mb())


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_process_memory_mb(process=None):
    if process is not None:
        return process.memory_info().rss / (1024 * 1024)

    if tracemalloc.is_tracing():
        _, peak = tracemalloc.get_traced_memory()
        return peak / (1024 * 1024)

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    success = ctypes.windll.psapi.GetProcessMemoryInfo(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    if not success:
        return 0.0
    return counters.WorkingSetSize / (1024 * 1024)


def calculate_final_metrics():
    lat_mean = float(np.mean(st.session_state.latencies)) if st.session_state.latencies else 0.0
    fps = 1 / lat_mean if lat_mean > 0 else 0.0
    cpu_mean = float(np.mean(st.session_state.cpu)) if st.session_state.cpu else 0.0
    ram_mean = float(np.mean(st.session_state.ram)) if st.session_state.ram else 0.0

    process = st.session_state.get("process_monitor")
    cpu_start = st.session_state.get("process_cpu_start")
    wall_start = st.session_state.get("process_wall_start")
    if cpu_start is not None and wall_start is not None:
        if psutil is not None and process is not None:
            current_times = process.cpu_times()
            cpu_delta = (current_times.user + current_times.system) - cpu_start
        else:
            cpu_delta = time.process_time() - cpu_start

        wall_delta = max(time.time() - wall_start, 1e-6)
        session_cpu_mean = (cpu_delta / wall_delta) * 100
        cpu_mean = session_cpu_mean if session_cpu_mean > 0 else cpu_mean
        ram_mean = get_process_memory_mb(process if psutil is not None else None)
    elif st.session_state.ram:
        ram_mean = float(np.mean(st.session_state.ram))

    failure_pct = (
        (st.session_state.failures / st.session_state.total_frames) * 100
        if st.session_state.total_frames > 0
        else 0.0
    )

    st.session_state.final_metrics = {
        "fps": fps,
        "latency": lat_mean,
        "cpu": cpu_mean,
        "ram": ram_mean,
        "failures": failure_pct,
    }
    return st.session_state.final_metrics


if "run" not in st.session_state:
    st.session_state.run = False
if "timeline" not in st.session_state:
    st.session_state.timeline = []
if "att_hist" not in st.session_state:
    st.session_state.att_hist = []
if "observed_emotions" not in st.session_state:
    st.session_state.observed_emotions = []
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "latencies" not in st.session_state:
    st.session_state.latencies = []
if "cpu" not in st.session_state:
    st.session_state.cpu = []
if "ram" not in st.session_state:
    st.session_state.ram = []
if "failures" not in st.session_state:
    st.session_state.failures = 0
if "total_frames" not in st.session_state:
    st.session_state.total_frames = 0
if "final_metrics" not in st.session_state:
    st.session_state.final_metrics = {}
if "process_monitor" not in st.session_state:
    st.session_state.process_monitor = None
if "process_cpu_start" not in st.session_state:
    st.session_state.process_cpu_start = None
if "process_wall_start" not in st.session_state:
    st.session_state.process_wall_start = None
if "session_started_at" not in st.session_state:
    st.session_state.session_started_at = None
if "session_ended_at" not in st.session_state:
    st.session_state.session_ended_at = None


col1, col2 = st.columns(2)
if col1.button("Start"):
    st.session_state.run = True
    st.session_state.timeline = []
    st.session_state.att_hist = []
    st.session_state.observed_emotions = []
    st.session_state.latencies = []
    st.session_state.cpu = []
    st.session_state.ram = []
    st.session_state.failures = 0
    st.session_state.total_frames = 0
    st.session_state.final_metrics = {}
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    else:
        tracemalloc.clear_traces()
    cognitive_estimator.reset()
    if psutil is not None:
        st.session_state.process_monitor = psutil.Process(os.getpid())
        st.session_state.process_monitor.cpu_percent(interval=None)
        process_times = st.session_state.process_monitor.cpu_times()
        st.session_state.process_cpu_start = process_times.user + process_times.system
        st.session_state.process_wall_start = time.time()
    else:
        st.session_state.process_monitor = None
        st.session_state.process_cpu_start = time.process_time()
        st.session_state.process_wall_start = time.time()
    st.session_state.session_started_at = time.time()
    st.session_state.session_ended_at = None

if col2.button("Stop"):
    st.session_state.run = False
    st.session_state.session_ended_at = time.time()


col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    frame_placeholder = st.empty()

performance_placeholder = st.empty()


if st.session_state.run:
    cap = cv2.VideoCapture(int(camera_index))

    while st.session_state.run:
        frame_start = time.time()
        frame_failed = False
        ret, frame = cap.read()
        if not ret:
            st.session_state.failures += 1
            break

        frame = cv2.flip(frame, 1)

        attention_score, attention_details, cog_level, att_pct = cognitive_estimator.update(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
            face = gray[y:y + h, x:x + w]
            tensor = preprocess_face(face).to(device)

            res = predict_emotion(model, tensor, conf_thr)
            confidence = float(res["confidence"])

            if res["is_reliable"]:
                emotion = EMOTION_LABELS[int(res["emotion"])]
                current_emotion_level = get_emotional_level(emotion)
                st.session_state.observed_emotions.append(emotion)

                window_emotions = st.session_state.observed_emotions[-emotional_window_size:]
                emotional_level_window, _, _ = calculate_emotional_engagement(window_emotions)
                emotional_level_session, _, _ = calculate_emotional_engagement(
                    st.session_state.observed_emotions
                )
                emotional_window_pct = calculate_emotional_engagement_percentage(window_emotions)
                emotional_session_pct = calculate_emotional_engagement_percentage(
                    st.session_state.observed_emotions
                )

                st.session_state.timeline.append(
                    {
                        "time": time.time(),
                        "emotion": emotion,
                        "emotion_level_current": current_emotion_level,
                        "emo_level_window": emotional_level_window,
                        "emo_level_session": emotional_level_session,
                        "emo_window_pct": emotional_window_pct,
                        "emo_session_pct": emotional_session_pct,
                        "cog_level": cog_level,
                        "attention": att_pct,
                        "attention_score": attention_score,
                        "base_attention": attention_details["base_attention"],
                        "blink_score": attention_details["blink_score"],
                        "head_score": attention_details["head_score"],
                        "gaze_score": attention_details["gaze_score"],
                        "yaw_score": attention_details["yaw_score"],
                        "pitch_score": attention_details["pitch_score"],
                        "confidence": confidence,
                    }
                )

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{emotion} ({confidence:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Emotional: {emotional_level_window}",
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Cognitive: {cog_level}",
                    (x, y + h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(
                    frame,
                    f"Low confidence ({confidence:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                )
                frame_failed = True
        else:
            frame_failed = True

        cv2.putText(
            frame,
            f"Attention: {att_pct:.1f}% ({cog_level})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Head: {attention_details['head_score']:.2f}  Gaze: {attention_details['gaze_score']:.2f}  Blink: {attention_details['blink_score']:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, use_container_width=True)
        st.session_state.last_frame = frame_rgb
        collect_performance_sample(time.time() - frame_start, frame_failed)

        if st.session_state.total_frames > 20:
            live_metrics = calculate_final_metrics()
            performance_placeholder.markdown(
                f"""
                <div style="padding:12px;border-radius:10px;background:#eef3ff;border:1px solid #99aaff;">
                <b>System Performance</b><br>
                FPS: {live_metrics['fps']:.2f}<br>
                Latency: {live_metrics['latency']:.4f} s<br>
                Process CPU: {live_metrics['cpu']:.2f}%<br>
                RAM: {live_metrics['ram']:.2f} MB<br>
                Failures: {live_metrics['failures']:.2f}%
                </div>
                """,
                unsafe_allow_html=True,
            )

    cap.release()
    st.session_state.session_ended_at = time.time()


if st.button("Generate report"):
    if not st.session_state.timeline:
        st.warning("No data available.")
    else:
        performance_metrics = calculate_final_metrics()
        report = generate_report(
            timeline=st.session_state.timeline,
            observed_emotions=st.session_state.observed_emotions,
            user_name=user_name,
            session_id=session_id,
            performance_metrics=performance_metrics,
            session_started_at=st.session_state.session_started_at,
            session_ended_at=st.session_state.session_ended_at,
            level_order=LEVEL_ORDER,
            level_map=LEVEL_MAP,
            level_percent_map=LEVEL_PERCENT_MAP,
            emotion_labels=EMOTION_LABELS,
            calculate_emotional_engagement=calculate_emotional_engagement,
            calculate_emotional_engagement_percentage=calculate_emotional_engagement_percentage,
            get_cognitive_level=get_cognitive_level,
        )

        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        col1.metric("Final emotional engagement", report["emo_final"])
        col2.metric("Final cognitive engagement", report["cog_final"])

        st.subheader("Performance Metrics")
        perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
        perf = report["performance_metrics"]
        perf_col1.metric("FPS", f"{perf['fps']:.2f}")
        perf_col2.metric("Latency", f"{perf['latency']:.3f} s")
        perf_col3.metric("Process CPU", f"{perf['cpu']:.1f}%")
        perf_col4.metric("RAM", f"{perf['ram']:.1f} MB")
        perf_col5.metric("Failures", f"{perf['failures']:.1f}%")

        st.download_button(
            "Download PDF",
            report["pdf_bytes"],
            file_name="engagement_session_report.pdf",
        )
