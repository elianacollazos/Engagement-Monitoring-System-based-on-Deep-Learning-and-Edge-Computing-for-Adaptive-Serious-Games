import os
import tempfile
import time
from collections import Counter
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
from fpdf import FPDF
from matplotlib.patches import Wedge

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def format_duration(seconds):
    seconds = int(max(0, round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def make_temp_path(suffix):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def save_emotion_histogram(df, emotion_labels):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts = df["emotion"].value_counts().reindex(emotion_labels, fill_value=0)
    ax.bar(counts.index, counts.values, color="#4C78A8")
    ax.set_title("Emotion Frequency")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    tmp_img_path = make_temp_path(".png")
    fig.savefig(tmp_img_path, dpi=160)
    plt.close(fig)
    return tmp_img_path


def save_half_gauge(level, title, level_order):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#D62728", "#FF7F0E", "#FFD700", "#2CA02C"]
    selected_idx = level_order.index(level) if level in level_order else 0

    for idx, label in enumerate(level_order):
        theta1 = 180 - (idx + 1) * 45
        theta2 = 180 - idx * 45
        alpha = 1.0 if idx == selected_idx else 0.28
        width = 0.32 if idx == selected_idx else 0.24
        wedge = Wedge(
            (0, 0),
            1,
            theta1,
            theta2,
            width=width,
            facecolor=colors[idx],
            edgecolor="white",
            linewidth=2,
            alpha=alpha,
        )
        ax.add_patch(wedge)
        angle = np.deg2rad((theta1 + theta2) / 2)
        ax.text(
            0.78 * np.cos(angle),
            0.78 * np.sin(angle),
            label,
            ha="center",
            va="center",
            fontsize=8,
        )

    ax.text(0, -0.08, level, ha="center", va="center", fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=13)
    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-0.16, 1.08)
    ax.axis("off")
    fig.tight_layout()
    tmp_img_path = make_temp_path(".png")
    fig.savefig(tmp_img_path, dpi=160)
    plt.close(fig)
    return tmp_img_path


def save_cognitive_signals_chart(df):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    base_attention_col = "base_attention" if "base_attention" in df else "attention_score"
    ax.plot(df["t"], df[base_attention_col], label="Base Attention")
    ax.plot(df["t"], df["gaze_score"], label="Gaze Score")
    ax.set_title("Cognitive Signals Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.05)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    tmp_img_path = make_temp_path(".png")
    fig.savefig(tmp_img_path, dpi=160)
    plt.close(fig)
    return tmp_img_path


def save_emotion_changes_chart(df, emotion_labels):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    emotion_to_y = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
    y_values = df["emotion"].map(emotion_to_y)

    ax.step(df["t"], y_values, where="post", linewidth=2, color="#4C78A8")
    ax.scatter(df["t"], y_values, color="#E45756", s=28, zorder=3)
    ax.set_title("Emotion Changes Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Emotion")
    ax.set_yticks(range(len(emotion_labels)))
    ax.set_yticklabels(emotion_labels)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    tmp_img_path = make_temp_path(".png")
    fig.savefig(tmp_img_path, dpi=160)
    plt.close(fig)
    return tmp_img_path


def save_engagement_timeline_chart(df):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(df["t"], df["emo_window_pct"], linewidth=2, label="Emotional engagement")
    ax.plot(df["t"], df["attention"], linewidth=2, label="Cognitive engagement")
    ax.set_title("Engagement Percentage Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Engagement (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    tmp_img_path = make_temp_path(".png")
    fig.savefig(tmp_img_path, dpi=160)
    plt.close(fig)
    return tmp_img_path


def get_dominant_emotion(observed_emotions):
    if not observed_emotions:
        return "No reliable emotion detected"
    return Counter(observed_emotions).most_common(1)[0][0]


def get_emotional_change_metrics(df):
    if df.empty or len(df["emotion"]) < 2:
        return 0, 0.0

    emotion_changes = int((df["emotion"] != df["emotion"].shift()).sum() - 1)
    duration_minutes = max((df["time"].iloc[-1] - df["time"].iloc[0]) / 60.0, 1e-6)
    return emotion_changes, emotion_changes / duration_minutes


def generate_report(
    timeline,
    observed_emotions,
    user_name,
    session_id,
    performance_metrics,
    session_started_at,
    session_ended_at,
    level_order,
    level_map,
    level_percent_map,
    emotion_labels,
    calculate_emotional_engagement,
    calculate_emotional_engagement_percentage,
    get_cognitive_level,
):
    df = pd.DataFrame(timeline)
    df["t"] = df["time"] - df["time"].iloc[0]

    emo_final, _, _ = calculate_emotional_engagement(observed_emotions)
    cognitive_score = float(df["attention"].mean())
    cog_final = get_cognitive_level(cognitive_score)
    emotional_score = calculate_emotional_engagement_percentage(observed_emotions)
    dominant_emotion = get_dominant_emotion(observed_emotions)
    emotional_changes, emotional_changes_per_minute = get_emotional_change_metrics(df)
    conf_avg = float(df["confidence"].mean())
    session_end = session_ended_at or time.time()
    session_start = session_started_at or df["time"].iloc[0]
    session_duration = session_end - session_start
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df["emo_window_num"] = df["emo_level_window"].map(level_map)
    df["cog_num"] = df["cog_level"].map(level_map)
    if "emo_window_pct" not in df:
        df["emo_window_pct"] = df["emo_level_window"].map(level_percent_map)

    emotion_hist_path = save_emotion_histogram(df, emotion_labels)
    emotional_gauge_path = save_half_gauge(
        emo_final,
        "Emotional Engagement Level",
        level_order,
    )
    emotion_changes_path = save_emotion_changes_chart(df, emotion_labels)
    cognitive_signals_path = save_cognitive_signals_chart(df)
    cognitive_gauge_path = save_half_gauge(
        cog_final,
        "Cognitive Engagement Level",
        level_order,
    )
    engagement_timeline_path = save_engagement_timeline_chart(df)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)

    def add_pdf_heading(text):
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, txt=text, ln=True)
        pdf.set_font("Arial", size=11)

    def add_pdf_image(path, width=185, height=100):
        if pdf.get_y() + height > 282:
            pdf.add_page()
        pdf.image(path, x=12, y=pdf.get_y(), w=width)
        pdf.ln(height + 6)

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="Engagement Analysis Session Report", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, txt=f"User name: {user_name}", ln=True)
    pdf.cell(0, 8, txt=f"Session ID: {session_id}", ln=True)
    pdf.cell(0, 8, txt=f"Date: {report_date}", ln=True)
    pdf.cell(0, 8, txt=f"Session duration: {format_duration(session_duration)}", ln=True)
    pdf.cell(0, 8, txt=f"Average confidence: {conf_avg:.2f}", ln=True)
    pdf.ln(4)

    pdf.cell(0, 8, txt=f"Dominant emotion during the session: {dominant_emotion}", ln=True)
    pdf.ln(2)

    add_pdf_heading("Emotion Frequency Histogram")
    add_pdf_image(emotion_hist_path, height=104)

    add_pdf_heading("Emotional Engagement Level")
    pdf.cell(0, 8, txt=f"Emotional engagement level: {emo_final}", ln=True)
    pdf.cell(0, 8, txt=f"Average emotional engagement: {emotional_score:.2f}%", ln=True)
    pdf.ln(2)
    add_pdf_image(emotional_gauge_path, width=170, height=85)

    add_pdf_heading("Emotion Changes Over Time")
    pdf.cell(0, 8, txt=f"Emotional Changes: {emotional_changes}", ln=True)
    pdf.cell(
        0,
        8,
        txt=f"Emotional Changes per Minute: {emotional_changes_per_minute:.2f}",
        ln=True,
    )
    pdf.ln(2)
    add_pdf_image(emotion_changes_path, height=99)
    pdf.ln(4)

    add_pdf_heading("Cognitive Signals Over Time")
    add_pdf_image(cognitive_signals_path, height=99)

    add_pdf_heading("Cognitive Engagement Level")
    pdf.cell(0, 8, txt=f"Cognitive engagement level: {cog_final}", ln=True)
    pdf.ln(2)
    add_pdf_image(cognitive_gauge_path, width=170, height=85)

    pdf.cell(0, 8, txt=f"Average cognitive engagement: {cognitive_score:.2f}%", ln=True)
    pdf.ln(4)

    add_pdf_heading("Engagement Variation Over Time")
    add_pdf_image(engagement_timeline_path, height=99)

    add_pdf_heading("System Performance")
    pdf.cell(0, 8, txt=f"FPS: {performance_metrics['fps']:.2f}", ln=True)
    pdf.cell(0, 8, txt=f"Latency: {performance_metrics['latency']:.3f} s", ln=True)
    pdf.cell(0, 8, txt=f"Process CPU: {performance_metrics['cpu']:.1f}%", ln=True)
    pdf.cell(0, 8, txt=f"RAM: {performance_metrics['ram']:.1f} MB", ln=True)
    pdf.cell(0, 8, txt=f"Failures: {performance_metrics['failures']:.1f}%", ln=True)

    tmp_pdf_path = make_temp_path(".pdf")
    pdf.output(tmp_pdf_path)

    with open(tmp_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    return {
        "pdf_bytes": pdf_bytes,
        "emo_final": emo_final,
        "cog_final": cog_final,
        "performance_metrics": performance_metrics,
    }
