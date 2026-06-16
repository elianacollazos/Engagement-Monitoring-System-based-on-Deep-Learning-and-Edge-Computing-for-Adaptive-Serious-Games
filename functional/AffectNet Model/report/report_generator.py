import io
import math
import tempfile
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fpdf import FPDF


EMOTION_COLORS = {
    "Neutral": "#6b7280",
    "Happy": "#22c55e",
    "Sad": "#3b82f6",
    "Surprise": "#facc15",
    "Fear": "#8b5cf6",
    "Disgust": "#a16207",
    "Anger": "#ef4444",
    "Contempt": "#ec4899",
}


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


def get_level_label_from_percentage(percentage):
    percentage = max(0, min(100, percentage))

    if percentage >= 75:
        level = "Fully Engaged"
    elif percentage >= 50:
        level = "Highly Engaged"
    elif percentage >= 25:
        level = "Engaged"
    else:
        level = "Disengaged"

    return f"{level} {percentage:.1f}%"


def create_emotion_histogram(df):
    emotion_counts = df["emotion"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        emotion_counts.index,
        emotion_counts.values,
        color="#2563eb",
        edgecolor="#0f172a",
        linewidth=1,
    )


    ax.set_title("Emotion Frequency Histogram", fontsize=13, fontweight="bold")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)

    for spine in ax.spines.values():
        spine.set_visible(False)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=200)
    buffer.seek(0)
    plt.close(fig)

    return buffer


def create_engagement_gauge(percentage, title, label):
    percentage = max(0, min(100, percentage))

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.set_aspect("equal")
    ax.axis("off")

    segments = [
        (0, 25, "#ef4444"),
        (25, 50, "#f97316"),
        (50, 75, "#facc15"),
        (75, 100, "#22c55e"),
    ]

    center = (0, 0)
    radius = 1.0
    width = 0.22

    for start, end, color in segments:
        theta1 = 180 - (end / 100) * 180
        theta2 = 180 - (start / 100) * 180

        wedge = patches.Wedge(
            center,
            radius,
            theta1,
            theta2,
            width=width,
            facecolor=color,
            edgecolor="#0f172a",
            linewidth=1.5,
        )
        ax.add_patch(wedge)

    segment_labels = [
        (12.5, "D"),
        (37.5, "E"),
        (62.5, "HE"),
        (87.5, "FE"),
    ]

    for value, text in segment_labels:
        angle = math.radians(180 - (value / 100) * 180)
        label_radius = radius + 0.18

        x = label_radius * math.cos(angle)
        y = label_radius * math.sin(angle)

        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
        )

    outer_border = patches.Wedge(
        center,
        radius + 0.08,
        0,
        180,
        width=0.08,
        facecolor="none",
        edgecolor="#0f172a",
        linewidth=2,
    )
    ax.add_patch(outer_border)

    inner_border = patches.Wedge(
        center,
        radius - width,
        0,
        180,
        width=0.04,
        facecolor="none",
        edgecolor="#0f172a",
        linewidth=2,
    )
    ax.add_patch(inner_border)

    for tick in range(0, 101, 10):
        angle = math.radians(180 - (tick / 100) * 180)

        r_outer = radius - 0.07
        r_inner = radius - 0.16 if tick % 25 == 0 else radius - 0.12

        x1 = r_inner * math.cos(angle)
        y1 = r_inner * math.sin(angle)
        x2 = r_outer * math.cos(angle)
        y2 = r_outer * math.sin(angle)

        ax.plot([x1, x2], [y1, y2], color="#0f172a", linewidth=1.4)

    needle_angle = math.radians(180 - (percentage / 100) * 180)
    needle_length = radius - 0.18

    needle_x = needle_length * math.cos(needle_angle)
    needle_y = needle_length * math.sin(needle_angle)

    ax.plot(
        [0, needle_x],
        [0, needle_y],
        color="#dc2626",
        linewidth=4,
        solid_capstyle="round",
        zorder=5,
    )

    ax.plot(
        [0, needle_x],
        [0, needle_y],
        color="#0f172a",
        linewidth=1.2,
        solid_capstyle="round",
        zorder=6,
    )

    hub = patches.Circle(
        center,
        0.09,
        facecolor="#94a3b8",
        edgecolor="#0f172a",
        linewidth=2,
        zorder=7,
    )
    ax.add_patch(hub)

    base = patches.FancyBboxPatch(
        (-1.08, -0.13),
        2.16,
        0.16,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor="#cbd5e1",
        edgecolor="#0f172a",
        linewidth=2,
        zorder=0,
    )
    ax.add_patch(base)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.35, 1.38)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.text(
        0,
        -0.27,
        label,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=200)
    buffer.seek(0)
    plt.close(fig)

    return buffer


def create_emotion_changes_curve(df):
    emotions = list(df["emotion"].dropna().unique())
    emotion_to_num = {emotion: idx for idx, emotion in enumerate(emotions)}
    emotion_numeric = df["emotion"].map(emotion_to_num)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.step(
        range(len(df)),
        emotion_numeric,
        where="post",
        color="#7c3aed",
        linewidth=2,
        label="Emotion",
    )

    ax.set_yticks(list(emotion_to_num.values()))
    ax.set_yticklabels(list(emotion_to_num.keys()))
    ax.set_xlabel("Frame")
    ax.set_ylabel("Emotion")
    ax.set_title("Emotion Changes Over Time", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    for spine in ax.spines.values():
        spine.set_visible(False)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=200)
    buffer.seek(0)
    plt.close(fig)

    return buffer


def create_valence_arousal_plot(df):
    fig, ax = plt.subplots(figsize=(7.5, 5))

    for emotion, group in df.groupby("emotion"):
        ax.scatter(
            group["valence"],
            group["arousal"],
            alpha=0.65,
            s=35,
            color=EMOTION_COLORS.get(emotion, "#374151"),
            edgecolors="white",
            linewidth=0.4,
            label=emotion,
        )

    ax.set_title("Valence-Arousal Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.axhline(0, color="#9ca3af", linewidth=1)
    ax.axvline(0, color="#9ca3af", linewidth=1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.2)

    ax.legend(
        title="Emotion",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=200)
    buffer.seek(0)
    plt.close(fig)

    return buffer


def create_attention_gaze_plot(df):
    required_cols = ["base_attention", "gaze_score"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        df["base_attention"].values,
        label="Base Attention",
        color="#2563eb",
        linewidth=2,
    )

    ax.plot(
        df["gaze_score"].values,
        label="Gaze Score",
        color="#16a34a",
        linewidth=2,
    )

    ax.set_title("Base Attention and Gaze Score Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    for spine in ax.spines.values():
        spine.set_visible(False)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=200)
    buffer.seek(0)
    plt.close(fig)

    return buffer


def create_engagement_components_plot(df):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        df["cognitive_engagement"],
        label="Cognitive Engagement",
        color="#2563eb",
        linewidth=2,
    )

    ax.plot(
        df["emotional_engagement"],
        label="Emotional Engagement",
        color="#ea580c",
        linewidth=2,
    )

    ax.set_ylim(0, 1)
    ax.set_title("Engagement Components Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Score")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    for spine in ax.spines.values():
        spine.set_visible(False)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=200)
    buffer.seek(0)
    plt.close(fig)

    return buffer


def add_image(pdf, image_buffer, x, w):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(image_buffer.getvalue())
        pdf.image(tmp.name, x=x, w=w)


def generate_report(df, participant_name, session_id, session_duration_seconds=None):
    required_cols = [
        "valence",
        "arousal",
        "emotion",
        "cognitive_engagement",
        "emotional_engagement",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df.dropna()

    if df.empty:
        raise ValueError("The dataframe is empty after removing NaN values.")

    avg_cognitive = df["cognitive_engagement"].mean()
    avg_emotional = df["emotional_engagement"].mean()
    avg_valence = df["valence"].mean()
    avg_arousal = df["arousal"].mean()

    avg_confidence = (
        df["score"].mean()
        if "score" in df.columns else None
    )

    avg_affective_intensity = (
        df["affective_intensity"].mean()
        if "affective_intensity" in df.columns else None
    )

    dominant_emotion = (
        df["emotion"].mode().iloc[0]
        if not df["emotion"].mode().empty else "N/A"
    )

    dominant_emo_level = (
        df["emo_base"].mode().iloc[0]
        if "emo_base" in df.columns and not df["emo_base"].mode().empty else "N/A"
    )

    dominant_cog_level = (
        df["cog_level"].mode().iloc[0]
        if "cog_level" in df.columns and not df["cog_level"].mode().empty else "N/A"
    )

    emotion_changes = (df["emotion"] != df["emotion"].shift()).sum() - 1
    emotion_changes = max(0, int(emotion_changes))

    duration_minutes = (
        session_duration_seconds / 60
        if session_duration_seconds and session_duration_seconds > 0
        else None
    )

    emotion_changes_per_minute = (
        emotion_changes / duration_minutes
        if duration_minutes else None
    )

    emotional_gauge_percentage = (
        df["emo_percentage"].mean()
        if "emo_percentage" in df.columns else avg_emotional * 100
    )

    cognitive_gauge_percentage = (
        df["cog_percentage"].mean()
        if "cog_percentage" in df.columns else avg_cognitive * 100
    )

    emotional_gauge_label = get_level_label_from_percentage(emotional_gauge_percentage)
    cognitive_gauge_label = get_level_label_from_percentage(cognitive_gauge_percentage)

    emotion_histogram = create_emotion_histogram(df)
    emotional_gauge = create_engagement_gauge(
        emotional_gauge_percentage,
        "Emotional Engagement Gauge",
        emotional_gauge_label,
    )
    emotion_changes_curve = create_emotion_changes_curve(df)
    valence_arousal_plot = create_valence_arousal_plot(df)
    cognitive_gauge = create_engagement_gauge(
        cognitive_gauge_percentage,
        "Cognitive Engagement Gauge",
        cognitive_gauge_label,
    )
    attention_gaze_plot = create_attention_gaze_plot(df)
    engagement_components_plot = create_engagement_components_plot(df)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "ENGAGEMENT ANALYSIS REPORT", ln=True, align="C")

    pdf.ln(3)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())

    pdf.ln(6)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"User Name: {participant_name or 'N/A'}", ln=True)
    pdf.cell(0, 7, f"Session ID: {session_id or 'N/A'}", ln=True)
    pdf.cell(0, 7, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 7, f"Session Duration: {format_duration(session_duration_seconds)}", ln=True)

    if avg_confidence is not None:
        pdf.cell(0, 7, f"Average Confidence: {avg_confidence * 100:.2f}%", ln=True)
    else:
        pdf.cell(0, 7, "Average Confidence: N/A", ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Emotion Frequency Histogram", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Dominant Emotion: {dominant_emotion}", ln=True)

    pdf.ln(2)
    add_image(pdf, emotion_histogram, x=20, w=170)

    pdf.add_page()

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Emotional Engagement Level", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Dominant Emotional Level: {dominant_emo_level}", ln=True)
    pdf.cell(0, 7, f"Average Emotional Engagement: {avg_emotional * 100:.2f}%", ln=True)

    if avg_affective_intensity is not None:
        pdf.cell(0, 7, f"Average Affective Intensity: {avg_affective_intensity:.3f}", ln=True)

    pdf.ln(2)
    add_image(pdf, emotional_gauge, x=35, w=140)

    pdf.add_page()

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Emotion Changes Over Time", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Emotional Changes: {emotion_changes}", ln=True)

    if emotion_changes_per_minute is not None:
        pdf.cell(0, 7, f"Emotional Changes per Minute: {emotion_changes_per_minute:.2f}", ln=True)
    else:
        pdf.cell(0, 7, "Emotional Changes per Minute: N/A", ln=True)

    pdf.ln(2)
    add_image(pdf, emotion_changes_curve, x=12, w=186)

    pdf.add_page()

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Valence-Arousal Distribution", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Average Valence: {avg_valence:.3f}", ln=True)
    pdf.cell(0, 7, f"Average Arousal: {avg_arousal:.3f}", ln=True)

    pdf.multi_cell(
        0,
        7,
        "This chart shows the distribution of emotional states in the valence-arousal space. Each color represents a detected emotion.",
    )

    pdf.ln(2)
    add_image(pdf, valence_arousal_plot, x=18, w=175)

    pdf.add_page()

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Cognitive Engagement Level", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Dominant Cognitive Engagement Level: {dominant_cog_level}", ln=True)

    pdf.ln(2)
    add_image(pdf, cognitive_gauge, x=35, w=140)

    if attention_gaze_plot is not None:
        pdf.add_page()

        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Base Attention and Gaze Score", ln=True)

        pdf.ln(2)
        add_image(pdf, attention_gaze_plot, x=12, w=186)

    pdf.add_page()

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Engagement Components", ln=True)

    pdf.ln(2)
    add_image(pdf, engagement_components_plot, x=12, w=186)

    pdf_output = pdf.output(dest="S")

    if isinstance(pdf_output, str):
        pdf_output = pdf_output.encode("latin-1")

    buffer = io.BytesIO(pdf_output)
    return buffer
