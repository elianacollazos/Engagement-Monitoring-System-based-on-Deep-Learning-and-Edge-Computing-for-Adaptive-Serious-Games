# ==========================================
# LEVEL MAPPING (EMOTION + VA + COGNITIVE)
# ==========================================

"""
This module maps:
- Emotion -> base engagement level
- Valence / Arousal -> affective intensity
- Cognitive attention -> cognitive engagement level
- Level -> gauge percentage and color
"""

import math

EMOTION_TO_BASE_LEVEL = {
    "Surprise": "Fully Engaged",
    "Happy": "Highly Engaged",
    "Anger": "Highly Engaged",
    "Fear": "Highly Engaged",
    "Neutral": "Engaged",
    "Sad": "Disengaged",
    "Disgust": "Disengaged",
    "Contempt": "Disengaged",
}

LEVEL_RANGES = {
    "Disengaged": (0, 25),
    "Engaged": (25, 50),
    "Highly Engaged": (50, 75),
    "Fully Engaged": (75, 100),
}

LEVEL_COLORS = {
    "Disengaged": "red",
    "Engaged": "orange",
    "Highly Engaged": "yellow",
    "Fully Engaged": "green",
}


def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def get_base_emotional_level(emotion):
    return EMOTION_TO_BASE_LEVEL.get(emotion, "Engaged")


def compute_affective_intensity(valence, arousal):
    intensity = math.sqrt(valence**2 + arousal**2) / math.sqrt(2)
    return float(clamp(intensity))


def get_level_percentage(level, intensity):
    """
    Converts a level into a gauge percentage.

    Disengaged: 0-25
    Engaged: 25-50
    Highly Engaged: 50-75
    Fully Engaged: 75-100
    """
    min_value, max_value = LEVEL_RANGES.get(level, LEVEL_RANGES["Engaged"])
    intensity = clamp(intensity)

    percentage = min_value + intensity * (max_value - min_value)
    return round(percentage, 1)


def get_level_color(level):
    return LEVEL_COLORS.get(level, "orange")


def get_cognitive_level(att):
    att = clamp(att)

    if att >= 0.75:
        return "Fully Engaged"
    elif att >= 0.5:
        return "Highly Engaged"
    elif att >= 0.25:
        return "Engaged"
    else:
        return "Disengaged"


def get_emotional_level_from_emotion_va(emotion, valence, arousal):
    base_level = get_base_emotional_level(emotion)
    intensity = compute_affective_intensity(valence, arousal)

    percentage = get_level_percentage(base_level, intensity)
    color = get_level_color(base_level)
    label = f"{base_level} {percentage}%"

    return base_level, percentage, label, intensity, color


def compute_all_levels(emotion, valence, arousal, cognitive_engagement):
    emo_base, emo_percentage, emo_label, affective_intensity, emo_color = (
        get_emotional_level_from_emotion_va(emotion, valence, arousal)
    )

    cognitive_engagement = clamp(cognitive_engagement)
    cog_level = get_cognitive_level(cognitive_engagement)
    cog_percentage = round(cognitive_engagement * 100, 1)
    cog_color = get_level_color(cog_level)
    cog_label = f"{cog_level} {cog_percentage}%"

    return (
        emo_base,
        emo_percentage,
        emo_label,
        affective_intensity,
        emo_color,
        cog_level,
        cog_percentage,
        cog_label,
        cog_color,
    ) 



