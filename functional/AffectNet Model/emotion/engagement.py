import numpy as np

# ==========================================
# ENGAGEMENT EMOCIONAL (VA -> 0-1)
# ==========================================

def compute_engagement_emotional(valence, arousal):
    sa = (arousal + 1.0) / 2.0
    sv = (valence + 1.0) / 2.0
    e = 0.5 * sa + 0.5 * sv
    return float(np.clip(e, 0.0, 1.0))


# ==========================================
# SUAVIZADO (VENTANA TEMPORAL)
# ==========================================

def smooth_signal(buffer, value):
    buffer.append(value)
    return sum(buffer) / len(buffer)

