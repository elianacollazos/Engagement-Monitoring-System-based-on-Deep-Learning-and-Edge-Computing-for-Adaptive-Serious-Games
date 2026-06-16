import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# CONFIG
# ==========================================

EMOTION_LABELS = [
    "Neutral", "Happy", "Sad", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt"
]

IMG_SIZE = 160  
CONFIDENCE_THRESHOLD = 0.2
DEBUG = False


# ==========================================
# PREPROCESSING
# ==========================================

def preprocess_face(face):

    if face is None or face.size == 0:
        return None

    if len(face.shape) == 2:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    else:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Centered square crop
    h, w, _ = face.shape
    size = min(h, w)

    start_x = (w - size) // 2
    start_y = (h - size) // 2

    face = face[start_y:start_y+size, start_x:start_x+size]

    # Resize
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype(np.float32)

    # MobileNet normalization
    face = preprocess_input(face)

    # Expand dimensions for the model
    face = np.expand_dims(face, axis=0)

    return face


# ==========================================
# EMOTION + VALENCE/AROUSAL PREDICTION
# ==========================================

def predict_emotion_va(face, model_emotion, model_va):

    processed = preprocess_face(face)

    if processed is None:
        return None, None, None, None

    # ======================================
    # EMOTION
    # ======================================

    emotion_preds = model_emotion.predict(processed, verbose=0)[0]

    idx = int(np.argmax(emotion_preds))
    score = float(emotion_preds[idx])

    # Confidence filter
    if score < CONFIDENCE_THRESHOLD:
        return None, None, None, None

    label = EMOTION_LABELS[idx]

    # ======================================
    # VALENCE - AROUSAL
    # ======================================

    va_preds = model_va.predict(processed, verbose=0)[0]

    valence = float(np.clip(va_preds[0], -1, 1))
    arousal = float(np.clip(va_preds[1], -1, 1))

    # ======================================
    # DEBUG (optional)
    # ======================================

    if DEBUG:
        print("Top emotion:", label, "Score:", score)
        print("Valence:", valence, "Arousal:", arousal)
        print("--------------------------------------------------")

    return label, score, valence, arousal
