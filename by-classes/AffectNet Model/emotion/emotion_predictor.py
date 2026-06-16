import cv2
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


EMOTION_LABELS = [
    "Neutral", "Happy", "Sad", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt"
]

IMG_SIZE = 160
CONFIDENCE_THRESHOLD = 0.2
DEBUG = False


@dataclass
class ProcessedFrame:
    """Input already prepared for an emotion-recognition model."""

    data: Any


@dataclass
class EmotionResult:
    """EmotionRecognition output described by the architecture diagram."""

    emotion: str
    emotion_index: int
    confidence: float
    is_reliable: bool
    probabilities: list
    valence: Optional[float] = None
    arousal: Optional[float] = None


class AffectNetPreprocessor:
    """AffectNet preprocessor used by AffectNetEmotionRecognition."""

    def preprocess(self, face):
        if face is None or face.size == 0:
            return None

        if len(face.shape) == 2:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        else:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        h, w, _ = face.shape
        size = min(h, w)

        start_x = (w - size) // 2
        start_y = (h - size) // 2

        face = face[start_y:start_y + size, start_x:start_x + size]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype(np.float32)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        return ProcessedFrame(face)

    def __call__(self, face):
        return self.preprocess(face)


class EmotionRecognition:
    """Base class from the UML diagram."""

    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def infer(self, input: ProcessedFrame) -> EmotionResult:
        raise NotImplementedError


class AffectNetEmotionRecognition(EmotionRecognition):
    """AffectNet specialization of EmotionRecognition."""

    def __init__(
        self,
        model_emotion,
        model_va,
        preprocessor=None,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    ):
        self.model_emotion = model_emotion
        self.model_va = model_va
        self.confidence_threshold = confidence_threshold

        super().__init__(
            {"emotion": model_emotion, "valence_arousal": model_va},
            preprocessor or AffectNetPreprocessor(),
        )

    def infer(self, input: ProcessedFrame) -> EmotionResult:
        processed = input if isinstance(input, ProcessedFrame) else self.preprocessor(input)

        if processed is None:
            return EmotionResult("", -1, 0.0, False, [])

        emotion_preds = self.model_emotion.predict(processed.data, verbose=0)[0]
        idx = int(np.argmax(emotion_preds))
        score = float(emotion_preds[idx])

        if score < self.confidence_threshold:
            return EmotionResult("", idx, score, False, emotion_preds.tolist())

        va_preds = self.model_va.predict(processed.data, verbose=0)[0]
        valence = float(np.clip(va_preds[0], -1, 1))
        arousal = float(np.clip(va_preds[1], -1, 1))

        if DEBUG:
            print("Top emotion:", EMOTION_LABELS[idx], "Score:", score)
            print("Valence:", valence, "Arousal:", arousal)
            print("--------------------------------------------------")

        return EmotionResult(
            emotion=EMOTION_LABELS[idx],
            emotion_index=idx,
            confidence=score,
            is_reliable=True,
            probabilities=emotion_preds.tolist(),
            valence=valence,
            arousal=arousal,
        )


def preprocess_face(face):
    processed = AffectNetPreprocessor().preprocess(face)
    return None if processed is None else processed.data


def predict_emotion_va(face, model_emotion, model_va):
    result = AffectNetEmotionRecognition(model_emotion, model_va).infer(face)

    if not result.is_reliable:
        return None, None, None, None

    return result.emotion, result.confidence, result.valence, result.arousal
