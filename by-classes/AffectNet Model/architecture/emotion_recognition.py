from emotion.emotion_predictor import AffectNetPreprocessor as LegacyAffectNetPreprocessor
from emotion.emotion_predictor import CONFIDENCE_THRESHOLD
from models.model_loader import load_models

from .types import EmotionResult, ProcessedFrame


class DLModel:
    def __init__(self):
        self.emotionModel, self.valenceArousalModel = load_models()

    def predict(self, input: ProcessedFrame):
        emotion_preds = self.emotionModel.predict(input.data, verbose=0)[0]
        va_preds = self.valenceArousalModel.predict(input.data, verbose=0)[0]
        return emotion_preds, va_preds


class AffectNetPreprocessor:
    def __init__(self):
        self.legacyPreprocessor = LegacyAffectNetPreprocessor()

    def process(self, face) -> ProcessedFrame:
        processed = self.legacyPreprocessor.preprocess(face.data)
        return ProcessedFrame(data=None if processed is None else processed.data)


class EmotionRecognition:
    def __init__(self, model: DLModel | None = None, preprocessor=None):
        self.model = model or DLModel()
        self.preprocessor = preprocessor

    def infer(self, input: ProcessedFrame) -> EmotionResult:
        return EmotionResult(label="Unknown", confidence=0.0)


class AffectNetEmotionRecognition(EmotionRecognition):
    EMOTION_LABELS = [
        "Neutral", "Happy", "Sad", "Surprise",
        "Fear", "Disgust", "Anger", "Contempt"
    ]

    def __init__(
        self,
        model: DLModel | None = None,
        preprocessor: AffectNetPreprocessor | None = None,
        confidenceThreshold: float = CONFIDENCE_THRESHOLD,
    ):
        super().__init__(model or DLModel(), preprocessor or AffectNetPreprocessor())
        self.confidenceThreshold = confidenceThreshold

    def infer(self, input: ProcessedFrame) -> EmotionResult:
        if input.data is None:
            return EmotionResult(label="", confidence=0.0)
        emotion_preds, va_preds = self.model.predict(input)
        emotion_index = int(emotion_preds.argmax())
        confidence = float(emotion_preds[emotion_index])
        if confidence < self.confidenceThreshold:
            return EmotionResult(label="", confidence=confidence, raw=emotion_preds)
        return EmotionResult(
            label=self.EMOTION_LABELS[emotion_index],
            confidence=confidence,
            scores={"probabilities": emotion_preds.tolist()},
            valence=float(va_preds[0]),
            arousal=float(va_preds[1]),
            raw={"emotion": emotion_preds, "valence_arousal": va_preds},
        )
