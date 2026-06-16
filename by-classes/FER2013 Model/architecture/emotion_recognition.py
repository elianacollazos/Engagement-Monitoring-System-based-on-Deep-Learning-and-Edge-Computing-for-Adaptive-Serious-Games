import torch

from config import CONFIDENCE_THRESHOLD, MODEL_PATH, MODEL_TYPE
from models.emotion_model import load_model, predict_emotion, preprocess_face

from .types import EmotionResult, ProcessedFrame


class DLModel:
    def __init__(self, model_path=MODEL_PATH, model_type=MODEL_TYPE):
        self.model_path = model_path
        self.model_type = model_type
        self.model, self.metadata = load_model(model_path, model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input: ProcessedFrame, confidence_threshold=CONFIDENCE_THRESHOLD):
        tensor = input.data.to(self.device)
        return predict_emotion(self.model, tensor, confidence_threshold)


class FERPreprocessor:
    def process(self, face) -> ProcessedFrame:
        return ProcessedFrame(data=preprocess_face(face.data))


class EmotionRecognition:
    def __init__(self, model: DLModel | None = None, preprocessor=None):
        self.model = model or DLModel()
        self.preprocessor = preprocessor

    def infer(self, input: ProcessedFrame) -> EmotionResult:
        prediction = self.model.predict(input)
        return EmotionResult(
            label=str(prediction["emotion"]),
            confidence=float(prediction["confidence"]),
            scores={"probabilities": prediction["probabilities"]},
            raw=prediction,
        )


class FER2013EmotionRecognition(EmotionRecognition):
    EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def __init__(self, model: DLModel | None = None, preprocessor: FERPreprocessor | None = None):
        super().__init__(model or DLModel(), preprocessor or FERPreprocessor())

    def infer(self, input: ProcessedFrame) -> EmotionResult:
        prediction = self.model.predict(input)
        emotion_index = int(prediction["emotion"])
        label = self.EMOTION_LABELS[emotion_index]
        return EmotionResult(
            label=label,
            confidence=float(prediction["confidence"]),
            scores={"probabilities": prediction["probabilities"]},
            raw=prediction,
        )
