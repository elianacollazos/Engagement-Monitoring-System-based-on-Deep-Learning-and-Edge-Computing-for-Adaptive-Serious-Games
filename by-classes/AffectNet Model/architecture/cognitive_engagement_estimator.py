from vision.eye_tracking import compute_cognitive_engagement

from .types import AttentionResult, CognitiveResult, Face, GazeResult, Landmarks, PostureResult


class Model:
    def predict(self, input):
        return 0.0


class PostureAnalyzer:
    def analyze(self, landmarks: Landmarks) -> PostureResult:
        return PostureResult(score=0.0, details={"landmarks": landmarks.points})


class GazeAnalyzer:
    def analyze(self, landmarks: Landmarks) -> GazeResult:
        return GazeResult(score=0.0, details={"landmarks": landmarks.points})


class CognitiveEngagementEstimator:
    def __init__(
        self,
        attentionModel: Model | None = None,
        postureAnalyzer: PostureAnalyzer | None = None,
        gazeAnalyzer: GazeAnalyzer | None = None,
    ):
        self.attentionModel = attentionModel or Model()
        self.postureAnalyzer = postureAnalyzer or PostureAnalyzer()
        self.gazeAnalyzer = gazeAnalyzer or GazeAnalyzer()

    def estimateAttention(self, face: Face, landmarks: Landmarks) -> AttentionResult:
        score = float(self.attentionModel.predict({"face": face, "landmarks": landmarks}))
        return AttentionResult(score=score)

    def estimatePosture(self, landmarks: Landmarks) -> PostureResult:
        return self.postureAnalyzer.analyze(landmarks)

    def estimateGaze(self, landmarks: Landmarks) -> GazeResult:
        return self.gazeAnalyzer.analyze(landmarks)

    def infer(self, frame, face: Face, landmarks: Landmarks) -> CognitiveResult:
        attention = self.estimateAttention(face, landmarks)
        posture = self.estimatePosture(landmarks)
        gaze = self.estimateGaze(landmarks)
        return CognitiveResult(attention=attention, posture=posture, gaze=gaze, score=attention.score)
