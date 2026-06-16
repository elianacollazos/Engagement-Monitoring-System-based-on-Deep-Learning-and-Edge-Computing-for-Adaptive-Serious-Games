from architecture import EngagementMonitoringSystem, Frame, SystemConfig


class EngagementAnalyzer:
    """Compatibility adapter over the UML-based EngagementMonitoringSystem."""

    def __init__(self, model_path=None, model_type_prod=None, window_size=30):
        self.system = EngagementMonitoringSystem(
            SystemConfig(modelPath=model_path, modelType=model_type_prod)
        )
        self.window_size = window_size

    def detect_emotion(self, frame):
        face = self.system.visualAnalysis.detectFace(Frame(data=frame, frameId=0))
        if face is None:
            return frame
        processed = self.system.emotionRecognition.preprocessor.process(face)
        self.system.emotionRecognition.infer(processed)
        return frame
