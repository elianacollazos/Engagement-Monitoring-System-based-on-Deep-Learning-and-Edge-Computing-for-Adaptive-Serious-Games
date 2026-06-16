from datetime import datetime

from .cognitive_engagement_estimator import CognitiveEngagementEstimator
from .emotion_recognition import FER2013EmotionRecognition
from .engagement_mapper import EngagementMapper
from .session_manager import SessionManager
from .temporal_record import TemporalRecord
from .types import SystemConfig
from .ui_visualizer import UIVisualizer
from .video_acquisition import CameraDevice, VideoAcquisition
from .visual_analysis import VisualAnalysis


class EngagementMonitoringSystem:
    def __init__(self, config: SystemConfig | None = None):
        self.config = config or SystemConfig()
        self.sessionManager = SessionManager(self.config)
        self.videoAcquisition = VideoAcquisition(
            device=CameraDevice(self.config.cameraSource),
            frameRate=self.config.frameRate,
        )
        self.visualAnalysis = VisualAnalysis()
        self.emotionRecognition = FER2013EmotionRecognition()
        self.cognitiveEstimator = CognitiveEngagementEstimator()
        self.engagementMapper = EngagementMapper()
        self.uiVisualizer = UIVisualizer()
        self.records = []

    def startSession(self, userId: str):
        self.videoAcquisition.start()
        return self.sessionManager.startSession(userId)

    def endSession(self):
        self.videoAcquisition.stop()
        self.sessionManager.endSession()

    def processFrame(self) -> TemporalRecord | None:
        frame = self.videoAcquisition.getFrame()
        face = self.visualAnalysis.detectFace(frame)
        if face is None:
            return None
        landmarks = self.visualAnalysis.extractLandmarks(face)
        processed = self.emotionRecognition.preprocessor.process(face)
        emotion = self.emotionRecognition.infer(processed)
        cognitive = self.cognitiveEstimator.infer(frame, face, landmarks)
        engagement = self.engagementMapper.map(emotion, cognitive)
        record = TemporalRecord(
            timestamp=datetime.now(),
            frameId=frame.frameId,
            face=face,
            emotion=emotion,
            cognitive=cognitive,
            engagement=engagement,
        )
        self.records.append(record)
        self.uiVisualizer.update(record)
        return record
