import cv2

from .types import Face, Frame, Landmarks, ProcessedFrame


class FaceDetector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )

    def detect(self, frame: Frame) -> Face | None:
        if frame.data is None:
            return None
        gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 3)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        return Face(data=gray[y:y + h, x:x + w], bbox=(x, y, w, h))


class LandmarkExtractor:
    def extract(self, face: Face) -> Landmarks | None:
        return Landmarks(points=None) if face is not None else None


class VisualAnalysis:
    def __init__(
        self,
        faceDetector: FaceDetector | None = None,
        landmarkExtractor: LandmarkExtractor | None = None,
    ):
        self.faceDetector = faceDetector or FaceDetector()
        self.landmarkExtractor = landmarkExtractor or LandmarkExtractor()

    def detectFace(self, frame: Frame) -> Face | None:
        return self.faceDetector.detect(frame)

    def extractLandmarks(self, face: Face) -> Landmarks | None:
        return self.landmarkExtractor.extract(face)

    def preprocess(self, face: Face) -> ProcessedFrame:
        return ProcessedFrame(data=face.data)
