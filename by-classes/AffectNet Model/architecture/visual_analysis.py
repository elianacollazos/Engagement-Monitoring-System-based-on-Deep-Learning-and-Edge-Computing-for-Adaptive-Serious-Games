import cv2
import mediapipe as mp

from .types import Face, Frame, Landmarks, ProcessedFrame


class FaceDetector:
    def detect(self, frame: Frame) -> Face | None:
        if frame.data is None:
            return None
        h, w, _ = frame.data.shape
        rgb = cv2.cvtColor(frame.data, cv2.COLOR_BGR2RGB)
        mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        results = mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        x_coords = [int(p.x * w) for p in landmarks]
        y_coords = [int(p.y * h) for p in landmarks]
        x, y = min(x_coords), min(y_coords)
        bw, bh = max(x_coords) - x, max(y_coords) - y
        face = frame.data[y:y + bh, x:x + bw]
        return Face(data=face, bbox=(x, y, bw, bh))


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
