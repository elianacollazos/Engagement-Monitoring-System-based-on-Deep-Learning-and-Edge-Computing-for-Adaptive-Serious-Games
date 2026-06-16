import time
from collections import Counter

import cv2
import torch
import torchvision.transforms as transforms

from config import ANALYSIS_INTERVAL, CONFIDENCE_THRESHOLD, MODEL_PATH, MODEL_TYPE_PROD
from .emotion_model import (
    EmotionCNN,
    EmotionCNNModern,
    EmotionCNNRegularized,
    TransferLearningModel,
)


class EngagementAnalyzer:
    LEVEL_ORDER = ["Disengaged", "Engaged", "Highly Engaged", "Fully Engaged"]
    EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def __init__(self, model_path=MODEL_PATH, model_type_prod=MODEL_TYPE_PROD, window_size=30):
        self.start_time = time.time()
        self.last_analysis_time = 0
        self.analysis_interval = ANALYSIS_INTERVAL

        self.selected_face = None
        self.tracked_face = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.metadata = self.load_model(model_path, model_type_prod)
        self.model.to(self.device)
        self.model.eval()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.transform = self.get_preprocessing_pipeline(model_type_prod)

        self.window_size = window_size
        self.session_emotions = []

        self.emotional_level_groups = {
            "Fully Engaged": ["Surprise"],
            "Highly Engaged": ["Happy", "Angry", "Fear"],
            "Engaged": ["Neutral"],
            "Disengaged": ["Sad", "Disgust"],
        }
        self.emotion_to_level = {
            emotion: level
            for level, emotions in self.emotional_level_groups.items()
            for emotion in emotions
        }
        self.level_impact_values = {
            level: 1.0 / len(emotions)
            for level, emotions in self.emotional_level_groups.items()
        }

    def get_model_class(self, model_type_prod):
        return {
            "modern": EmotionCNNModern,
            "regularized": EmotionCNNRegularized,
            "transfer": TransferLearningModel,
            "original": EmotionCNN,
        }.get(model_type_prod, EmotionCNN)

    def get_preprocessing_pipeline(self, model_type_prod):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),
            ]
        )

    def load_model(self, model_path, model_type_prod):
        model = self.get_model_class(model_type_prod)()
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        return model, {}

    def preprocess_face(self, face_img):
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_tensor = self.transform(face_img)
        return face_tensor.unsqueeze(0)

    def get_emotional_level(self, emotion):
        return self.emotion_to_level.get(emotion, "Disengaged")

    def calculate_emotional_engagement(self, observed_emotions):
        scores = {level: 0.0 for level in self.LEVEL_ORDER}
        counts = {level: 0 for level in self.LEVEL_ORDER}
        total_observed = len(observed_emotions)

        if total_observed == 0:
            return "Disengaged", scores, counts

        level_counts = Counter(self.get_emotional_level(emotion) for emotion in observed_emotions)

        for level in self.LEVEL_ORDER:
            count = level_counts.get(level, 0)
            counts[level] = count
            if count == 0:
                continue

            impact_sum = count * self.level_impact_values[level]
            scores[level] = impact_sum * (count / total_observed)

        final_level = max(
            self.LEVEL_ORDER,
            key=lambda level: (scores[level], self.LEVEL_ORDER.index(level)),
        )
        return final_level, scores, counts

    def detect_emotion(self, frame):
        current_time = time.time()
        if current_time - self.last_analysis_time < self.analysis_interval:
            return frame

        self.last_analysis_time = current_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 3)

        window_emotions = self.session_emotions[-self.window_size :]
        emotional_level_window, _, _ = self.calculate_emotional_engagement(window_emotions)

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
            face_roi = gray[y:y + h, x:x + w]
            face_tensor = self.preprocess_face(face_roi).to(self.device)

            with torch.no_grad():
                outputs = self.model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, 1).item()
                confidence = probabilities[0][predicted].item()

            if confidence >= CONFIDENCE_THRESHOLD:
                emotion = self.EMOTION_LABELS[predicted]
                self.session_emotions.append(emotion)
                window_emotions = self.session_emotions[-self.window_size :]
                emotional_level_window, _, _ = self.calculate_emotional_engagement(window_emotions)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{emotion} ({confidence:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Emotional win: {emotional_level_window}",
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Emotional: {emotional_level_window}",
                    (x, y + h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(
                    frame,
                    f"Low confidence ({confidence:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                )

        cv2.putText(
            frame,
            f"Emotional: {emotional_level_window}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return frame
