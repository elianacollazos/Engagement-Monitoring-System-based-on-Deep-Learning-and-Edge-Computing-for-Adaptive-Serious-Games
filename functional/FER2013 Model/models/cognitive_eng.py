from collections import deque

import cv2
import mediapipe as mp
import numpy as np

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

LEFT_IRIS_IDX = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]

NOSE_TIP_IDX = 1
FACE_LEFT_IDX = 234
FACE_RIGHT_IDX = 454
FOREHEAD_IDX = 10
CHIN_IDX = 152

DEFAULT_LEVEL_ORDER = ["Disengaged", "Engaged", "Highly Engaged", "Fully Engaged"]


def create_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def get_primary_face_landmarks(frame, face_mesh):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0].landmark


def get_eye_points(landmarks, indices, w, h):
    return np.array([
        [landmarks[i].x * w, landmarks[i].y * h]
        for i in indices
    ], dtype=np.float32)


def get_iris_center(landmarks, indices, w, h):
    points = np.array([
        [landmarks[i].x * w, landmarks[i].y * h]
        for i in indices
    ], dtype=np.float32)

    return np.mean(points, axis=0)


def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)


def compute_eye_gaze_ratio(eye, iris_center):
    eye_left = eye[0]
    eye_right = eye[3]

    width = eye_right[0] - eye_left[0]

    if abs(width) < 1e-6:
        return 0.5

    ratio = (iris_center[0] - eye_left[0]) / width
    return float(np.clip(ratio, 0.0, 1.0))


def compute_gaze_attention(left_eye, right_eye, left_iris, right_iris):
    l = compute_eye_gaze_ratio(left_eye, left_iris)
    r = compute_eye_gaze_ratio(right_eye, right_iris)

    gaze = (l + r) / 2.0

    center_tolerance = 0.12
    deviation = abs(gaze - 0.5)

    if deviation <= center_tolerance:
        return 1.0

    penalty = (deviation - center_tolerance) / (0.5 - center_tolerance)
    return float(np.clip(1.0 - penalty, 0.0, 1.0))


def compute_head_pose_score(landmarks, w, h):
    nose = np.array(
        [landmarks[NOSE_TIP_IDX].x * w, landmarks[NOSE_TIP_IDX].y * h],
        dtype=np.float32,
    )
    face_left = np.array(
        [landmarks[FACE_LEFT_IDX].x * w, landmarks[FACE_LEFT_IDX].y * h],
        dtype=np.float32,
    )
    face_right = np.array(
        [landmarks[FACE_RIGHT_IDX].x * w, landmarks[FACE_RIGHT_IDX].y * h],
        dtype=np.float32,
    )
    forehead = np.array(
        [landmarks[FOREHEAD_IDX].x * w, landmarks[FOREHEAD_IDX].y * h],
        dtype=np.float32,
    )
    chin = np.array(
        [landmarks[CHIN_IDX].x * w, landmarks[CHIN_IDX].y * h],
        dtype=np.float32,
    )

    face_width = np.linalg.norm(face_right - face_left)
    face_height = np.linalg.norm(chin - forehead)

    if face_width < 1e-6 or face_height < 1e-6:
        return 0.5

    face_center = (face_left + face_right) / 2.0

    yaw = abs(nose[0] - face_center[0]) / face_width
    pitch = abs(nose[1] - face_center[1]) / face_height

    yaw_score = 1.0 - np.clip(yaw / 0.18, 0.0, 1.0)
    pitch_score = 1.0 - np.clip(pitch / 0.22, 0.0, 1.0)

    head_pose_score = 0.7 * yaw_score + 0.3 * pitch_score

    return float(np.clip(head_pose_score, 0.0, 1.0))


def compute_cognitive_engagement(
    left_eye,
    right_eye,
    prev_eye_center,
    left_iris=None,
    right_iris=None,
    landmarks=None,
    w=None,
    h=None,
):
    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    blink_score = np.clip((ear - 0.15) / 0.15, 0.0, 1.0)

    eye_center = (
        left_eye[0] + left_eye[3] +
        right_eye[0] + right_eye[3]
    ) / 4.0

    if prev_eye_center is None:
        movement = 0.0
    else:
        movement = np.linalg.norm(eye_center - prev_eye_center)

    stability = np.clip(1.0 - movement / 40.0, 0.0, 1.0)

    if landmarks is not None and w is not None and h is not None:
        head_pose_score = compute_head_pose_score(landmarks, w, h)
    else:
        head_pose_score = 1.0

    base_attention = (
        0.35 * stability +
        0.35 * blink_score +
        0.30 * head_pose_score
    )

    if left_iris is not None and right_iris is not None:
        gaze_score = compute_gaze_attention(left_eye, right_eye, left_iris, right_iris)
    else:
        gaze_score = 1.0

    cognitive = (
        0.75 * base_attention +
        0.25 * gaze_score
    )

    if gaze_score < 0.25:
        cognitive *= 0.65

    if head_pose_score < 0.35:
        cognitive *= 0.65

    if blink_score < 0.20:
        cognitive *= 0.45

    cognitive = float(np.clip(cognitive, 0.0, 1.0))


    return cognitive, eye_center, blink_score, gaze_score, head_pose_score, base_attention


def compute_attention(frame, face_mesh, prev_eye_center=None):
    landmarks = get_primary_face_landmarks(frame, face_mesh)
    if landmarks is None:
        return 0.0, {
            "presence_score": 0.0,
            "blink_score": 0.0,
            "base_attention": 0.0,
            "gaze_score": 0.0,
            "head_score": 0.0,
            "yaw_score": 0.0,
            "pitch_score": 0.0,
            "eye_center": None,
        }

    h, w, _ = frame.shape
    left_eye = get_eye_points(landmarks, LEFT_EYE_IDX, w, h)
    right_eye = get_eye_points(landmarks, RIGHT_EYE_IDX, w, h)
    left_iris = get_iris_center(landmarks, LEFT_IRIS_IDX, w, h)
    right_iris = get_iris_center(landmarks, RIGHT_IRIS_IDX, w, h)

    (
        cognitive,
        eye_center,
        blink_score,
        gaze_score,
        head_pose_score,
        base_attention,
    ) = (
        compute_cognitive_engagement(
            left_eye,
            right_eye,
            prev_eye_center,
            left_iris=left_iris,
            right_iris=right_iris,
            landmarks=landmarks,
            w=w,
            h=h,
        )
    )

    return cognitive, {
        "presence_score": 1.0,
        "blink_score": blink_score,
        "base_attention": base_attention,
        "gaze_score": gaze_score,
        "head_score": head_pose_score,
        "yaw_score": head_pose_score,
        "pitch_score": head_pose_score,
        "eye_center": eye_center,
    }


def get_cognitive_level(attention_pct, level_order=None):
    levels = level_order or DEFAULT_LEVEL_ORDER
    if attention_pct >= 75:
        return levels[3]
    if attention_pct >= 50:
        return levels[2]
    if attention_pct >= 25:
        return levels[1]
    return levels[0]


def get_attention_percentage(attention_history):
    if not attention_history:
        return 0.0
    return float(np.mean(attention_history) * 100.0)


class CognitiveEngagementEstimator:
    def __init__(self, window_size=30, level_order=None):
        self.face_mesh = create_face_mesh()
        self.attention_history = deque(maxlen=window_size)
        self.level_order = level_order or DEFAULT_LEVEL_ORDER
        self.prev_eye_center = None

    def reset(self):
        self.attention_history.clear()
        self.prev_eye_center = None

    def update(self, frame):
        attention_score, attention_details = compute_attention(
            frame,
            self.face_mesh,
            self.prev_eye_center,
        )
        self.prev_eye_center = attention_details["eye_center"]
        self.attention_history.append(attention_score)
        attention_pct = get_attention_percentage(self.attention_history)
        cognitive_level = get_cognitive_level(attention_pct, self.level_order)
        return attention_score, attention_details, cognitive_level, attention_pct
