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

