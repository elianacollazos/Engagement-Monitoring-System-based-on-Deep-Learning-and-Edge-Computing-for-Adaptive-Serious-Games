import mediapipe as mp


def create_face_mesh():

    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # Required for iris landmarks
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    return face_mesh
