import numpy as np
from scipy.spatial import distance as dist

# ===== Module-level EAR utilities (preferred entry points) =====
def calculate_ear_from_dlib_landmarks(landmarks):
    """
    Calculate precise Eye Aspect Ratio (EAR) from dlib 68-point facial landmarks.

    dlib indices:
      - Left eye:  36-41
      - Right eye: 42-47

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Returns average EAR of both eyes. Typical closed-eye EAR ≈ 0.2
    """
    try:
        left_eye = [landmarks[i] for i in range(36, 42)]
        right_eye = [landmarks[i] for i in range(42, 48)]

        # Left EAR
        lv1 = dist.euclidean(left_eye[1], left_eye[5])
        lv2 = dist.euclidean(left_eye[2], left_eye[4])
        lh = dist.euclidean(left_eye[0], left_eye[3])
        left_ear = (lv1 + lv2) / (2.0 * lh) if lh != 0 else 0.0

        # Right EAR
        rv1 = dist.euclidean(right_eye[1], right_eye[5])
        rv2 = dist.euclidean(right_eye[2], right_eye[4])
        rh = dist.euclidean(right_eye[0], right_eye[3])
        right_ear = (rv1 + rv2) / (2.0 * rh) if rh != 0 else 0.0

        return (left_ear + right_ear) / 2.0
    except Exception:
        return 0.0


def calculate_ear_from_keypoints(left_eye, right_eye, nose):
    """
    Approximate EAR from MTCNN keypoints (left_eye, right_eye, nose) when full
    landmarks are not available. Returns a normalized proxy in [0, 1].
    """
    try:
        eye_distance = dist.euclidean(left_eye, right_eye)
        if eye_distance == 0:
            return 0.0

        left_to_nose = dist.euclidean(left_eye, nose)
        right_to_nose = dist.euclidean(right_eye, nose)
        avg_vertical = (left_to_nose + right_to_nose) / 2.0

        ear_proxy = avg_vertical / eye_distance
        return float(np.clip(ear_proxy / 0.4, 0, 1))
    except Exception:
        return 0.0