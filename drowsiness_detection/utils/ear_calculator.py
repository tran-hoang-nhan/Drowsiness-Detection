"""
EAR (Eye Aspect Ratio) Calculator Module
"""
import cv2
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

class EARCalculator:
    def __init__(self):
        pass
    
    def calculate_mtcnn_ear(self, left_eye, right_eye, nose):
        """
        Calculate Eye Aspect Ratio (EAR) using MTCNN keypoints
        Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        MTCNN keypoints: {'left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right'}
        For approximation without full landmarks, we use eye-to-nose distance
        """
        try:
            eye_distance = dist.euclidean(left_eye, right_eye)
            if eye_distance == 0:
                return 0.0
            
            # Distance from each eye to nose (approximation of vertical eye height)
            left_to_nose = dist.euclidean(left_eye, nose)
            right_to_nose = dist.euclidean(right_eye, nose)
            avg_vertical_dist = (left_to_nose + right_to_nose) / 2
            
            # EAR approximation
            ear = avg_vertical_dist / eye_distance
            
            # Normalize to 0-1 range for consistency
            # Typical range: 0.05 (closed) to 0.35 (open)
            ear_normalized = np.clip(ear / 0.4, 0, 1)
            
            return ear_normalized
        except:
            return 0.0
    
    def calculate_haar_ear(self, left_eye, right_eye):
        """Calculate EAR from Haar eye regions"""
        eye_width_avg = (left_eye[2] + right_eye[2]) / 2
        eye_height_avg = (left_eye[3] + right_eye[3]) / 2
        
        if eye_width_avg > 0:
            return eye_height_avg / eye_width_avg
        else:
            return 0.1
    
    def extract_eye_region(self, gray_frame, eye_point, size=25):
        """Extract eye region from frame"""
        try:
            x, y = eye_point
            x_min = max(0, x - size)
            x_max = min(gray_frame.shape[1], x + size)
            y_min = max(0, y - size)
            y_max = min(gray_frame.shape[0], y + size)
            
            if x_max > x_min and y_max > y_min:
                return gray_frame[y_min:y_max, x_min:x_max]
            return None
        except:
            return None
    
    def analyze_eye_pixels(self, eye_region):
        """Analyze eye pixels for closed/open detection"""
        if eye_region is None or eye_region.size == 0:
            return 0.5
        
        eye_region = cv2.resize(eye_region, (32, 32))
        
        mean_intensity = np.mean(eye_region)
        variance = np.var(eye_region)
        dark_pixels = np.sum(eye_region < 50) / eye_region.size
        
        closed_score = 0
        
        if mean_intensity < 80:
            closed_score += 0.4
        
        if variance < 200:
            closed_score += 0.3
            
        if dark_pixels > 0.6:
            closed_score += 0.3
        
        return closed_score