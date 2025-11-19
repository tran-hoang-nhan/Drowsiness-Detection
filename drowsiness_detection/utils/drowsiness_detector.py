import cv2
import pickle
import os
from .face_detectors import FaceDetector
from .ml_predictor import MLPredictor
from .feature_extractor import extract_eye_features
from .ear_calculator import calculate_ear_from_dlib_landmarks, calculate_ear_from_keypoints

class DrowsinessDetector:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.ml_predictor = MLPredictor()

        # Detection parameters
        self.EAR_THRESHOLD = 0.2  # dlib EAR threshold (0.2 = mắt đóng)
        self.CONSECUTIVE_FRAMES = 10
        self.frame_counter = 0
        self.alarm_playing = False
        
        # Sound system
        try:
            import winsound
            self.winsound = winsound
        except ImportError:
            self.winsound = None
    
    def extract_eye_region(self, gray, x, y, size=30):
        """Extract eye region from coordinates"""
        if gray is None:
            return None
        try:
            padding = 10
            x1 = max(0, int(x) - padding)
            y1 = max(0, int(y) - padding)
            x2 = min(gray.shape[1], int(x) + size + padding)
            y2 = min(gray.shape[0], int(y) + size + padding)

            region = gray[y1:y2, x1:x2]
            return region if region.size > 0 else None
        except:
            return None
    
    def detect(self, frame):
        """Main detection pipeline with dlib EAR-based eye state recognition"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face and get dlib landmarks
        detection = self.face_detector.detect(frame)
        method = "Unknown"
        ear_value = 0.5  # Default: unknown state
        ml_confidence = 0.5
        eyes_detected = False

        if detection:
            # ===== PRIMARY: dlib Landmarks (Most Accurate) =====
            if 'dlib_landmarks' in detection:
                landmarks = detection['dlib_landmarks']

                # Calculate precise EAR from 68-point landmarks
                ear_value = calculate_ear_from_dlib_landmarks(landmarks)
                eyes_detected = True
                method = "dlib EAR"

                # Get eye region coordinates for ML prediction
                # Left eye center: average of landmarks 36-41
                left_eye_points = [landmarks[i] for i in range(36, 42)]
                left_eye_center = (
                    int(sum(p[0] for p in left_eye_points) / 6),
                    int(sum(p[1] for p in left_eye_points) / 6)
                )

                # Right eye center: average of landmarks 42-47
                right_eye_points = [landmarks[i] for i in range(42, 48)]
                right_eye_center = (
                    int(sum(p[0] for p in right_eye_points) / 6),
                    int(sum(p[1] for p in right_eye_points) / 6)
                )

                # Get ML predictions for confidence
                left_region = self.extract_eye_region(gray, left_eye_center[0], left_eye_center[1])
                right_region = self.extract_eye_region(gray, right_eye_center[0], right_eye_center[1])

                left_pred = self.ml_predictor.predict_eye_state(left_region) or 0.5
                right_pred = self.ml_predictor.predict_eye_state(right_region) or 0.5
                ml_confidence = (left_pred + right_pred) / 2.0

                # Draw face box
                if 'box' in detection:
                    x, y, w, h = detection['box']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw 68 landmarks on face
                for i, (px, py) in landmarks.items():
                    # Eye landmarks in red
                    if 36 <= i <= 47:
                        cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)
                    # Other landmarks in yellow
                    else:
                        cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

            # ===== FALLBACK: MTCNN Keypoints =====
            elif 'keypoints' in detection:
                keypoints = detection['keypoints']
                if 'left_eye' in keypoints and 'right_eye' in keypoints and 'nose' in keypoints:
                    left_eye = keypoints['left_eye']
                    right_eye = keypoints['right_eye']
                    nose = keypoints['nose']
                    
                    ear_value = calculate_ear_from_keypoints(left_eye, right_eye, nose)
                    eyes_detected = True
                    method = "MTCNN EAR (fallback)"

                    # Get ML predictions
                    left_region = self.extract_eye_region(gray, left_eye[0], left_eye[1])
                    right_region = self.extract_eye_region(gray, right_eye[0], right_eye[1])
                    
                    left_pred = self.ml_predictor.predict_eye_state(left_region) or 0.5
                    right_pred = self.ml_predictor.predict_eye_state(right_region) or 0.5
                    ml_confidence = (left_pred + right_pred) / 2.0

                    # Draw face box and keypoints
                    if 'box' in detection:
                        x, y, w, h = detection['box']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    for key in ['left_eye', 'right_eye', 'nose']:
                        if key in keypoints:
                            pt = keypoints[key]
                            cv2.circle(frame, tuple(map(int, pt)), 4, (0, 255, 255), -1)

        # ===== DROWSINESS DETECTION LOGIC =====
        drowsy_status = False
        
        if eyes_detected:
            if ml_confidence > 0.8:
                # Mắt đóng/nửa đóng (theo ML model)
                self.frame_counter += 1
            else:
                # Mắt mở (theo ML model)
                self.frame_counter = 0
        
        # Trigger alert if consecutive closed eye frames exceed threshold
        if self.frame_counter >= self.CONSECUTIVE_FRAMES:
            drowsy_status = True
            if not self.alarm_playing:
                print("🚨 DROWSINESS ALERT!")
                if self.winsound:
                    try:
                        self.winsound.Beep(1000, 500)
                    except:
                        pass
                self.alarm_playing = True
        else:
            self.alarm_playing = False
        
        # ===== DRAW INFORMATION ON FRAME =====
        # Status color: Green=Open, Red=Closed/Drowsy
        status_color = (0, 255, 0) if ear_value >= self.EAR_THRESHOLD else (0, 0, 255)
        
        cv2.putText(frame, f"Method: {method} | EAR: {ear_value:.3f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Frames Closed: {self.frame_counter}/{self.CONSECUTIVE_FRAMES}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"ML Confidence: {ml_confidence:.3f}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if drowsy_status:
            cv2.putText(frame, "DROWSINESS DETECTED!", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        return frame, drowsy_status, ear_value