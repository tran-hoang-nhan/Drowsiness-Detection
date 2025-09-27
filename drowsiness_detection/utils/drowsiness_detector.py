import cv2
import numpy as np
from scipy.spatial import distance as dist
import pickle
import os
from .feature_extractor import extract_eye_features

try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
    print("✅ MTCNN imported successfully")
except Exception as e:
    MTCNN_AVAILABLE = False
    print(f"⚠️ MTCNN not available: {e}")

class DrowsinessDetector:
    def __init__(self):
        # === MACHINE LEARNING COMPONENTS ===
        self.ml_model = None
        self.load_ml_model()
        
        # === COMPUTER VISION COMPONENTS ===
        self.setup_cv_detectors()
        
        # === DETECTION PARAMETERS ===
        self.EAR_THRESHOLD = 0.25
        self.CONSECUTIVE_FRAMES = 45
        self.YAWN_THRESHOLD = 0.6
        
        # === STATE TRACKING ===
        self.frame_counter = 0
        self.ear_history = []
        self.alarm_playing = False
        
        # === SOUND SYSTEM ===
        self.setup_sound_system()
        
    def load_ml_model(self):
        """Load trained ML model"""
        try:
            with open('models/eye_classifier.pkl', 'rb') as f:
                model_data = pickle.load(f)
                if isinstance(model_data, dict):
                    self.ml_model = model_data.get('pipeline', model_data.get('model'))
                    print(f"✅ Loaded ML model: {model_data.get('pipeline_name', 'Unknown')}")
                    print(f"   Accuracy: {model_data.get('accuracy', 0):.3f}")
                else:
                    self.ml_model = model_data
                    print("✅ Loaded ML model")
        except Exception as e:
            print(f"⚠️ ML model not found: {e}")
            
    def setup_cv_detectors(self):
        """Setup MTCNN detector only"""
        if MTCNN_AVAILABLE:
            try:
                self.mtcnn_detector = MTCNN()
                self.use_mtcnn = True
                print("✅ MTCNN Face Detection (robust with angles & glasses)")
            except Exception as e:
                print(f"⚠️ MTCNN initialization failed: {e}")
                self.use_mtcnn = False
        else:
            self.use_mtcnn = False
            print("❌ No face detection available")
        
    def setup_sound_system(self):
        """Setup sound alert system"""
        try:
            import winsound
            self.winsound = winsound
        except ImportError:
            self.winsound = None
    

    
    def predict_eye_state(self, eye_region):
        """Predict eye state using ML model"""
        if self.ml_model is None:
            return None
            
        try:
            features = extract_eye_features(eye_region)
            features = features.reshape(1, -1)
            
            if hasattr(self.ml_model, 'predict_proba'):
                prob = self.ml_model.predict_proba(features)[0]
                return prob[0]  # Probability of closed eye
            else:
                return self.ml_model.predict(features)[0]
        except Exception as e:
            return None
    
    def play_alarm_sound(self):
        """Play alarm sound"""
        print("🚨 DROWSINESS ALERT!")
        
        try:
            if self.winsound:
                self.winsound.Beep(1000, 500)
        except:
            pass
    
    def _mtcnn_ear(self, left_eye, right_eye, nose):
        """Calculate EAR approximation from MTCNN keypoints"""
        # Distance between eyes
        eye_distance = dist.euclidean(left_eye, right_eye)
        
        # Distance from nose to eye center
        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        nose_eye_dist = dist.euclidean(nose, eye_center)
        
        # EAR approximation
        ear = nose_eye_dist / eye_distance if eye_distance > 0 else 0.3
        return min(ear * 0.5, 0.4)  # Normalize and cap
    
    def _extract_mtcnn_eye_region(self, gray_frame, eye_point):
        """Extract eye region from MTCNN keypoint"""
        try:
            x, y = eye_point
            size = 25
            x_min = max(0, x - size)
            x_max = min(gray_frame.shape[1], x + size)
            y_min = max(0, y - size)
            y_max = min(gray_frame.shape[0], y + size)
            
            if x_max > x_min and y_max > y_min:
                return gray_frame[y_min:y_max, x_min:x_max]
            return None
        except:
            return None
    
    def detect(self, frame):
        """Main detection pipeline - MTCNN + ML"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        drowsy_status = False
        ear_value = 0.0
        ml_confidence = 0.0
        
        if self.use_mtcnn:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.mtcnn_detector.detect_faces(rgb_frame)
            
            if result:
                for face in result:
                    x, y, w, h = face['box']
                    confidence = face['confidence']
                    
                    if confidence > 0.9:
                        keypoints = face['keypoints']
                        left_eye = keypoints['left_eye']
                        right_eye = keypoints['right_eye']
                        nose = keypoints['nose']
                        
                        # Calculate EAR
                        ear_value = self._mtcnn_ear(left_eye, right_eye, nose)
                        
                        # ML prediction
                        if self.ml_model:
                            left_eye_region = self._extract_mtcnn_eye_region(gray, left_eye)
                            right_eye_region = self._extract_mtcnn_eye_region(gray, right_eye)
                            
                            left_pred = self.predict_eye_state(left_eye_region)
                            right_pred = self.predict_eye_state(right_eye_region)
                            
                            if left_pred is not None and right_pred is not None:
                                ml_confidence = (left_pred + right_pred) / 2.0
                        
                        # Draw visualization
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.circle(frame, left_eye, 3, (0, 255, 0), -1)
                        cv2.circle(frame, right_eye, 3, (0, 255, 0), -1)
                        cv2.circle(frame, nose, 3, (0, 255, 255), -1)
        
        # Update history
        self.ear_history.append(ear_value)
        if len(self.ear_history) > 30:
            self.ear_history.pop(0)
        
        # Drowsiness detection
        drowsy_indicators = 0
        
        if ear_value < self.EAR_THRESHOLD:
            self.frame_counter += 1
            drowsy_indicators += 1
        else:
            self.frame_counter = 0
            
        if ml_confidence > 0.7:
            drowsy_indicators += 1
        
        # Decision logic
        if self.frame_counter >= self.CONSECUTIVE_FRAMES or drowsy_indicators >= 2:
            drowsy_status = True
            if not self.alarm_playing:
                self.play_alarm_sound()
                self.alarm_playing = True
        else:
            self.alarm_playing = False
        
        # Draw metrics
        self._draw_metrics(frame, ear_value, ml_confidence, drowsy_indicators)
        
        return frame, drowsy_status, ear_value
    
    def _draw_metrics(self, frame, ear, ml_conf, indicators):
        """Draw metrics on frame"""
        y_offset = 30
        
        cv2.putText(frame, "MTCNN + ML Detection", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(frame, f"EAR: {ear:.3f} (T: {self.EAR_THRESHOLD})", 
                   (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if ml_conf > 0:
            cv2.putText(frame, f"ML Confidence: {ml_conf:.3f}", 
                       (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Closed Frames: {self.frame_counter}/{self.CONSECUTIVE_FRAMES}", 
                   (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(frame, f"Risk Indicators: {indicators}/2", 
                   (10, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 0, 255) if indicators >= 2 else (0, 255, 0), 2)
        
        if self.alarm_playing:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, y_offset + 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)