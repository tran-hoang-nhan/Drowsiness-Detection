import cv2
import numpy as np
from scipy.spatial import distance as dist
from sklearn.preprocessing import StandardScaler
import pickle
import os

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("⚠️ Dlib không có, dùng Haar Cascade")

class DrowsinessDetector:
    def __init__(self):
        # === MACHINE LEARNING COMPONENTS ===
        self.ml_model = None
        self.scaler = StandardScaler()
        self.load_ml_model()
        
        # === COMPUTER VISION COMPONENTS ===
        self.setup_cv_detectors()
        
        # === DETECTION PARAMETERS ===
        self.EAR_THRESHOLD = 0.25
        self.BLINK_THRESHOLD = 0.2
        self.CONSECUTIVE_FRAMES = 45  # 1.5 giây @ 30fps
        self.YAWN_THRESHOLD = 0.6
        
        # === STATE TRACKING ===
        self.frame_counter = 0
        self.blink_counter = 0
        self.yawn_counter = 0
        self.ear_history = []
        self.mar_history = []
        self.alarm_playing = False
        
        # === FEATURE EXTRACTION ===
        self.feature_buffer = []
        self.buffer_size = 10
        
        # === SOUND SYSTEM ===
        self.setup_sound_system()
        
    def load_ml_model(self):
        """Load trained ML model"""
        try:
            with open('models/eye_classifier.pkl', 'rb') as f:
                model_data = pickle.load(f)
                if isinstance(model_data, dict):
                    self.ml_model = model_data['model']
                    print(f"✅ Loaded ML model: {model_data.get('model_name', 'Unknown')}")
                    print(f"   Accuracy: {model_data.get('accuracy', 0):.3f}")
                else:
                    self.ml_model = model_data
                    print("✅ Loaded legacy ML model")
        except Exception as e:
            print(f"⚠️ ML model không có: {e}")
            
    def setup_cv_detectors(self):
        """Setup Computer Vision detectors"""
        if DLIB_AVAILABLE:
            self.detector = dlib.get_frontal_face_detector()
            try:
                self.predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
                self.use_dlib = True
                print("✅ Dlib 68-point facial landmarks")
            except:
                print("❌ shape_predictor_68_face_landmarks.dat not found")
                self.use_dlib = False
        else:
            self.use_dlib = False
            
        if not self.use_dlib:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            print("✅ Haar Cascade classifiers")
            
        # Landmark indices
        self.LEFT_EYE_POINTS = list(range(36, 42))
        self.RIGHT_EYE_POINTS = list(range(42, 48))
        self.MOUTH_POINTS = list(range(48, 68))
        
    def setup_sound_system(self):
        """Setup sound alert system"""
        try:
            import winsound
            self.winsound = winsound
            self.test_sound()
        except ImportError:
            self.winsound = None
            print("⚠️ Sound system unavailable")
    
    def eye_aspect_ratio(self, eye_landmarks):
        """Tính Eye Aspect Ratio (EAR) - Computer Vision metric"""
        # Vertical distances
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal distance
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # EAR formula
        ear = (A + B) / (2.0 * C) if C > 0 else 0
        return ear
    
    def mouth_aspect_ratio(self, mouth_landmarks):
        """Tính Mouth Aspect Ratio (MAR) - để phát hiện ngáp"""
        # Vertical distances
        A = dist.euclidean(mouth_landmarks[2], mouth_landmarks[10])  # 50, 58
        B = dist.euclidean(mouth_landmarks[4], mouth_landmarks[8])   # 52, 56
        
        # Horizontal distance
        C = dist.euclidean(mouth_landmarks[0], mouth_landmarks[6])   # 48, 54
        
        mar = (A + B) / (2.0 * C) if C > 0 else 0
        return mar
    
    def extract_eye_features(self, eye_region):
        """Trích xuất đặc trưng từ vùng mắt cho ML"""
        if eye_region is None or eye_region.size == 0:
            return np.zeros(12)
            
        # Resize to standard size
        eye_resized = cv2.resize(eye_region, (32, 32))
        
        features = []
        
        # Statistical features
        features.extend([
            np.mean(eye_resized),
            np.std(eye_resized),
            np.var(eye_resized),
            np.min(eye_resized),
            np.max(eye_resized)
        ])
        
        # Texture features (LBP-inspired)
        center = eye_resized[12:20, 12:20]
        features.append(np.mean(center))
        
        # Edge density
        edges = cv2.Canny(eye_resized, 50, 150)
        features.append(np.sum(edges) / (32 * 32))
        
        # Gradient features
        grad_x = cv2.Sobel(eye_resized, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(eye_resized, cv2.CV_64F, 0, 1, ksize=3)
        features.extend([
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y)),
            np.std(grad_x),
            np.std(grad_y)
        ])
        
        # Symmetry feature
        left_half = eye_resized[:, :16]
        right_half = cv2.flip(eye_resized[:, 16:], 1)
        symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
        features.append(symmetry if not np.isnan(symmetry) else 0)
        
        return np.array(features)
    
    def predict_eye_state(self, eye_region):
        """Dự đoán trạng thái mắt bằng ML model"""
        if self.ml_model is None:
            return None
            
        try:
            features = self.extract_eye_features(eye_region)
            features = features.reshape(1, -1)
            
            # Predict probability
            if hasattr(self.ml_model, 'predict_proba'):
                prob = self.ml_model.predict_proba(features)[0]
                return prob[0]  # Probability of closed eye
            else:
                return self.ml_model.predict(features)[0]
        except Exception as e:
            print(f"ML prediction error: {e}")
            return None
    
    def analyze_blink_pattern(self):
        """Phân tích pattern nhấp nháy mắt"""
        if len(self.ear_history) < 10:
            return False
            
        recent_ears = self.ear_history[-10:]
        
        # Detect rapid blinks (sign of fatigue)
        blink_count = sum(1 for ear in recent_ears if ear < self.BLINK_THRESHOLD)
        
        # Normal: 15-20 blinks/minute, Fatigue: >25 blinks/minute
        if blink_count > 6:  # >6 blinks in 10 frames (~0.33s)
            return True
            
        return False
    
    def play_alarm_sound(self):
        """Multi-level alert system"""
        print("🚨 DROWSINESS ALERT TRIGGERED!")
        
        try:
            if self.winsound:
                # Progressive alarm pattern
                frequencies = [800, 1000, 1200]
                for freq in frequencies:
                    self.winsound.Beep(freq, 300)
                print("✅ Progressive alarm played")
                return
        except:
            pass
            
        # Fallback visual alert
        print("\n" + "="*60)
        print("🚨 WAKE UP! DROWSINESS DETECTED! 🚨")
        print("="*60 + "\n")
    

    
    def detect(self, frame):
        """Main detection pipeline - ML + Computer Vision"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        drowsy_status = False
        ear_value = 0.0
        mar_value = 0.0
        ml_confidence = 0.0
        
        # === COMPUTER VISION PROCESSING ===
        if self.use_dlib:
            faces = self.detector(gray)
            
            for face in faces:
                landmarks = self.predictor(gray, face)
                landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
                
                # Extract facial features
                left_eye = landmarks[self.LEFT_EYE_POINTS]
                right_eye = landmarks[self.RIGHT_EYE_POINTS]
                mouth = landmarks[self.MOUTH_POINTS]
                
                # === COMPUTER VISION METRICS ===
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                ear_value = (left_ear + right_ear) / 2.0
                
                # Mouth analysis for yawning
                mar_value = self.mouth_aspect_ratio(mouth)
                
                # === MACHINE LEARNING PREDICTION ===
                if self.ml_model:
                    left_eye_region = self._extract_eye_region(gray, left_eye)
                    right_eye_region = self._extract_eye_region(gray, right_eye)
                    
                    left_pred = self.predict_eye_state(left_eye_region)
                    right_pred = self.predict_eye_state(right_eye_region)
                    
                    if left_pred is not None and right_pred is not None:
                        ml_confidence = (left_pred + right_pred) / 2.0
                
                # === VISUALIZATION ===
                # Draw eye contours
                cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 255), 1)
                
                # Face bounding box
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)
                
        else:
            # === HAAR CASCADE FALLBACK ===
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                if len(eyes) >= 2:
                    total_ratio = 0
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(frame[y:y+h, x:x+w], (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                        eye_ratio = eh / ew if ew > 0 else 0.3
                        total_ratio += eye_ratio
                    
                    ear_value = (total_ratio / len(eyes)) * 0.4  # Normalize
        
        # === UPDATE HISTORY FOR PATTERN ANALYSIS ===
        self.ear_history.append(ear_value)
        self.mar_history.append(mar_value)
        if len(self.ear_history) > 30:  # Keep last 1 second
            self.ear_history.pop(0)
            self.mar_history.pop(0)
        
        # === MULTI-MODAL DROWSINESS DETECTION ===
        drowsy_indicators = 0
        
        # 1. EAR-based detection
        if ear_value < self.EAR_THRESHOLD:
            self.frame_counter += 1
            drowsy_indicators += 1
        else:
            self.frame_counter = 0
            
        # 2. ML-based detection
        if ml_confidence > 0.7:  # High confidence of closed eyes
            drowsy_indicators += 1
            
        # 3. Yawning detection
        if mar_value > self.YAWN_THRESHOLD:
            self.yawn_counter += 1
            drowsy_indicators += 1
        else:
            self.yawn_counter = 0
            
        # 4. Blink pattern analysis
        if self.analyze_blink_pattern():
            drowsy_indicators += 1
        
        # === DECISION LOGIC ===
        if (self.frame_counter >= self.CONSECUTIVE_FRAMES or 
            drowsy_indicators >= 2 or 
            self.yawn_counter >= 15):
            
            drowsy_status = True
            if not self.alarm_playing:
                print(f"🚨 DROWSINESS: EAR={ear_value:.3f}, ML={ml_confidence:.3f}, MAR={mar_value:.3f}")
                self.play_alarm_sound()
                self.alarm_playing = True
        else:
            self.alarm_playing = False
        
        # === DISPLAY METRICS ===
        self._draw_metrics(frame, ear_value, mar_value, ml_confidence, drowsy_indicators)
        
        return frame, drowsy_status, ear_value
    
    def _draw_metrics(self, frame, ear, mar, ml_conf, indicators):
        """Draw all metrics on frame"""
        y_offset = 30
        
        # Main metrics
        cv2.putText(frame, f"EAR: {ear:.3f} (T: {self.EAR_THRESHOLD})", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"MAR: {mar:.3f} (Yawn: {self.YAWN_THRESHOLD})", 
                   (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if ml_conf > 0:
            cv2.putText(frame, f"ML Confidence: {ml_conf:.3f}", 
                       (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Status indicators
        cv2.putText(frame, f"Closed Frames: {self.frame_counter}/{self.CONSECUTIVE_FRAMES}", 
                   (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(frame, f"Risk Indicators: {indicators}/4", 
                   (10, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 0, 255) if indicators >= 2 else (0, 255, 0), 2)
        
        # Alert status
        if self.alarm_playing:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, y_offset + 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    def test_sound(self):
        """Test sound system on startup"""
        try:
            if self.winsound:
                self.winsound.Beep(440, 200)
                print("🔊 Sound system OK")
        except:
            print("⚠️ Sound system unavailable")
    
    def _extract_eye_region(self, gray_frame, eye_landmarks):
        """Extract eye region for ML processing"""
        try:
            x_coords = [point[0] for point in eye_landmarks]
            y_coords = [point[1] for point in eye_landmarks]
            
            x_min, x_max = max(0, min(x_coords) - 5), min(gray_frame.shape[1], max(x_coords) + 5)
            y_min, y_max = max(0, min(y_coords) - 5), min(gray_frame.shape[0], max(y_coords) + 5)
            
            if x_max > x_min and y_max > y_min:
                eye_region = gray_frame[y_min:y_max, x_min:x_max]
                return eye_region
            else:
                return None
        except:
            return None