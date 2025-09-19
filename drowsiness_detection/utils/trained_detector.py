import cv2
import numpy as np
import pickle
import os
from scipy.spatial import distance as dist
import time

class TrainedDrowsinessDetector:
    def __init__(self, model_path='models/eye_classifier.pkl'):
        # Load trained model
        self.model = None
        self.model_info = None
        self.load_model(model_path)
        
        # Face and eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Drowsiness parameters
        self.CONSECUTIVE_FRAMES = 15
        self.frame_counter = 0
        self.drowsy_counter = 0
        self.alarm_playing = False
        
        # Performance tracking
        self.detection_confidence = 0.0
        self.last_prediction_time = 0
        
    def load_model(self, model_path):
        """Load trained model"""
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.model_info = {
                    'name': model_data.get('model_name', 'Unknown'),
                    'accuracy': model_data.get('accuracy', 0.0),
                    'feature_count': model_data.get('feature_count', 15)
                }
                
                print(f"✅ Loaded model: {self.model_info['name']} (Accuracy: {self.model_info['accuracy']:.3f})")
                
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print("🔄 Falling back to simple detection...")
                self.model = None
        else:
            print(f"❌ Model not found: {model_path}")
            print("🔄 Please train a model first using train.py")
            self.model = None
    
    def extract_features(self, image):
        """Trích xuất features giống như trong training"""
        if image is None or image.size == 0:
            return np.zeros(15)
        
        # Resize về kích thước chuẩn
        image = cv2.resize(image, (32, 32))
        
        features = []
        
        # 1. Thống kê cơ bản
        features.append(np.mean(image))           # Mean intensity
        features.append(np.std(image))            # Standard deviation
        features.append(np.var(image))            # Variance
        features.append(np.min(image))            # Min intensity
        features.append(np.max(image))            # Max intensity
        
        # 2. Histogram features
        hist = cv2.calcHist([image], [0], None, [8], [0, 256])
        features.extend(hist.flatten())           # 8 histogram bins
        
        # 3. Edge density
        edges = cv2.Canny(image, 50, 150)
        features.append(np.sum(edges) / (32 * 32))  # Edge density
        
        # 4. Center region analysis
        center = image[8:24, 8:24]
        features.append(np.mean(center))          # Center mean
        
        return np.array(features).reshape(1, -1)
    
    def predict_eye_state(self, eye_image):
        """Predict eye state using trained model"""
        if self.model is None:
            # Fallback to simple detection
            return self.simple_eye_detection(eye_image)
        
        try:
            # Extract features
            features = self.extract_features(eye_image)
            
            # Predict
            prediction = self.model.predict(features)[0]
            
            # Get confidence if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                confidence = np.max(probabilities)
                self.detection_confidence = confidence
            else:
                self.detection_confidence = 0.8  # Default confidence
            
            return prediction  # 1 = open, 0 = closed
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self.simple_eye_detection(eye_image)
    
    def simple_eye_detection(self, eye_image):
        """Simple fallback detection method"""
        if eye_image is None or eye_image.size == 0:
            return 0
        
        # Calculate variance - closed eyes have less variance
        variance = np.var(eye_image)
        return 1 if variance > 300 else 0
    
    def detect(self, frame):
        """Main detection function"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        drowsy_status = False
        confidence = 0.0
        eyes_detected = 0
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # ROI for eyes
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            eyes_detected = len(eyes)
            
            # Analyze each eye
            open_eyes = 0
            total_confidence = 0
            
            for (ex, ey, ew, eh) in eyes:
                # Draw eye rectangle
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Extract eye image
                eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
                
                # Predict eye state
                eye_state = self.predict_eye_state(eye_img)
                open_eyes += eye_state
                total_confidence += self.detection_confidence
                
                # Draw eye state
                state_text = "OPEN" if eye_state == 1 else "CLOSED"
                color = (0, 255, 0) if eye_state == 1 else (0, 0, 255)
                cv2.putText(roi_color, state_text, (ex, ey-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Calculate average confidence
            if eyes_detected > 0:
                confidence = total_confidence / eyes_detected
            
            # Determine drowsiness
            if eyes_detected >= 2:
                # Both eyes closed or mostly closed
                if open_eyes <= eyes_detected * 0.3:  # 30% threshold
                    self.frame_counter += 1
                else:
                    self.frame_counter = 0
            else:
                # No eyes detected - might be closed
                self.frame_counter += 1
            
            # Check for drowsiness alert
            if self.frame_counter >= self.CONSECUTIVE_FRAMES:
                drowsy_status = True
                self.drowsy_counter += 1
                
                # Draw warning
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                if not self.alarm_playing:
                    self.alarm_playing = True
            else:
                self.alarm_playing = False
            
            # Display detection info
            info_y = 60
            cv2.putText(frame, f"Eyes: {eyes_detected}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if self.model is not None:
                cv2.putText(frame, f"Model: {self.model_info['name']}", (10, info_y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, info_y + 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"Frames: {self.frame_counter}", (10, info_y + 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            break  # Process only first face
        
        return frame, drowsy_status, confidence
    
    def get_model_info(self):
        """Get model information"""
        if self.model_info:
            return self.model_info
        else:
            return {
                'name': 'Simple Detection',
                'accuracy': 0.0,
                'feature_count': 0
            }