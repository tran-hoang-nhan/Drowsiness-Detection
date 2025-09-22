import cv2
import numpy as np
import pickle
import os

class DrowsinessDetector:
    def __init__(self):
        # Load trained model
        model_path = 'models/eye_classifier.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("✅ Đã load trained model")
        else:
            print("❌ Không tìm thấy trained model!")
            self.model = None
        
        # Load Haar Cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Detection parameters - 3 giây = 90 frames (30fps)
        self.CONSECUTIVE_FRAMES = 90  # 3 giây
        self.frame_counter = 0
        self.drowsy_counter = 0
        self.alarm_playing = False
        
        # Âm thanh cảnh báo
        self.sound_enabled = True
        try:
            import winsound
            self.winsound = winsound
        except ImportError:
            self.winsound = None
        
    def extract_features(self, eye_img):
        """Trích xuất features từ ảnh mắt"""
        if eye_img is None or eye_img.size == 0:
            return None
            
        # Resize về 32x32 như training data
        eye_resized = cv2.resize(eye_img, (32, 32))
        
        # Convert to grayscale nếu cần
        if len(eye_resized.shape) == 3:
            eye_gray = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2GRAY)
        else:
            eye_gray = eye_resized
        
        # Tính toán features (giống như trong train.py)
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(eye_gray),
            np.std(eye_gray), 
            np.var(eye_gray),
            np.min(eye_gray),
            np.max(eye_gray)
        ])
        
        # Histogram features
        hist = cv2.calcHist([eye_gray], [0], None, [8], [0, 256])
        features.extend(hist.flatten())
        
        # Edge features
        edges = cv2.Canny(eye_gray, 50, 150)
        features.append(np.sum(edges) / (32 * 32))
        
        # Center region analysis
        center = eye_gray[8:24, 8:24]
        features.append(np.mean(center))
        
        return np.array(features).reshape(1, -1)
    
    def play_alarm_sound(self):
        """Phát âm thanh cảnh báo"""
        try:
            if self.winsound:
                # Windows beep sound
                self.winsound.Beep(800, 500)  # 800Hz, 500ms
            else:
                # Fallback - system bell
                import os
                os.system('echo \a')
        except Exception as e:
            print(f"Không thể phát âm thanh: {e}")
    
    def detect(self, frame):
        """Phát hiện buồn ngủ bằng trained ML model"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        drowsy_status = False
        confidence = 0.0
        
        for (x, y, w, h) in faces:
            # Vẽ khung mặt
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Tìm mắt trong vùng mặt
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            closed_eyes = 0
            total_eyes = len(eyes)
            
            for (ex, ey, ew, eh) in eyes:
                # Vẽ khung mắt
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Trích xuất vùng mắt
                eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
                
                if self.model is not None:
                    # Trích xuất features
                    features = self.extract_features(eye_img)
                    
                    if features is not None:
                        # Dự đoán
                        prediction = self.model.predict(features)[0]
                        prob = self.model.predict_proba(features)[0]
                        confidence = max(prob)
                        
                        # Kiểm tra prediction và confidence
                        if prediction == 0 and confidence > 0.6:  # Mắt nhắm với confidence cao
                            closed_eyes += 1
                            cv2.putText(frame, "CLOSED", (x+ex, y+ey-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        elif prediction == 1 and confidence > 0.6:  # Mắt mở với confidence cao
                            cv2.putText(frame, "OPEN", (x+ex, y+ey-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            # Confidence thấp - bỏ qua
                            cv2.putText(frame, "UNSURE", (x+ex, y+ey-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Kiểm tra buồn ngủ - Chỉ đếm khi chắc chắn
            if total_eyes > 0:
                # Chỉ đếm khi có mắt nhắm với confidence cao
                if closed_eyes >= total_eyes * 0.8:  # 80% mắt nhắm
                    self.frame_counter += 1
                    
                    # Hiển thị đếm ngược
                    seconds_left = max(0, (self.CONSECUTIVE_FRAMES - self.frame_counter) / 30)
                    cv2.putText(frame, f"Eyes closed: {seconds_left:.1f}s", (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    if self.frame_counter >= self.CONSECUTIVE_FRAMES:
                        drowsy_status = True
                        self.drowsy_counter += 1
                        
                        # Hiển thị cảnh báo
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        
                        # Âm thanh cảnh báo
                        if not self.alarm_playing:
                            self.play_alarm_sound()
                            self.alarm_playing = True
                else:
                    if self.frame_counter > 0:
                        self.frame_counter = max(0, self.frame_counter - 2)  # Giảm dần
                    self.alarm_playing = False
            
            # Hiển thị thông tin chi tiết
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Closed: {closed_eyes}/{total_eyes}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame count: {self.frame_counter}", (300, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame, drowsy_status, confidence