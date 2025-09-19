import cv2
import numpy as np
from scipy.spatial import distance as dist
import time

class SimpleDrowsinessDetector:
    def __init__(self):
        # Load Haar cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Drowsiness parameters
        self.EAR_THRESHOLD = 0.25
        self.CONSECUTIVE_FRAMES = 15
        self.frame_counter = 0
        self.drowsy_counter = 0
        self.alarm_playing = False
        
        # Eye tracking
        self.last_eye_positions = []
        self.eye_closed_time = 0
        
    def eye_aspect_ratio(self, eye_points):
        """Calculate simplified EAR from eye rectangle"""
        if len(eye_points) < 4:
            return 0.3
        
        # Use eye rectangle dimensions as approximation
        x, y, w, h = eye_points
        # Simplified EAR calculation
        ear = h / w if w > 0 else 0.3
        return ear
    
    def detect_blink(self, eyes, frame_gray):
        """Detect if eyes are closed based on eye detection"""
        if len(eyes) < 2:
            return True  # Assume closed if can't detect both eyes
        
        blink_detected = False
        total_ear = 0
        
        for (ex, ey, ew, eh) in eyes:
            # Extract eye region
            eye_roi = frame_gray[ey:ey+eh, ex:ex+ew]
            
            # Calculate variance (closed eyes have less variance)
            variance = np.var(eye_roi)
            
            # Threshold for closed eye (adjust as needed)
            if variance < 200:  # Closed eye threshold
                blink_detected = True
            
            # Calculate simplified EAR
            ear = self.eye_aspect_ratio((ex, ey, ew, eh))
            total_ear += ear
        
        avg_ear = total_ear / len(eyes) if eyes else 0.3
        return blink_detected, avg_ear
    
    def detect(self, frame):
        """Main detection function"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        drowsy_status = False
        ear_value = 0.3
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Region of interest for eyes
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            
            # Draw eye rectangles
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Check for drowsiness
            if len(eyes) >= 2:
                blink_detected, ear_value = self.detect_blink(eyes, roi_gray)
                
                if blink_detected or len(eyes) == 0:
                    self.frame_counter += 1
                    self.eye_closed_time = time.time()
                    
                    if self.frame_counter >= self.CONSECUTIVE_FRAMES:
                        drowsy_status = True
                        self.drowsy_counter += 1
                        
                        # Draw warning
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        if not self.alarm_playing:
                            self.alarm_playing = True
                else:
                    self.frame_counter = 0
                    self.alarm_playing = False
            else:
                # No eyes detected - might be closed
                self.frame_counter += 1
                if self.frame_counter >= self.CONSECUTIVE_FRAMES:
                    drowsy_status = True
            
            # Display info
            cv2.putText(frame, f"Eyes: {len(eyes)}", (300, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Frames: {self.frame_counter}", (300, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            break  # Process only first face
        
        return frame, drowsy_status, ear_value