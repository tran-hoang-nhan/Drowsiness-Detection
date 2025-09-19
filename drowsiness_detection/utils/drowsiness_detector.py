import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import pygame
import time

class DrowsinessDetector:
    def __init__(self):
        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks indices for MediaPipe
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Drowsiness parameters
        self.EAR_THRESHOLD = 0.25
        self.CONSECUTIVE_FRAMES = 20
        self.frame_counter = 0
        self.drowsy_counter = 0
        
        # Initialize pygame for alarm
        pygame.mixer.init()
        self.alarm_sound = None
        self.alarm_playing = False
        
    def eye_aspect_ratio(self, eye):
        """Calculate Eye Aspect Ratio (EAR)"""
        # Vertical distances
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # Horizontal distance
        C = dist.euclidean(eye[0], eye[3])
        
        # EAR formula
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect(self, frame):
        """Main detection function"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        drowsy_status = False
        ear_value = 0.0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get landmarks
                h, w, _ = frame.shape
                landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])
                
                # Extract eye coordinates
                left_eye = landmarks[self.LEFT_EYE]
                right_eye = landmarks[self.RIGHT_EYE]
                
                # Calculate EAR for both eyes
                left_ear = self.eye_aspect_ratio(left_eye[:6])  # Use first 6 points
                right_ear = self.eye_aspect_ratio(right_eye[:6])
                ear_value = (left_ear + right_ear) / 2.0
                
                # Draw eye contours
                cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
                
                # Check for drowsiness
                if ear_value < self.EAR_THRESHOLD:
                    self.frame_counter += 1
                    
                    if self.frame_counter >= self.CONSECUTIVE_FRAMES:
                        drowsy_status = True
                        self.drowsy_counter += 1
                        
                        # Draw warning
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Play alarm (simplified for web)
                        if not self.alarm_playing:
                            self.alarm_playing = True
                else:
                    self.frame_counter = 0
                    self.alarm_playing = False
                
                # Display EAR value
                cv2.putText(frame, f"EAR: {ear_value:.2f}", (300, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw face bounding box
                x_coords = [lm[0] for lm in landmarks]
                y_coords = [lm[1] for lm in landmarks]
                x, y, w, h = min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        return frame, drowsy_status, ear_value