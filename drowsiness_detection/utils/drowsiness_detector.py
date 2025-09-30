"""
Main Drowsiness Detector - Tích hợp tất cả modules
"""
import cv2
import numpy as np
from .face_detectors import FaceDetector
from .ear_calculator import EARCalculator
from .ml_predictor import MLPredictor
from .alert_system import AlertSystem
from .visualizer import Visualizer

class DrowsinessDetector:
    def __init__(self):
        print("🚀 Initializing Drowsiness Detection System...")
        
        # Initialize all components
        self.face_detector = FaceDetector()
        self.ear_calculator = EARCalculator()
        self.ml_predictor = MLPredictor()
        self.alert_system = AlertSystem()
        self.visualizer = Visualizer()
        
        print("✅ System ready!")
    
    def detect(self, frame):
        """Main detection pipeline"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Detect face
        detection_result = self.face_detector.detect(frame)
        
        ear_value = 0.0
        ml_confidence = 0.0
        method = "None"
        
        if detection_result:
            method = detection_result['method']
            
            # 2. Calculate EAR
            if method == 'MTCNN':
                keypoints = detection_result['keypoints']
                ear_value = self.ear_calculator.calculate_mtcnn_ear(
                    keypoints['left_eye'], keypoints['right_eye'], keypoints['nose']
                )
                
                # 3. ML prediction
                left_region = self.ear_calculator.extract_eye_region(gray, keypoints['left_eye'])
                right_region = self.ear_calculator.extract_eye_region(gray, keypoints['right_eye'])
                
                left_pred = self.ml_predictor.predict_eye_state(left_region)
                right_pred = self.ml_predictor.predict_eye_state(right_region)
                
                if left_pred is not None and right_pred is not None:
                    ml_confidence = (left_pred + right_pred) / 2.0
                else:
                    # Backup pixel analysis
                    left_pixel = self.ear_calculator.analyze_eye_pixels(left_region)
                    right_pixel = self.ear_calculator.analyze_eye_pixels(right_region)
                    ml_confidence = (left_pixel + right_pixel) / 2.0
                    
            elif method == 'Haar':
                eyes = detection_result['eyes']
                if len(eyes) >= 2:
                    ear_value = self.ear_calculator.calculate_haar_ear(eyes[0], eyes[-1])
                    
                    # ML prediction for Haar
                    box = detection_result['box']
                    x, y, w, h = box
                    roi_gray = gray[y:y+h, x:x+w]
                    
                    left_roi = roi_gray[eyes[0][1]:eyes[0][1]+eyes[0][3], eyes[0][0]:eyes[0][0]+eyes[0][2]]
                    right_roi = roi_gray[eyes[-1][1]:eyes[-1][1]+eyes[-1][3], eyes[-1][0]:eyes[-1][0]+eyes[-1][2]]
                    
                    left_pred = self.ml_predictor.predict_eye_state(left_roi)
                    right_pred = self.ml_predictor.predict_eye_state(right_roi)
                    
                    if left_pred is not None and right_pred is not None:
                        ml_confidence = (left_pred + right_pred) / 2.0
            
            # 4. Draw visualization
            self.visualizer.draw_face_detection(frame, detection_result)
        
        # 5. Process alert
        drowsy_status, drowsy_indicators = self.alert_system.process_frame(ear_value, ml_confidence)
        
        # 6. Draw metrics
        self.visualizer.draw_metrics(frame, ear_value, ml_confidence, drowsy_indicators, method, self.alert_system)
        
        return frame, drowsy_status, ear_value
    
