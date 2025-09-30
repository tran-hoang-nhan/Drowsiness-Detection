"""
Visualization Module
"""
import cv2

class Visualizer:
    def __init__(self):
        pass
    
    def draw_face_detection(self, frame, detection_result):
        """Draw face detection results"""
        if not detection_result:
            return
        
        method = detection_result['method']
        
        if method == 'MTCNN':
            x, y, w, h = detection_result['box']
            keypoints = detection_result['keypoints']
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(frame, keypoints['left_eye'], 3, (0, 255, 0), -1)
            cv2.circle(frame, keypoints['right_eye'], 3, (0, 255, 0), -1)
            cv2.circle(frame, keypoints['nose'], 3, (0, 255, 255), -1)
            
        elif method == 'Haar':
            x, y, w, h = detection_result['box']
            eyes = detection_result['eyes']
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            for eye in eyes[:2]:  # Draw first 2 eyes
                cv2.rectangle(frame, (x+eye[0], y+eye[1]), 
                            (x+eye[0]+eye[2], y+eye[1]+eye[3]), (255, 255, 0), 2)
    
    def draw_metrics(self, frame, ear_value, ml_confidence, drowsy_indicators, 
                    method, alert_system):
        """Draw metrics on frame"""
        y_offset = 30
        
        cv2.putText(frame, f"{method} + ML Detection", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # EAR with warning color
        ear_color = (0, 0, 255) if ear_value < alert_system.EAR_THRESHOLD else (255, 255, 255)
        cv2.putText(frame, f"EAR: {ear_value:.3f} (T: {alert_system.EAR_THRESHOLD})", 
                   (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)
        
        if ml_confidence > 0:
            ml_color = (0, 0, 255) if ml_confidence > alert_system.ML_THRESHOLD else (0, 255, 255)
            cv2.putText(frame, f"ML Confidence: {ml_confidence:.3f}", 
                       (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ml_color, 2)
        
        cv2.putText(frame, f"Closed Frames: {alert_system.frame_counter}/{alert_system.CONSECUTIVE_FRAMES}", 
                   (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(frame, f"Risk Indicators: {drowsy_indicators}/3", 
                   (10, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 0, 255) if drowsy_indicators >= 2 else (0, 255, 0), 2)
        
        if alert_system.alarm_playing:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, y_offset + 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)