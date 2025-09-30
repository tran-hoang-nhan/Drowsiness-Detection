"""
EAR (Eye Aspect Ratio) Calculator Module
"""
import cv2
import numpy as np
from scipy.spatial import distance as dist

class EARCalculator:
    def __init__(self):
        pass
    
    def calculate_mtcnn_ear(self, left_eye, right_eye, nose):

        eye_distance = dist.euclidean(left_eye, right_eye)
        if eye_distance == 0:
            return 0.1
        
        left_to_nose = dist.euclidean(left_eye, nose)
        right_to_nose = dist.euclidean(right_eye, nose)
        avg_eye_nose_dist = (left_to_nose + right_to_nose) / 2
        
        ear = avg_eye_nose_dist / eye_distance
        
        if ear > 0.8:
            return 0.3
        elif ear < 0.4:
            return 0.1
        else:
            return 0.2
    
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