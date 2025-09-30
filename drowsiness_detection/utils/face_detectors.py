"""
Face Detection Module - MTCNN + Haar Cascade
"""
import cv2

try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except Exception:
    MTCNN_AVAILABLE = False

class FaceDetector:
    def __init__(self):
        self.setup_detectors()
    
    def setup_detectors(self):
        if MTCNN_AVAILABLE:
            try:
                self.mtcnn = MTCNN()
                self.use_mtcnn = True
                print("✅ MTCNN loaded")
            except:
                self.use_mtcnn = False
        else:
            self.use_mtcnn = False
        
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            print("✅ Haar Cascade loaded")
        except:
            print("❌ No face detection available")
    
    def detect_mtcnn(self, frame):
        if not self.use_mtcnn:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mtcnn.detect_faces(rgb_frame)
        
        if result:
            for face in result:
                if face['confidence'] > 0.8:
                    return {
                        'box': face['box'],
                        'keypoints': face['keypoints'],
                        'method': 'MTCNN'
                    }
        return None
    
    def detect_haar(self, gray_frame):
        if not hasattr(self, 'face_cascade'):
            return None
        
        faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: e[0])
                return {
                    'box': (x, y, w, h),
                    'eyes': eyes,
                    'method': 'Haar'
                }
        return None
    
    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        result = self.detect_mtcnn(frame)
        if result:
            return result
        
        return self.detect_haar(gray)