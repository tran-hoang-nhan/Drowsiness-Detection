"""
Face Detection Module - Haar Cascade + dlib Facial Landmarks
"""
import cv2
import dlib

class FaceDetector:
    def __init__(self):
        self.setup_detectors()
    
    def setup_detectors(self):
        # Initialize Haar Cascade for face detection
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
            self.use_haar = True
            print("✅ Haar Cascade loaded")
        except:
            self.use_haar = False
            print("⚠️ Haar Cascade failed to load")

        # Initialize dlib for facial landmarks (for precise EAR calculation)
        try:
            import os
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'shape_predictor_68_face_landmarks.dat')
            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_predictor = dlib.shape_predictor(model_path)
            self.use_dlib = True
            print("✅ dlib facial landmarks loaded")
        except Exception as e:
            self.use_dlib = False
            print(f"⚠️ dlib facial landmarks not available: {e}")

    def detect_haar(self, frame):
        """Detect face using Haar Cascade"""
        if not self.use_haar:
            return None
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            
            if len(faces) > 0:
                x, y, w, h = faces[0]  # Get first face
                return {
                    'box': (x, y, w, h),
                    'method': 'Haar Cascade'
                }
        except:
            pass

        return None
    
    def detect_dlib_landmarks(self, frame, face_box=None):
        """Detect 68 facial landmarks using dlib"""
        if not self.use_dlib:
            return None
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # If face_box provided (from MTCNN), use it; otherwise detect with dlib
            if face_box:
                x, y, w, h = face_box
                dlib_rect = dlib.rectangle(x, y, x + w, y + h)
            else:
                # Use dlib detector
                dlib_rects = self.dlib_detector(gray, 1)
                if len(dlib_rects) == 0:
                    return None
                dlib_rect = dlib_rects[0]

            # Get 68 landmarks
            landmarks = self.dlib_predictor(gray, dlib_rect)

            # Convert to dictionary with landmark indices
            landmark_dict = {}
            for i in range(68):
                pt = landmarks.part(i)
                landmark_dict[i] = (pt.x, pt.y)

            return landmark_dict
        except:
            return None

    def detect(self, frame):
        """Main detection pipeline: Haar Cascade + dlib landmarks"""
        detection = self.detect_haar(frame)

        if detection:
            face_box = detection['box']
            # Get dlib landmarks using Haar face box
            landmarks = self.detect_dlib_landmarks(frame, face_box)

            if landmarks:
                detection['dlib_landmarks'] = landmarks

            return detection

        # Fallback: try dlib direct detection
        landmarks = self.detect_dlib_landmarks(frame)
        if landmarks:
            return {
                'landmarks': landmarks,
                'method': 'dlib',
                'dlib_landmarks': landmarks
            }

        return None

