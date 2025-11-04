"""
Face Detection Module - MTCNN + dlib Facial Landmarks
"""
import cv2
import dlib

try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except Exception:
    MTCNN_AVAILABLE = False

class FaceDetector:
    def __init__(self):
        self.setup_detectors()
    
    def setup_detectors(self):
        # Initialize MTCNN for face detection
        if MTCNN_AVAILABLE:
            try:
                self.mtcnn = MTCNN()
                self.use_mtcnn = True
                print("✅ MTCNN loaded")
            except:
                self.use_mtcnn = False
                print("⚠️ MTCNN failed to load")
        else:
            self.use_mtcnn = False
            print("⚠️ MTCNN not available")

        # Initialize dlib for facial landmarks (for precise EAR calculation)
        try:
            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
            self.use_dlib = True
            print("✅ dlib facial landmarks loaded")
        except:
            self.use_dlib = False
            print("⚠️ dlib facial landmarks not available")

    def detect_mtcnn(self, frame):
        """Detect face using MTCNN"""
        if not self.use_mtcnn:
            return None
        
        try:
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
        """Main detection pipeline: MTCNN + dlib landmarks"""
        detection = self.detect_mtcnn(frame)

        if detection:
            face_box = detection['box']
            # Get dlib landmarks using MTCNN face box
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

