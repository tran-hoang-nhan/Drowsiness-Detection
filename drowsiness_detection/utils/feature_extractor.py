import cv2
import numpy as np
from scipy.spatial import distance as dist

def calculate_ear_from_dlib_landmarks(landmarks):
    """
    Calculate Eye Aspect Ratio (EAR) using dlib 68-point facial landmarks

    dlib landmarks indices:
    Left eye:  36-41 (6 points)
    Right eye: 42-47 (6 points)

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    where p1 is left corner, p4 is right corner, p2,p3,p5,p6 are top/bottom points

    Returns normalized EAR (0.0 = closed, 1.0 = fully open)
    """
    try:
        # Left eye landmarks (indices 36-41)
        left_eye = [landmarks[i] for i in range(36, 42)]

        # Right eye landmarks (indices 42-47)
        right_eye = [landmarks[i] for i in range(42, 48)]

        # Calculate EAR for left eye
        # Distance between vertical points (p2-p6, p3-p5)
        left_vertical_1 = dist.euclidean(left_eye[1], left_eye[5])
        left_vertical_2 = dist.euclidean(left_eye[2], left_eye[4])
        # Distance between horizontal points (p1-p4)
        left_horizontal = dist.euclidean(left_eye[0], left_eye[3])

        left_ear = (left_vertical_1 + left_vertical_2) / (2.0 * left_horizontal)

        # Calculate EAR for right eye
        right_vertical_1 = dist.euclidean(right_eye[1], right_eye[5])
        right_vertical_2 = dist.euclidean(right_eye[2], right_eye[4])
        right_horizontal = dist.euclidean(right_eye[0], right_eye[3])

        right_ear = (right_vertical_1 + right_vertical_2) / (2.0 * right_horizontal)

        # Average both eyes
        ear = (left_ear + right_ear) / 2.0

        return ear
    except:
        return 0.0

def calculate_ear_from_keypoints(left_eye, right_eye, nose):
    """
    Calculate Eye Aspect Ratio (EAR) from MTCNN keypoints (fallback method)
    Returns normalized EAR value (0.0 = closed, 1.0 = fully open)
    """
    try:
        eye_distance = dist.euclidean(left_eye, right_eye)
        if eye_distance == 0:
            return 0.0
        
        left_to_nose = dist.euclidean(left_eye, nose)
        right_to_nose = dist.euclidean(right_eye, nose)
        avg_vertical_dist = (left_to_nose + right_to_nose) / 2
        
        ear = avg_vertical_dist / eye_distance
        ear_normalized = np.clip(ear / 0.4, 0, 1)
        
        return ear_normalized
    except:
        return 0.0

def extract_eye_features(image):
    """Extract 25 advanced features from eye region"""
    if image is None or image.size == 0:
        return np.zeros(25)

    image = cv2.resize(image, (32, 32))
    features = []

    # Statistical features (6)
    features.extend([
        np.mean(image), np.std(image), np.var(image),
        np.min(image), np.max(image), np.median(image)
    ])

    # Texture features (3)
    center = image[12:20, 12:20]
    features.extend([
        np.mean(center), np.std(center),
        np.mean(center) - np.mean(image)
    ])

    # Edge features (1)
    edges = cv2.Canny(image, 30, 100)
    features.append(np.sum(edges) / (32 * 32))

    # Gradient features (4)
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    features.extend([
        np.mean(np.abs(grad_x)), np.mean(np.abs(grad_y)),
        np.std(grad_x), np.std(grad_y)
    ])

    # Morphological features (4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    features.extend([
        np.mean(opened), np.mean(closed),
        np.mean(image - opened), np.mean(closed - image)
    ])

    # Histogram features (7)
    hist = cv2.calcHist([image], [0], None, [7], [0, 256])
    features.extend(hist.flatten())

    return np.array(features)

def preprocess_eye_image(image):
    """
    Preprocess eye image for better feature extraction
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)

    # Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image

def analyze_eye_features(image):
    """
    Analyze and return detailed eye features for debugging
    """
    features = extract_eye_features(image)

    feature_names = [
        'mean', 'std', 'var', 'min', 'max', 'median',  # Statistical (6)
        'center_mean', 'center_std', 'center_diff',     # Texture (3)
        'edge_density',                                 # Edge (1)
        'grad_x_mean', 'grad_y_mean', 'grad_x_std', 'grad_y_std',  # Gradient (4)
        'morph_open', 'morph_close', 'open_diff', 'close_diff',    # Morphological (4)
        'hist_0', 'hist_1', 'hist_2', 'hist_3', 'hist_4', 'hist_5', 'hist_6',  # Histogram (7)
    ]

    return dict(zip(feature_names, features))
