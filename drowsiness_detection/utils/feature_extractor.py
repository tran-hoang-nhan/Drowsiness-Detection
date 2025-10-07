import cv2
import numpy as np

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

    # Histogram features (8)
    hist = cv2.calcHist([image], [0], None, [8], [0, 256])
    features.extend(hist.flatten())

    # Symmetry feature (1)
    left_half = image[:, :16]
    right_half = cv2.flip(image[:, 16:], 1)
    correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
    features.append(correlation if not np.isnan(correlation) else 0)

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
        'hist_0', 'hist_1', 'hist_2', 'hist_3', 'hist_4', 'hist_5', 'hist_6', 'hist_7',  # Histogram (8)
        'symmetry'                                      # Symmetry (1)
    ]

    return dict(zip(feature_names, features))
