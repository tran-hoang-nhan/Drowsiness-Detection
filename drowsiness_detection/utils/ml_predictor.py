"""
Machine Learning Predictor Module
"""
import pickle
from .feature_extractor import extract_eye_features

class MLPredictor:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained ML model"""
        try:
            import os
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'eye_classifier.pkl')
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                if isinstance(model_data, dict):
                    self.model = model_data.get('pipeline', model_data.get('model'))
                    print(f"✅ ML model loaded: {model_data.get('pipeline_name', 'Unknown')}")
                else:
                    self.model = model_data
                    print("✅ ML model loaded")
        except Exception as e:
            print(f"⚠️ ML model not found: {e}")
    
    def predict_eye_state(self, eye_region):
        """Predict eye state (closed/open)"""
        if self.model is None or eye_region is None:
            return None
            
        try:
            features = extract_eye_features(eye_region)
            features = features.reshape(1, -1)
            
            if hasattr(self.model, 'predict_proba'):
                prob = self.model.predict_proba(features)[0]
                return prob[0]  # Probability of closed eye
            else:
                return self.model.predict(features)[0]
        except Exception:
            return None