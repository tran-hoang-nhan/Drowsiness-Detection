import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle

class EyeStateClassifier:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
            # SVM removed - too slow with 84K+ samples
        }
        self.best_model = None
        self.best_accuracy = 0
        
    def extract_features(self, image):
        """Trích xuất đặc trưng từ ảnh mắt"""
        if image is None or image.size == 0:
            return np.zeros(15)
        
        image = cv2.resize(image, (32, 32))
        features = []
        
        # Basic statistics
        features.append(np.mean(image))
        features.append(np.std(image))
        features.append(np.var(image))
        features.append(np.min(image))
        features.append(np.max(image))
        
        # Histogram
        hist = cv2.calcHist([image], [0], None, [8], [0, 256])
        features.extend(hist.flatten())
        
        # Edge density
        edges = cv2.Canny(image, 50, 150)
        features.append(np.sum(edges) / (32 * 32))
        
        # Center region
        center = image[8:24, 8:24]
        features.append(np.mean(center))
        
        return np.array(features)
    
    def load_dataset(self, data_path='data/eyes'):
        """Load dataset từ thư mục"""
        X, y = [], []
        
        print("📂 Loading dataset...")
        
        # Load open eyes
        open_path = os.path.join(data_path, 'open')
        if os.path.exists(open_path):
            open_files = [f for f in os.listdir(open_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            print(f"   Found {len(open_files)} open eye images")
            
            for img_name in open_files:
                img_path = os.path.join(open_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    features = self.extract_features(img)
                    X.append(features)
                    y.append(1)
        
        # Load closed eyes
        closed_path = os.path.join(data_path, 'closed')
        if os.path.exists(closed_path):
            closed_files = [f for f in os.listdir(closed_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            print(f"   Found {len(closed_files)} closed eye images")
            
            for img_name in closed_files:
                img_path = os.path.join(closed_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    features = self.extract_features(img)
                    X.append(features)
                    y.append(0)
        
        return np.array(X), np.array(y)
    
    def train_models(self, X, y):
        """Train multiple models"""
        print("\n🚀 Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n📊 Training {name}...")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"   Accuracy: {accuracy:.4f}")
            
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name
        
        return results
    
    def save_model(self):
        """Save best model"""
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'accuracy': self.best_accuracy,
            'feature_count': 15
        }
        
        with open('models/eye_classifier.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved: models/eye_classifier.pkl")

def main():
    print("🎯 Eye State Classifier Training")
    print("=" * 50)
    
    classifier = EyeStateClassifier()
    
    # Load dataset
    data_path = 'data/eyes'
    X, y = classifier.load_dataset(data_path)
    
    if len(X) == 0:
        print("❌ No dataset found!")
        print("📁 Please put your dataset in:")
        print("   data/eyes/open/    (for open eye images)")
        print("   data/eyes/closed/  (for closed eye images)")
        return
    
    print(f"\n📊 Dataset Info:")
    print(f"   Total samples: {len(X):,}")
    print(f"   Open eyes: {np.sum(y):,} ({np.sum(y)/len(y)*100:.1f}%)")
    print(f"   Closed eyes: {len(y) - np.sum(y):,} ({(len(y) - np.sum(y))/len(y)*100:.1f}%)")
    print(f"   Features per sample: {X.shape[1]}")
    
    # Train models
    results = classifier.train_models(X, y)
    
    # Save best model
    classifier.save_model()
    
    print(f"\n✅ Training completed!")
    print(f"🏆 Best model: {classifier.best_model_name}")
    print(f"📊 Best accuracy: {classifier.best_accuracy:.4f}")
    print(f"🌐 Run web app: python web.py")

if __name__ == "__main__":
    main()