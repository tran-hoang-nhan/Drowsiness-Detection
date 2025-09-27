import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.feature_extractor import extract_eye_features

class AdvancedEyeStateClassifier:
    def __init__(self):
        # === PIPELINES ===
        self.pipelines = {
            'random_forest': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
            ]),
            'gradient_boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(n_estimators=50, random_state=42))
            ]),
            'svm': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(probability=True, random_state=42))
            ]),
            'logistic': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ])
        }
        
        self.best_pipeline = None
        self.best_accuracy = 0
        self.best_pipeline_name = ""
        

    
    def load_dataset(self, data_path='data/eyes'):
        """Load and preprocess dataset with advanced feature extraction"""
        X, y = [], []
        
        print("📂 Loading dataset with advanced features...")
        
        # Load open eyes (label = 1)
        open_path = os.path.join(data_path, 'open')
        if os.path.exists(open_path):
            open_files = [f for f in os.listdir(open_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            print(f"   Processing {len(open_files)} open eye images...")
            
            for img_name in tqdm(open_files, desc="Open eyes"):
                img_path = os.path.join(open_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None and img.size > 0:
                    features = extract_eye_features(img)
                    if not np.any(np.isnan(features)):
                        X.append(features)
                        y.append(1)
        
        # Load closed eyes (label = 0)
        closed_path = os.path.join(data_path, 'closed')
        if os.path.exists(closed_path):
            closed_files = [f for f in os.listdir(closed_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            print(f"   Processing {len(closed_files)} closed eye images...")
            
            for img_name in tqdm(closed_files, desc="Closed eyes"):
                img_path = os.path.join(closed_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None and img.size > 0:
                    features = extract_eye_features(img)
                    if not np.any(np.isnan(features)):
                        X.append(features)
                        y.append(0)
        
        return np.array(X), np.array(y)
    
    def train_models_with_cv(self, X, y):
        """Train pipelines"""
        print("\n🚀 Pipeline Training...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        for name, pipeline in tqdm(self.pipelines.items(), desc="Training pipelines"):
            print(f"\n🤖 Training {name}...")
            
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='accuracy')
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'pipeline': pipeline,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_pipeline = pipeline
                self.best_pipeline_name = name
        
        return results
    
    def plot_results(self, results):
        """Visualize training results"""
        try:
            models = list(results.keys())
            accuracies = [results[model]['accuracy'] for model in models]
            
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
            plt.title('Pipeline Accuracy')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            plt.ylim(0.9, 1.0)
            
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                        f'{acc:.3f}', ha='center', va='bottom')
            
            plt.subplot(1, 2, 2)
            best_result = results[self.best_pipeline_name]
            cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
            
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Best: {self.best_pipeline_name}')
            plt.colorbar()
            
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['Closed', 'Open'])
            plt.yticks(tick_marks, ['Closed', 'Open'])
            
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha='center', va='center')
            
            plt.tight_layout()
            plt.savefig('models/training_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def save_model(self):
        """Save best pipeline"""
        os.makedirs('models', exist_ok=True)
        
        pipeline_data = {
            'pipeline': self.best_pipeline,
            'pipeline_name': self.best_pipeline_name,
            'accuracy': self.best_accuracy,
            'feature_count': 25
        }
        
        with open('models/eye_classifier.pkl', 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        print(f"💾 Pipeline saved: models/eye_classifier.pkl")

def main():
    print("🤖 Eye State Classifier Training")
    print("=" * 40)
    
    classifier = AdvancedEyeStateClassifier()
    
    data_path = 'data/eyes'
    X, y = classifier.load_dataset(data_path)
    
    if len(X) == 0:
        print("❌ No dataset found!")
        return
    
    print(f"\n📊 Dataset Info:")
    print(f"   Total samples: {len(X):,}")
    print(f"   Open eyes: {np.sum(y):,} ({np.sum(y)/len(y)*100:.1f}%)")
    print(f"   Closed eyes: {len(y) - np.sum(y):,} ({(len(y) - np.sum(y))/len(y)*100:.1f}%)")
    print(f"   Features per sample: {X.shape[1]}")
    
    results = classifier.train_models_with_cv(X, y)
    
    print(f"\n📈 Results:")
    for name, result in results.items():
        print(f"   {name:15} | Accuracy: {result['accuracy']:.4f}")
    
    classifier.plot_results(results)
    classifier.save_model()
    
    print(f"\n✅ Training Completed!")
    print(f"🏆 Best pipeline: {classifier.best_pipeline_name}")
    print(f"📊 Best accuracy: {classifier.best_accuracy:.4f}")

if __name__ == "__main__":
    main()