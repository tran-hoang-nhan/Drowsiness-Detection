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
from utils.feature_extractor import extract_eye_features, preprocess_eye_image

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
        """Load, preprocess and extract features from dataset"""
        X, y = [], []
        
        print("📂 Loading dataset with preprocessing + feature extraction...")

        # Load open eyes (label = 1)
        open_path = os.path.join(data_path, 'open')
        if os.path.exists(open_path):
            open_files = [f for f in os.listdir(open_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            for img_name in tqdm(open_files, desc=f"Processing {len(open_files)} open eye images"):
                img_path = os.path.join(open_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None and img.size > 0:
                    # PREPROCESSING
                    img = preprocess_eye_image(img)

                    # FEATURE EXTRACTION
                    features = extract_eye_features(img)
                    if not np.any(np.isnan(features)):
                        X.append(features)
                        y.append(1)
        
        # Load closed eyes (label = 0)
        closed_path = os.path.join(data_path, 'closed')
        if os.path.exists(closed_path):
            closed_files = [f for f in os.listdir(closed_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            for img_name in tqdm(closed_files, desc=f"Processing {len(closed_files)} closed eye images"):
                img_path = os.path.join(closed_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None and img.size > 0:
                    # ✅ BƯỚC 1: PREPROCESSING
                    img = preprocess_eye_image(img)

                    # ✅ BƯỚC 2: FEATURE EXTRACTION
                    features = extract_eye_features(img)
                    if not np.any(np.isnan(features)):
                        X.append(features)
                        y.append(0)
        
        return np.array(X), np.array(y)
    
    def train_models_with_cv(self, X, y):
        """Train pipelines with optimized strategy for best accuracy"""
        print("\n🚀 Pipeline Training...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        # Strategy 1: Quick screening of all models
        print("🔍 Quick model screening...")
        for name, pipeline in tqdm(self.pipelines.items(), desc="Screening models"):
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='accuracy')
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            print(f"   {name}: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # Strategy 2: Focus on top 2 performers with hyperparameter tuning
        sorted_models = sorted(results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
        top_models = [model[0] for model in sorted_models[:2]]

        print(f"\n🎯 Optimizing top models: {', '.join(top_models)}")

        final_results = {}

        for name in top_models:
            print(f"\n🤖 Optimizing {name}...")

            # Hyperparameter tuning for top models
            if name == 'random_forest':
                param_grid = {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [10, 20, None],
                    'classifier__min_samples_split': [2, 5]
                }
            elif name == 'gradient_boosting':
                param_grid = {
                    'classifier__n_estimators': [50, 100],
                    'classifier__learning_rate': [0.1, 0.2],
                    'classifier__max_depth': [3, 5]
                }
            elif name == 'svm':
                param_grid = {
                    'classifier__C': [1, 10],
                    'classifier__gamma': ['scale', 'auto']
                }
            else:  # logistic
                param_grid = {
                    'classifier__C': [1, 10],
                    'classifier__solver': ['liblinear', 'lbfgs']
                }

            # GridSearchCV with limited scope for speed
            grid_search = GridSearchCV(
                self.pipelines[name],
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_

            # Evaluate on test set
            y_pred = best_pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            final_results[name] = {
                'pipeline': best_pipeline,
                'accuracy': accuracy,
                'cv_mean': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"   Best params: {grid_search.best_params_}")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   CV Score: {grid_search.best_score_:.4f}")

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_pipeline = best_pipeline
                self.best_pipeline_name = name

        # Strategy 3: Create ensemble of top models if both perform well
        if len(final_results) >= 2:
            accuracies = [result['accuracy'] for result in final_results.values()]
            if min(accuracies) >= 0.95:  # Both models are good
                print("\n🏆 Creating ensemble classifier...")

                estimators = [(name, result['pipeline']) for name, result in final_results.items()]
                ensemble = VotingClassifier(estimators=estimators, voting='soft')
                ensemble.fit(X_train, y_train)

                y_pred_ensemble = ensemble.predict(X_test)
                ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

                print(f"   Ensemble accuracy: {ensemble_accuracy:.4f}")

                if ensemble_accuracy > self.best_accuracy:
                    self.best_accuracy = ensemble_accuracy
                    self.best_pipeline = ensemble
                    self.best_pipeline_name = "ensemble"

                    final_results['ensemble'] = {
                        'pipeline': ensemble,
                        'accuracy': ensemble_accuracy,
                        'cv_mean': ensemble_accuracy,
                        'y_test': y_test,
                        'y_pred': y_pred_ensemble
                    }

        return final_results

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
    """Main training workflow"""
    print("🎯 Advanced Eye State Classifier Training")
    print("="*50)

    # Initialize classifier
    classifier = AdvancedEyeStateClassifier()
    
    # Load and preprocess dataset
    print("\n📂 Loading dataset...")
    X, y = classifier.load_dataset()

    if len(X) == 0:
        print("❌ No valid data found! Please check:")
        print("   - data/eyes/open/ folder exists and contains images")
        print("   - data/eyes/closed/ folder exists and contains images")
        return
    
    print(f"✅ Dataset loaded: {len(X)} samples")
    print(f"   Features per sample: {X.shape[1] if len(X.shape) > 1 else 'Unknown'}")
    print(f"   Open eyes: {np.sum(y == 1)}")
    print(f"   Closed eyes: {np.sum(y == 0)}")

    # Train models
    results = classifier.train_models_with_cv(X, y)
    
    # Display results
    print(f"\n🏆 Best Model: {classifier.best_pipeline_name}")
    print(f"   Accuracy: {classifier.best_accuracy:.4f}")

    # Generate detailed report for best model
    best_result = results[classifier.best_pipeline_name]
    print(f"\n📊 Detailed Report for {classifier.best_pipeline_name}:")
    print(classification_report(
        best_result['y_test'],
        best_result['y_pred'],
        target_names=['Closed', 'Open']
    ))

    # Plot and save results
    classifier.plot_results(results)

    # Save the best model
    classifier.save_model()
    
    print("\n✅ Training completed successfully!")
    print("📁 Files saved:")
    print("   - models/eye_classifier.pkl (trained model)")
    print("   - models/training_results.png (visualization)")

if __name__ == "__main__":
    main()