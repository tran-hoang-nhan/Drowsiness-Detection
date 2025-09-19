#!/usr/bin/env python3
"""
Main application entry point for Drowsiness Detection System
"""

import sys
import os

def main():
    """Main application launcher"""
    print("🎯 Drowsiness Detection System")
    print("=" * 40)
    print("1. Train Model")
    print("2. Run Web Application") 
    print("3. Check Model Status")
    print("4. Exit")
    
    while True:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            print("\n🚀 Starting model training...")
            os.system("python train.py")
            break
            
        elif choice == '2':
            print("\n🌐 Starting web application...")
            os.system("python web.py")
            break
            
        elif choice == '3':
            print("\n🔍 Checking model status...")
            if os.path.exists('models/eye_classifier.pkl'):
                import pickle
                with open('models/eye_classifier.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                print(f"✅ Model found: {model_data['model_name']}")
                print(f"📊 Accuracy: {model_data['accuracy']:.4f}")
            else:
                print("❌ No trained model found!")
                print("💡 Run option 1 to train a model first")
            break
            
        elif choice == '4':
            print("👋 Goodbye!")
            sys.exit(0)
            
        else:
            print("❌ Invalid choice! Please select 1-4")

if __name__ == "__main__":
    main()