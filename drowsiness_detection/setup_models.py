#!/usr/bin/env python3
"""
Setup script cho MRL Eye Dataset
"""
import os

def create_directories():
    """Tạo cấu trúc thư mục cho MRL Eye Dataset"""
    dirs = [
        'models',
        'data/raw', 
        'data/eyes/open',
        'data/eyes/closed'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✅ Đã tạo cấu trúc thư mục")

def check_dataset():
    """Kiểm tra MRL Eye Dataset"""
    dataset_path = 'data/raw'
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("⚠️ MRL Eye Dataset chưa có!")
        print("📋 Hướng dẫn tải dataset:")
        print("   1. Truy cập: https://www.kaggle.com/datasets/kutaykutlu/drowsiness-detection")
        print("   2. Tải file mrlEyes_2018_01.zip")
        print("   3. Giải nén vào thư mục data/raw/")
        print("   4. Chạy: python dataset.py để organize dataset")
        return False
    else:
        print("✅ Dataset đã có sẵn")
        return True

def check_trained_model():
    """Kiểm tra trained model"""
    if os.path.exists('models/eye_classifier.pkl'):
        print("✅ Trained model đã có")
        return True
    else:
        print("⚠️ Chưa có trained model - chạy 'python train.py'")
        return False

def main():
    print("🚀 Setup MRL Eye Dataset...")
    
    create_directories()
    dataset_ok = check_dataset()
    model_ok = check_trained_model()
    
    print("\n📋 Các bước tiếp theo:")
    
    if not dataset_ok:
        print("   1. Tải MRL Eye Dataset từ Kaggle")
        print("   2. python dataset.py (organize dataset)")
        print("   3. python train.py (train model)")
    elif not model_ok:
        print("   1. python train.py (train model)")
    else:
        print("   ✅ Tất cả đã sẵn sàng!")
    
    print("   🚀 python app.py (chạy ứng dụng)")

if __name__ == "__main__":
    main()