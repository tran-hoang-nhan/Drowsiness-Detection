#!/usr/bin/env python3
"""
Setup script cho MRL Eye Dataset với kagglehub
"""
import os
import shutil

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

def download_dataset():
    """Tải MRL Eye Dataset bằng kagglehub"""
    try:
        import kagglehub
        print("📥 Đang tải MRL Eye Dataset...")
        
        # Tải dataset
        path = kagglehub.dataset_download("kutaykutlu/drowsiness-detection")
        print(f"✅ Đã tải dataset tại: {path}")
        
        # Copy vào thư mục data/raw
        if os.path.exists(path):
            for item in os.listdir(path):
                src = os.path.join(path, item)
                dst = os.path.join('data/raw', item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            print("✅ Đã copy dataset vào data/raw/")
            return True
        
    except ImportError:
        print("❌ kagglehub chưa được cài đặt!")
        print("Chạy: pip install kagglehub")
        return False
    except Exception as e:
        print(f"❌ Lỗi tải dataset: {e}")
        print("📋 Cách khắc phục:")
        print("   1. Đăng nhập Kaggle: kagglehub.login()")
        print("   2. Hoặc tải thủ công từ Kaggle")
        return False

def check_dataset():
    """Kiểm tra MRL Eye Dataset"""
    dataset_path = 'data/raw'
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("⚠️ MRL Eye Dataset chưa có!")
        return download_dataset()
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