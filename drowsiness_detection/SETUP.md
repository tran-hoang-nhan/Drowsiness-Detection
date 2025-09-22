# Setup Guide - MRL Eye Dataset

## Tại sao models/ và data/ không có trên GitHub?

**models/**: Chứa trained models (.pkl files ~5-10MB)
**data/**: MRL Eye Dataset (~2GB, 84,898 ảnh)

Các file này quá lớn cho GitHub và cần tải riêng.

## Cách setup

### 1. Chạy script setup
```bash
python setup_models.py
```

### 2. Tải MRL Eye Dataset
1. Truy cập: https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset
2. Tải file dataset
3. Giải nén vào `data/raw/`
4. Chạy: `python dataset.py`

### 3. Train model
```bash
python train.py
```

## Cấu trúc sau khi setup

```
data/
├── raw/                    # Raw dataset từ Kaggle
│   ├── closedLeftEyes/
│   ├── closedRightEyes/
│   ├── openLeftEyes/
│   └── openRightEyes/
└── eyes/                   # Organized dataset
    ├── open/              # 42,952 ảnh mắt mở
    └── closed/            # 41,946 ảnh mắt nhắm

models/
├── haarcascade_frontalface_default.xml
├── haarcascade_eye.xml
└── eye_classifier.pkl     # Trained Random Forest model
```

## Troubleshooting

**Lỗi "No module named cv2":**
```bash
pip install opencv-python
```

**Lỗi download:**
- Kiểm tra kết nối internet
- Chạy lại `python setup_models.py`

**Lỗi permission:**
```bash
# Windows
python setup_models.py

# Linux/Mac
sudo python setup_models.py
```