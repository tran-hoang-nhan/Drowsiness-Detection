# Hệ thống Cảnh báo Giấc ngủ cho Tài xế
## Driver Drowsiness Detection System

Hệ thống phát hiện và cảnh báo tình trạng buồn ngủ của tài xế sử dụng Computer Vision và Machine Learning với MRL Eye Dataset.

## Tính năng

- ✅ Phát hiện khuôn mặt và mắt real-time với OpenCV
- ✅ Machine Learning với Random Forest (85-90% accuracy)
- ✅ Dataset MRL Eye với 84,898 ảnh mắt chất lượng cao
- ✅ Cảnh báo âm thanh và visual khi phát hiện buồn ngủ
- ✅ Giao diện web responsive với Bootstrap
- ✅ Real-time streaming với WebSocket
- ✅ Jupyter Notebook phân tích dữ liệu chi tiết
- ✅ Menu launcher tích hợp

## Cài đặt

### 1. Clone repository
```bash
cd d:\XLA\drowsiness_detection
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup models
```bash
python setup_models.py
```

### 4. Chạy ứng dụng chính
```bash
python app.py
```

**Hoặc chạy từng thành phần:**

### Train model
```bash
python train.py
```

### Chạy web app
```bash
python web.py
```

### Phân tích dữ liệu
```bash
jupyter notebook analysis.ipynb
```

Truy cập web: http://localhost:8080

## Cấu trúc Project

```
drowsiness_detection/
├── app.py                 # Main launcher application
├── train.py              # Model training script
├── web.py                # Flask web application
├── analysis.ipynb        # Jupyter notebook phân tích
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
├── data/                # Dataset directory
│   ├── eyes/           # Organized eye images
│   │   ├── open/       # Open eye images
│   │   └── closed/     # Closed eye images
│   └── raw/            # Raw downloaded data
├── models/             # Trained models
│   └── eye_classifier.pkl
├── utils/              # Detection algorithms
│   ├── simple_detector.py
│   ├── trained_detector.py
│   └── drowsiness_detector.py
├── static/             # Web assets
│   ├── style.css
│   └── app.js
└── templates/          # HTML templates
    └── index.html
```

## Thuật toán

### Machine Learning Approach
- **Algorithm**: Random Forest Classifier
- **Features**: 15 statistical và visual features
- **Dataset**: MRL Eye Dataset (84,898 images)
- **Accuracy**: 85-90%
- **Training Time**: 30 giây - 2 phút

### Feature Extraction
1. **Basic Statistics**: Mean, std, variance, min, max intensity
2. **Histogram Features**: 8-bin intensity histogram
3. **Edge Detection**: Canny edge density
4. **Spatial Analysis**: Center region analysis

### Detection Pipeline
1. Phát hiện khuôn mặt bằng Haar Cascades
2. Phát hiện mắt trong vùng khuôn mặt
3. Trích xuất features từ ảnh mắt
4. Predict bằng trained Random Forest model
5. Cảnh báo nếu phát hiện mắt nhắm liên tiếp

## Dataset

- **Source**: MRL Eye Dataset từ Kaggle
- **Link**: https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset
- **Size**: 84,898 ảnh (42,952 open + 41,946 closed)
- **Format**: Grayscale images, 32x32 pixels
- **Quality**: High-quality eye images từ nhiều người khác nhau
- **Balance**: Tương đối cân bằng (50.6% open, 49.4% closed)

## Kết quả

- **Model Accuracy**: 85-90%
- **Real-time Performance**: 30+ FPS
- **False Positive Rate**: <10%
- **Response Time**: <1 giây
- **Model Size**: <10MB

## Phân tích dữ liệu

Sử dụng Jupyter Notebook `analysis.ipynb` để:
- Visualize dataset distribution
- Analyze sample images
- Evaluate model performance
- Generate confusion matrix và classification report
- Feature importance analysis

## Ứng dụng thực tế

- **Driver Monitoring**: Hệ thống cảnh báo buồn ngủ cho tài xế
- **Workplace Safety**: Giám sát nhân viên trong môi trường nguy hiểm
- **Medical Monitoring**: Theo dõi bệnh nhân
- **Education**: Đánh giá mức độ tập trung học sinh

## Công nghệ sử dụng

- **Backend**: Flask, OpenCV, scikit-learn
- **Frontend**: HTML5, Bootstrap 5, JavaScript  
- **Real-time**: WebSocket (Socket.IO)
- **Machine Learning**: Random Forest, Feature Engineering
- **Data Analysis**: Jupyter, Matplotlib, Seaborn
- **Computer Vision**: Haar Cascades, Image Processing

## Tác giả

Đề tài môn Máy học - Xử lý ảnh và Thị giác máy tính

## Technical Specifications

- **Python**: 3.8+
- **OpenCV**: 4.8+
- **scikit-learn**: 1.3+
- **Flask**: 2.3+
- **Memory Usage**: <50MB RAM
- **CPU Usage**: <10% on modern processors
- **Supported OS**: Windows, Linux, macOS