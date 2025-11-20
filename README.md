# Hệ thống Phát hiện Buồn ngủ Tài xế

## Advanced Driver Drowsiness Detection System

Đề tài: Máy học và Xử lý ảnh - Thị giác máy tính

Hệ thống phát hiện buồn ngủ tài xế sử dụng kết hợp Computer Vision và Machine Learning. Ứng dụng này kết hợp Haar Cascade để phát hiện khuôn mặt, dlib để trích xuất 68 điểm trên khuôn mặt, tính toán Eye Aspect Ratio (EAR), và sử dụng mô hình Machine Learning để phân loại trạng thái mắt (mở/đóng). Hệ thống cung cấp giao diện web thời gian thực để giám sát và phát cảnh báo khi phát hiện người dùng buồn ngủ.

## Tính năng chính

### Computer Vision
- Phát hiện khuôn mặt sử dụng Haar Cascade
- Trích xuất 68 điểm trên khuôn mặt với dlib
- Tính toán Eye Aspect Ratio (EAR) từ landmarks
- Xử lý ảnh thời gian thực (30+ FPS)

### Machine Learning
- Bốn mô hình: Random Forest, Gradient Boosting, SVM, Logistic Regression
- 25 đặc trưng được trích xuất từ ảnh mắt
- Ensemble voting cho độ chính xác cao
- Cross-validation và hyperparameter tuning

### Giao diện
- Web interface đơn giản, thân thiện
- Hiển thị trạng thái mắt, độ tin cậy ML, số frame đóng liên tiếp
- Phát cảnh báo âm thanh khi phát hiện buồn ngủ
- Giám sát thời gian thực qua WebSocket

## Cài đặt và Sử dụng

### 1. Setup môi trường
```bash
cd d:\XLA\drowsiness_detection
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu
```
data/eyes/
├── open/     # Ảnh mắt mở
└── closed/   # Ảnh mắt đóng
```

### 3. Huấn luyện mô hình
```bash
python train.py
```

### 4. Chạy ứng dụng
```bash
python web.py
```
Truy cập: http://127.0.0.1:5000

## Kiến trúc hệ thống

### Pipeline xử lý

1. Camera: Capture frame từ webcam (30 FPS)
2. Haar Cascade: Phát hiện khuôn mặt
3. dlib: Trích xuất 68 landmarks
4. EAR Calculation: Tính Eye Aspect Ratio
5. Feature Extraction: Trích 25 đặc trưng
6. ML Model: Dự đoán trạng thái mắt
7. Decision: Kiểm tra ngưỡng, phát cảnh báo

### Công nghệ sử dụng

Backend:
- Flask + Flask-SocketIO: Web framework
- OpenCV: Xử lý ảnh
- dlib: Facial landmarks
- scikit-learn: Machine Learning
- NumPy, Pandas, SciPy: Xử lý dữ liệu

Frontend:
- HTML5, CSS3, JavaScript
- WebSocket: Real-time communication
- Bootstrap: Responsive design

## Tham số chính

- EAR_THRESHOLD: 0.2 (ngưỡng mắt đóng)
- CONSECUTIVE_FRAMES: 10 (số frame liên tiếp)
- ML_CONFIDENCE_THRESHOLD: 0.7 (ngưỡng ML)
- Camera FPS: 30 (khung hình/giây)

## Hiệu suất

- Độ chính xác: 96%+ trên test set
- Tốc độ: 30+ FPS trên CPU
- Latency: < 100ms
- False positive rate: < 3%

## Cấu trúc thư mục

```
drowsiness_detection/
├── web.py                       # Flask application
├── train.py                     # Training pipeline
├── requirements.txt             # Dependencies
├── templates/
│   └── index.html              # Web interface
├── static/
│   ├── style.css               # Styling
│   └── app.js                  # Frontend logic
├── utils/
│   ├── drowsiness_detector.py  # Main detection engine
│   ├── face_detectors.py       # Haar + dlib
│   ├── ml_predictor.py         # ML model
│   ├── feature_extractor.py    # Feature extraction
│   ├── ear_calculator.py       # EAR calculation
│   └── alert_system.py         # Alert handling
├── data/eyes/
│   ├── open/                   # Open eye images
│   └── closed/                 # Closed eye images
└── models/
    ├── eye_classifier.pkl      # Trained model
    └── shape_predictor_68_face_landmarks.dat
```

## Thuật toán

### Eye Aspect Ratio (EAR)
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```
Trong đó p1-p6 là 6 điểm trên khuôn mắt từ 68 landmarks.

### Machine Learning Pipeline
1. Trích xuất 25 đặc trưng từ ảnh mắt
2. Dự đoán bằng 4 mô hình (RF, GB, SVM, LR)
3. Ensemble voting để kết luận cuối cùng

## Giao diện sản phẩm

Giao diện của hệ thống được thiết kế theo kiến trúc client-server, với backend Flask và frontend HTML/CSS/JavaScript. Người dùng tương tác thông qua một giao diện web đơn giản, nơi có thể khởi động/dừng camera để theo dõi trạng thái mắt trong thời gian thực. Giao diện hiển thị luồng video từ webcam, kèm theo các thông tin chính như trạng thái mắt (mở/đóng), độ tin cậy ML (ML Confidence), số frame mắt đóng liên tiếp (Frames Closed), phương pháp phát hiện được sử dụng (Haar Cascade + dlib EAR + ML), và chỉ số EAR (Eye Aspect Ratio). Khi hệ thống phát hiện người dùng buồn ngủ (mắt đóng liên tiếp >= 10 frame hoặc ML confidence > 0.7), cảnh báo sẽ hiển thị dưới dạng chữ "DROWSINESS DETECTED!" bằng màu đỏ nổi bật trên màn hình, đồng thời phát âm thanh cảnh báo để thông báo cho người dùng.

## Ứng dụng thực tế

- Giám sát tài xế chuyên nghiệp (xe tải, xe bus)
- Hệ thống cảnh báo trong ô tô hiện đại (ADAS)
- Nghiên cứu về mệt mỏi lái xe
- Quản lý đội xe real-time
- Ứng dụng di động cho lái xe cá nhân

## Tác giả

Trần Hoàng Nhân

## License

MIT License - Mục đích học tập và nghiên cứu
