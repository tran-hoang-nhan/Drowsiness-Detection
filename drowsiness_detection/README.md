# Hệ thống Cảnh báo Giấc ngủ cho Tài xế
## Driver Drowsiness Detection System

Hệ thống phát hiện và cảnh báo tình trạng buồn ngủ của tài xế sử dụng Computer Vision và Machine Learning.

## Tính năng

- ✅ Phát hiện khuôn mặt và mắt real-time
- ✅ Tính toán EAR (Eye Aspect Ratio) để phát hiện mắt nhắm
- ✅ Cảnh báo âm thanh và visual khi phát hiện buồn ngủ
- ✅ Giao diện web responsive với Bootstrap
- ✅ Real-time streaming với WebSocket
- ✅ Thống kê và theo dõi phiên làm việc
- ✅ Có thể điều chỉnh threshold và cài đặt

## Cài đặt

### 1. Clone repository
```bash
cd d:\XLA\drowsiness_detection
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Train model
```bash
python train.py
```

### 4. Chạy ứng dụng
```bash
python web.py
```

Truy cập: http://localhost:5000 hoặc http://0.0.0.0:5000

## Cấu trúc Project

```
drowsiness_detection/
├── web.py                 # Flask web application
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── dataset.py            # Dataset setup script
├── utils/
│   └── drowsiness_detector.py  # Core detection logic
├── models/               # Trained models và landmarks
├── data/                 # Dataset
├── static/              # CSS, JS files
│   ├── style.css
│   └── app.js
└── templates/           # HTML templates
    └── index.html
```

## Thuật toán

### Eye Aspect Ratio (EAR)
```
EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
```

Trong đó p1, p2, ..., p6 là các điểm landmark của mắt.

### Logic phát hiện buồn ngủ:
1. Phát hiện khuôn mặt bằng Haar Cascades
2. Phát hiện mắt trong vùng khuôn mặt
3. Phân tích trạng thái mắt (mở/nhắm)
4. Nếu mắt nhắm trong N frames liên tiếp → Cảnh báo

## Tham số có thể điều chỉnh

- **EAR_THRESHOLD**: 0.25 (mặc định)
- **CONSECUTIVE_FRAMES**: 20 frames
- **Cảnh báo âm thanh**: Bật/tắt

## Thuật toán Detection

Sử dụng OpenCV Haar Cascades:
- Phát hiện khuôn mặt
- Phát hiện mắt
- Phân tích variance của vùng mắt
- Đếm số frame mắt nhắm liên tiếp

## Kết quả mong đợi

- **Accuracy**: >95% trên test set
- **Real-time performance**: 30+ FPS
- **False positive rate**: <5%
- **Response time**: <1 giây

## Ứng dụng thực tế

- Hệ thống giám sát tài xế xe tải, xe bus
- Ứng dụng mobile cho tài xế cá nhân
- Tích hợp vào hệ thống ADAS (Advanced Driver Assistance Systems)
- Nghiên cứu về an toàn giao thông

## Công nghệ sử dụng

- **Backend**: Flask, OpenCV
- **Frontend**: HTML5, Bootstrap 5, JavaScript  
- **Real-time**: WebSocket (Socket.IO)
- **Computer Vision**: Haar Cascades, EAR algorithm

## Tác giả

Đề tài môn Máy học - Xử lý ảnh và Thị giác máy tính

## License

MIT License