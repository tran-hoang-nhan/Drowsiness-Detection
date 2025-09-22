from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
# import tensorflow as tf  # Not needed for simple version
from utils.trained_detector import TrainedDrowsinessDetector
import base64
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'drowsiness_detection_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
detector = TrainedDrowsinessDetector()
camera = None
detection_active = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

def generate_frames():
    global camera, detection_active
    camera = cv2.VideoCapture(0)
    
    while detection_active:
        success, frame = camera.read()
        if not success:
            break
        
        # Detect drowsiness
        result_frame, drowsy_status, ear_value = detector.detect(frame)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', result_frame)
        frame_bytes = buffer.tobytes()
        
        # Emit status to frontend + âm thanh
        socketio.emit('status_update', {
            'drowsy': drowsy_status,
            'ear_value': float(ear_value),
            'timestamp': time.time()
        })
        
        # Âm thanh cảnh báo qua web
        if drowsy_status:
            socketio.emit('play_alarm', {'sound': True})
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.1)
    
    if camera:
        camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('start_detection')
def start_detection():
    global detection_active
    detection_active = True
    emit('detection_started', {'status': 'started'})

@socketio.on('stop_detection')
def stop_detection():
    global detection_active, camera
    detection_active = False
    if camera:
        camera.release()
    emit('detection_stopped', {'status': 'stopped'})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='127.0.0.1', port=8080)