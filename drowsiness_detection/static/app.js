const socket = io();
let isDetecting = false;
let alertCount = 0;
let sessionStartTime = null;
let sessionTimer = null;

// DOM elements
const videoFeed = document.getElementById('video-feed');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusAlert = document.getElementById('status-alert');
const earProgress = document.getElementById('ear-progress');
const earValue = document.getElementById('ear-value');
const alertCountEl = document.getElementById('alert-count');
const sessionTimeEl = document.getElementById('session-time');
const earThreshold = document.getElementById('ear-threshold');
const soundAlert = document.getElementById('sound-alert');

// Audio for alerts
const alertAudio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT');

// Event listeners
startBtn.addEventListener('click', startDetection);
stopBtn.addEventListener('click', stopDetection);
earThreshold.addEventListener('input', updateThreshold);

function startDetection() {
    isDetecting = true;
    alertCount = 0;
    sessionStartTime = Date.now();
    
    startBtn.disabled = true;
    stopBtn.disabled = false;
    
    videoFeed.src = '/video_feed';
    statusAlert.className = 'alert alert-success';
    statusAlert.innerHTML = '<i class="fas fa-eye"></i> Đang phát hiện...';
    
    socket.emit('start_detection');
    startSessionTimer();
}

function stopDetection() {
    isDetecting = false;
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    
    videoFeed.src = '';
    statusAlert.className = 'alert alert-info';
    statusAlert.innerHTML = '<i class="fas fa-info-circle"></i> Đã dừng phát hiện';
    
    socket.emit('stop_detection');
    stopSessionTimer();
}

function updateThreshold() {
    const value = earThreshold.value;
    earThreshold.nextElementSibling.textContent = value;
}

function startSessionTimer() {
    sessionTimer = setInterval(() => {
        if (sessionStartTime) {
            const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            sessionTimeEl.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }, 1000);
}

function stopSessionTimer() {
    if (sessionTimer) {
        clearInterval(sessionTimer);
        sessionTimer = null;
    }
}

// Socket event listeners
socket.on('status_update', (data) => {
    if (!isDetecting) return;
    
    // Update EAR value
    const earVal = data.ear_value;
    earValue.textContent = earVal.toFixed(2);
    
    // Update progress bar
    const progressPercent = Math.min(100, (earVal / 0.4) * 100);
    earProgress.style.width = `${progressPercent}%`;
    
    if (earVal < parseFloat(earThreshold.value)) {
        earProgress.className = 'progress-bar bg-danger';
    } else {
        earProgress.className = 'progress-bar bg-success';
    }
    
    // Handle drowsiness alert
    if (data.drowsy) {
        alertCount++;
        alertCountEl.textContent = alertCount;
        
        statusAlert.className = 'alert alert-danger alert-drowsy';
        statusAlert.innerHTML = '<i class="fas fa-exclamation-triangle"></i> <strong>CẢNH BÁO: TÀI XẾ BUỒN NGỦ!</strong>';
        
        // Play sound alert
        if (soundAlert.checked) {
            alertAudio.play().catch(e => console.log('Audio play failed:', e));
        }
        
        // Vibrate if supported
        if (navigator.vibrate) {
            navigator.vibrate([200, 100, 200]);
        }
    } else {
        statusAlert.className = 'alert alert-success';
        statusAlert.innerHTML = '<i class="fas fa-eye"></i> Tài xế tỉnh táo';
    }
});

socket.on('detection_started', () => {
    console.log('Detection started');
});

socket.on('detection_stopped', () => {
    console.log('Detection stopped');
});