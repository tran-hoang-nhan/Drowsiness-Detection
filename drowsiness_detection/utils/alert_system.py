"""
Alert System Module
"""
import numpy as np

class AlertSystem:
    def __init__(self):
        self.EAR_THRESHOLD = 0.25
        self.CONSECUTIVE_FRAMES = 30
        self.ML_THRESHOLD = 0.5
        
        self.frame_counter = 0
        self.ear_history = []
        self.alarm_playing = False
        
        self.setup_sound()
    
    def setup_sound(self):
        """Setup sound system"""
        try:
            import winsound
            self.winsound = winsound
        except ImportError:
            self.winsound = None
    
    def update_ear_history(self, ear_value):
        """Update EAR history"""
        self.ear_history.append(ear_value)
        if len(self.ear_history) > 30:
            self.ear_history.pop(0)
    
    def analyze_drowsiness(self, ear_value, ml_confidence):
        """Analyze drowsiness indicators"""
        drowsy_indicators = 0
        
        # 1. EAR-based detection
        if ear_value < self.EAR_THRESHOLD:
            self.frame_counter += 1
            drowsy_indicators += 1
        else:
            self.frame_counter = max(0, self.frame_counter - 2)
            
        # 2. ML-based detection
        if ml_confidence > self.ML_THRESHOLD:
            drowsy_indicators += 1
        
        # 3. Trend analysis
        if len(self.ear_history) >= 10:
            recent_trend = np.mean(self.ear_history[-5:]) - np.mean(self.ear_history[-10:-5])
            if recent_trend < -0.02:
                drowsy_indicators += 1
        
        return drowsy_indicators
    
    def should_alert(self, ear_value, ml_confidence, drowsy_indicators):
        """Determine if alert should be triggered"""
        return (self.frame_counter >= self.CONSECUTIVE_FRAMES or 
                drowsy_indicators >= 2 or 
                (ear_value < 0.1 and ml_confidence > 0.5))
    
    def play_alarm(self):
        """Play alarm sound"""
        print("🚨 DROWSINESS ALERT!")
        
        try:
            if self.winsound:
                self.winsound.Beep(1000, 500)
        except:
            pass
    
    def process_frame(self, ear_value, ml_confidence):
        """Process frame and return drowsiness status"""
        self.update_ear_history(ear_value)
        drowsy_indicators = self.analyze_drowsiness(ear_value, ml_confidence)
        
        if self.should_alert(ear_value, ml_confidence, drowsy_indicators):
            if not self.alarm_playing:
                self.play_alarm()
                self.alarm_playing = True
            return True, drowsy_indicators
        else:
            self.alarm_playing = False
            return False, drowsy_indicators