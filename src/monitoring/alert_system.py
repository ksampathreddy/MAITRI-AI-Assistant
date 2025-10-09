import smtplib
import logging
from email.mime.text import MimeText
from typing import Dict
import json
from datetime import datetime

class AlertSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alert_history = []
    
    def send_alert(self, emotional_summary: Dict):
        """Send alert for critical emotional states"""
        
        critical_emotion = emotional_summary.get("critical_emotion", "")
        confidence = emotional_summary.get("confidence", 0)
        wellbeing = emotional_summary.get("wellbeing_score", 0)
        
        alert_message = {
            "timestamp": datetime.now().isoformat(),
            "type": "emotional_crisis",
            "severity": emotional_summary["risk_level"],
            "emotion": critical_emotion,
            "confidence": confidence,
            "wellbeing_score": wellbeing,
            "message": f"Critical emotional state detected: {critical_emotion} with {confidence:.2f} confidence"
        }
        
        # Log alert
        self.alert_history.append(alert_message)
        
        # In real implementation, this would send to ground control
        self._simulate_ground_control_alert(alert_message)
        
        self.logger.warning(f"ALERT: {alert_message['message']}")
    
    def _simulate_ground_control_alert(self, alert_message: Dict):
        """Simulate sending alert to ground control"""
        print(f"\n🚨 ALERT SENT TO GROUND CONTROL:")
        print(f"Time: {alert_message['timestamp']}")
        print(f"Severity: {alert_message['severity'].upper()}")
        print(f"Emotion: {alert_message['emotion']}")
        print(f"Confidence: {alert_message['confidence']:.2f}")
        print(f"Wellbeing Score: {alert_message['wellbeing_score']:.1f}")
        print("="*50)
    
    def generate_wellbeing_report(self) -> Dict:
        """Generate wellbeing report for ground control"""
        if not self.alert_history:
            return {"status": "normal", "message": "All systems normal"}
        
        recent_alerts = [alert for alert in self.alert_history 
                        if (datetime.now() - datetime.fromisoformat(alert["timestamp"])).days < 1]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "requires_attention" if recent_alerts else "normal",
            "total_alerts_today": len(recent_alerts),
            "recent_alerts": recent_alerts[-5:],  # Last 5 alerts
            "summary": f"Detected {len(recent_alerts)} critical emotional events in last 24 hours"
        }