import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logger(name: str = "MAITRI", log_level: str = "INFO", 
                log_file: str = "logs/maitri_system.log") -> logging.Logger:
    """
    Setup and configure logger for MAITRI system
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler (detailed logs)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (simple logs)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

class MAITRILogger:
    """
    Enhanced logger with MAITRI-specific functionality
    """
    
    def __init__(self, name: str = "MAITRI", log_level: str = "INFO"):
        self.logger = setup_logger(name, log_level)
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.performance_metrics = {}
        
    def session_start(self, system_info: dict):
        """Log system startup"""
        self.logger.info(f"🚀 MAITRI Session Started: {self.session_id}")
        self.logger.info(f"System Info: {system_info}")
        self._log_performance("session_start", datetime.now())
        
    def session_end(self, session_summary: dict):
        """Log system shutdown"""
        self.logger.info(f"🛑 MAITRI Session Ended: {self.session_id}")
        self.logger.info(f"Session Summary: {session_summary}")
        self._log_performance("session_end", datetime.now())
        
    def emotion_analysis(self, emotion_data: dict, confidence: float):
        """Log emotion analysis results"""
        dominant_emotion = max(emotion_data.items(), key=lambda x: x[1])[0] if emotion_data else "unknown"
        self.logger.info(f"😊 Emotion Analysis: {dominant_emotion} (confidence: {confidence:.2f})")
        
    def intervention_triggered(self, intervention_type: str, level: str, reason: str):
        """Log intervention triggering"""
        self.logger.warning(f"🔄 Intervention Triggered: {intervention_type} - {level} - Reason: {reason}")
        
    def crisis_detected(self, crisis_type: str, severity: str, actions_taken: list):
        """Log crisis detection"""
        self.logger.critical(f"🚨 CRISIS DETECTED: {crisis_type} - Severity: {severity}")
        for action in actions_taken:
            self.logger.critical(f"🚨 Crisis Action: {action}")
            
    def ground_control_alert(self, alert_type: str, priority: str, message: str):
        """Log ground control alerts"""
        self.logger.critical(f"📡 GROUND CONTROL ALERT: {alert_type} - Priority: {priority} - {message}")
        
    def performance_metric(self, operation: str, duration: float, success: bool = True):
        """Log performance metrics"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.debug(f"⏱️ Performance: {operation} - {duration:.3f}s - {status}")
        self._log_performance(operation, duration)
        
    def user_interaction(self, interaction_type: str, user_input: str, response: str):
        """Log user interactions (with privacy considerations)"""
        # Sanitize user input for privacy
        sanitized_input = self._sanitize_user_input(user_input)
        self.logger.info(f"💬 User Interaction: {interaction_type} - Input: {sanitized_input}")
        self.logger.debug(f"💬 MAITRI Response: {response}")
        
    def system_health(self, component: str, status: str, details: dict = None):
        """Log system health status"""
        if status == "ERROR":
            self.logger.error(f"🔧 System Health: {component} - {status}")
            if details:
                self.logger.error(f"🔧 Error Details: {details}")
        elif status == "WARNING":
            self.logger.warning(f"🔧 System Health: {component} - {status}")
        else:
            self.logger.info(f"🔧 System Health: {component} - {status}")
            
    def data_processed(self, data_type: str, records_processed: int, processing_time: float):
        """Log data processing information"""
        self.logger.info(f"📊 Data Processed: {data_type} - Records: {records_processed} - Time: {processing_time:.2f}s")
        
    def _sanitize_user_input(self, user_input: str) -> str:
        """Sanitize user input for privacy in logs"""
        if len(user_input) > 50:
            return user_input[:47] + "..."
        return user_input
        
    def _log_performance(self, operation: str, value):
        """Log performance metrics internally"""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        self.performance_metrics[operation].append(value)
        
    def get_performance_report(self) -> dict:
        """Generate performance report"""
        report = {
            "session_id": self.session_id,
            "total_operations": len(self.performance_metrics),
            "performance_metrics": {}
        }
        
        for operation, values in self.performance_metrics.items():
            if all(isinstance(v, (int, float)) for v in values):
                # Numeric performance metrics (durations)
                report["performance_metrics"][operation] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
            else:
                # Timestamp-based metrics
                report["performance_metrics"][operation] = {
                    "count": len(values),
                    "first": values[0].isoformat() if hasattr(values[0], 'isoformat') else str(values[0]),
                    "last": values[-1].isoformat() if hasattr(values[-1], 'isoformat') else str(values[-1])
                }
                
        return report

# Global logger instance
maitri_logger = MAITRILogger()

def get_maitri_logger() -> MAITRILogger:
    """Get the global MAITRI logger instance"""
    return maitri_logger