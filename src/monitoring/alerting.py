
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertManager:
    """
    Manages alerts for the monitoring system.
    Supports logging to console and file, and can be extended for Email/Slack.
    """
    
    def __init__(self, log_file: str = "monitoring_alerts.log"):
        self.log_file = log_file
        
    def send_alert(self, title: str, message: str, level: str = "WARNING", details: Optional[Dict[str, Any]] = None):
        """
        Send an alert through configured channels.
        
        Args:
            title: Short summary of the alert.
            message: Detailed description.
            level: Severity (INFO, WARNING, ERROR, CRITICAL).
            details: Optional dictionary with more context (e.g., drift scores).
        """
        timestamp = datetime.now().isoformat()
        full_message = f"[{level}] {title}: {message}"
        
        if details:
            full_message += f"\nDetails: {details}"
            
        # 1. Console Alert
        self._log_to_console(full_message, level)
        
        # 2. File Alert
        self._log_to_file(timestamp, full_message)
        
        # 3. External integrations (Placeholders)
        # self._send_email(title, message)
        # self._send_slack(title, message)
        
    def _log_to_console(self, message: str, level: str):
        if level == "CRITICAL":
            logger.critical(message)
        elif level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
            
    def _log_to_file(self, timestamp: str, message: str):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} - {message}\n")
                f.write("-" * 50 + "\n")
        except Exception as e:
            logger.error(f"Failed to write to alert log file: {e}")

    # Future implementations
    def _send_email(self, title, message):
        pass
        
    def _send_slack(self, title, message):
        pass
