import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class AuditLogger:
    def __init__(self, log_file: str = "audit_logs.json"):
        self.log_file = log_file
        self.ensure_log_file_exists()
    
    def ensure_log_file_exists(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)
    
    def log_action(self, 
                   action_type: str, 
                   summary: str, 
                   user_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Log an action with timestamp and details"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action_type": action_type,
            "summary": summary,
            "user_id": user_id or "anonymous",
            "metadata": metadata or {}
        }
        
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            logs.append(log_entry)
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            return log_entry
        except Exception as e:
            print(f"Error logging action: {e}")
            return log_entry
    
    def get_logs(self, 
                 limit: Optional[int] = None, 
                 action_type: Optional[str] = None,
                 user_id: Optional[str] = None) -> list:
        """Retrieve logs with optional filtering"""
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            # Apply filters
            if action_type:
                logs = [log for log in logs if log["action_type"] == action_type]
            if user_id:
                logs = [log for log in logs if log["user_id"] == user_id]
            
            # Sort by timestamp descending
            logs.sort(key=lambda x: x["timestamp"], reverse=True)
            
            if limit:
                logs = logs[:limit]
            
            return logs
        except Exception as e:
            print(f"Error retrieving logs: {e}")
            return []

# Create a singleton instance
audit_logger = AuditLogger() 