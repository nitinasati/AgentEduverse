import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors and structured output"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'agent': getattr(record, 'agent', 'unknown'),
            'session_id': getattr(record, 'session_id', 'unknown')
        }
        
        if hasattr(record, 'extra_data'):
            log_entry['extra_data'] = record.extra_data
            
        return json.dumps(log_entry)

class AgentEduverseLogger:
    """Centralized logger for the AgentEduverse system"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentEduverseLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_logger()
            self._initialized = True
    
    def _setup_logger(self):
        """Setup the centralized logger"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Get configuration from environment
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        log_file = os.getenv('LOG_FILE', 'logs/agent_eduverse.log')
        
        # Create logger
        self.logger = logging.getLogger('AgentEduverse')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with colors - Set to DEBUG level to show all logs
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_format = "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
        console_handler.setFormatter(CustomFormatter(console_format))
        console_handler.setStream(sys.stdout)  # Force stdout
        self.logger.addHandler(console_handler)
        
        # Ensure propagation is enabled
        self.logger.propagate = True
        
        # File handler with structured logging
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)
        
        # Log system startup
        self.logger.info("AgentEduverse System Logger initialized", extra={
            'agent': 'system',
            'session_id': 'startup',
            'extra_data': {
                'log_level': log_level,
                'log_file': log_file
            }
        })
    
    def get_logger(self, agent_name: str = None):
        """Get a logger instance with agent context"""
        if agent_name:
            return AgentLogger(self.logger, agent_name)
        return self.logger
    
    def log_agent_event(self, agent_name: str, event_type: str, message: str, extra_data: dict = None):
        """Log agent-specific events"""
        self.logger.info(f"[{agent_name}] {event_type}: {message}", extra={
            'agent': agent_name,
            'session_id': extra_data.get('session_id', 'unknown') if extra_data else 'unknown',
            'extra_data': extra_data or {}
        })
    
    def log_agent_error(self, agent_name: str, error: Exception, context: str = None):
        """Log agent errors"""
        self.logger.error(f"[{agent_name}] Error in {context}: {str(error)}", extra={
            'agent': agent_name,
            'session_id': 'error',
            'extra_data': {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context
            }
        })

class AgentLogger:
    """Agent-specific logger wrapper"""
    
    def __init__(self, base_logger, agent_name: str):
        self.base_logger = base_logger
        self.agent_name = agent_name
    
    def info(self, message: str, extra_data: dict = None):
        """Log info message with agent context"""
        self.base_logger.info(message, extra={
            'agent': self.agent_name,
            'session_id': extra_data.get('session_id', 'unknown') if extra_data else 'unknown',
            'extra_data': extra_data or {}
        })
    
    def warning(self, message: str, extra_data: dict = None):
        """Log warning message with agent context"""
        self.base_logger.warning(message, extra={
            'agent': self.agent_name,
            'session_id': extra_data.get('session_id', 'unknown') if extra_data else 'unknown',
            'extra_data': extra_data or {}
        })
    
    def error(self, message: str, extra_data: dict = None):
        """Log error message with agent context"""
        self.base_logger.error(message, extra={
            'agent': self.agent_name,
            'session_id': extra_data.get('session_id', 'unknown') if extra_data else 'unknown',
            'extra_data': extra_data or {}
        })
    
    def debug(self, message: str, extra_data: dict = None):
        """Log debug message with agent context"""
        self.base_logger.debug(message, extra={
            'agent': self.agent_name,
            'session_id': extra_data.get('session_id', 'unknown') if extra_data else 'unknown',
            'extra_data': extra_data or {}
        })

# Global logger instance
logger = AgentEduverseLogger()
