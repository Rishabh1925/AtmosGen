import logging
import sys
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    file_handler = logging.FileHandler('atmosgen.log')
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler]
    )
    
    # Set specific loggers
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.INFO)