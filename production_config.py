# Production configuration for Digital Ocean deployment
import os

class ProductionConfig:
    """Production configuration for OMR Scraper"""
    
    # Flask Configuration
    DEBUG = False
    HOST = '0.0.0.0'  # Allow external connections
    PORT = int(os.environ.get('PORT', 8002))  # Use environment variable or default to 8002
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
    
    # File Upload Configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = '/tmp/omr_uploads'
    
    # Cleanup Configuration
    CLEANUP_INTERVAL_HOURS = 24
    CLEANUP_TIME = '02:00'  # 2:00 AM daily cleanup
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = '/var/log/omr_scraper.log'
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
    
    @staticmethod
    def get_config():
        """Get production configuration"""
        return {
            'debug': ProductionConfig.DEBUG,
            'host': ProductionConfig.HOST,
            'port': ProductionConfig.PORT,
            'secret_key': ProductionConfig.SECRET_KEY,
            'max_content_length': ProductionConfig.MAX_CONTENT_LENGTH,
            'upload_folder': ProductionConfig.UPLOAD_FOLDER,
            'cleanup_interval_hours': ProductionConfig.CLEANUP_INTERVAL_HOURS,
            'cleanup_time': ProductionConfig.CLEANUP_TIME,
            'log_level': ProductionConfig.LOG_LEVEL,
            'log_file': ProductionConfig.LOG_FILE,
            'allowed_extensions': ProductionConfig.ALLOWED_EXTENSIONS
        }
