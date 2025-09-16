import os
from datetime import timedelta

class Config:
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'iph-forecasting-secret-key-2024-railway'
    DEBUG = False
    
    # File Upload Configuration - Use /tmp for Railway
    UPLOAD_FOLDER = '/tmp/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
    
    # Data Storage Configuration - Use /tmp for Railway (ephemeral storage)
    DATA_FOLDER = '/tmp/data'
    HISTORICAL_DATA_PATH = '/tmp/data/historical_data.csv'
    MODELS_PATH = '/tmp/data/models/'
    BACKUPS_PATH = '/tmp/data/backups/'
    
    # Model Configuration
    FORECAST_MIN_WEEKS = 4
    FORECAST_MAX_WEEKS = 12
    DEFAULT_FORECAST_WEEKS = 8
    
    # Performance Configuration
    MODEL_PERFORMANCE_THRESHOLD = 0.1
    AUTO_RETRAIN_THRESHOLD = 50
    
    # Dashboard Configuration
    CHART_HEIGHT = 500
    COMPARISON_CHART_HEIGHT = 400
    MAX_HISTORICAL_DISPLAY = 60
    
    # Railway specific
    PORT = int(os.environ.get('PORT', 5000))
    
    @staticmethod
    def init_app(app):
        """Initialize application with config for Railway"""
        directories = [
            Config.UPLOAD_FOLDER,
            Config.DATA_FOLDER,
            Config.MODELS_PATH,
            Config.BACKUPS_PATH
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create directory {directory}: {e}")
        
        print("✅ Application directories initialized for Railway deployment")

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production-secret-key-change-this-in-railway'

# Railway akan menggunakan production config
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': ProductionConfig
}