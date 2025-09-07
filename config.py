import os
from datetime import timedelta

class Config:
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'iph-forecasting-secret-key-2024'
    DEBUG = True
    
    # File Upload Configuration
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
    
    # Data Storage Configuration
    DATA_FOLDER = 'data'
    HISTORICAL_DATA_PATH = 'data/historical_data.csv'
    MODELS_PATH = 'data/models/'
    BACKUPS_PATH = 'data/backups/'
    
    # Model Configuration
    FORECAST_MIN_WEEKS = 4
    FORECAST_MAX_WEEKS = 12
    DEFAULT_FORECAST_WEEKS = 8
    
    # Performance Configuration
    MODEL_PERFORMANCE_THRESHOLD = 0.1  # 10% improvement threshold
    AUTO_RETRAIN_THRESHOLD = 50  # Retrain when new data > 50 records
    
    # Dashboard Configuration
    CHART_HEIGHT = 500
    COMPARISON_CHART_HEIGHT = 400
    MAX_HISTORICAL_DISPLAY = 60  # Show last 60 periods in chart
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        # Create necessary directories
        directories = [
            Config.UPLOAD_FOLDER,
            Config.DATA_FOLDER,
            Config.MODELS_PATH,
            Config.BACKUPS_PATH
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print("✅ Application directories initialized")

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production-secret-key'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}