#!/usr/bin/env python3
"""
IPH Forecasting Dashboard Application Runner
"""

import os
import sys
from app import app
from config import Config

def main():
    """Main application runner"""
    
    print("🚀 Starting IPH Forecasting Dashboard...")
    print("=" * 60)
    
    # Initialize configuration
    Config.init_app(app)
    
    # Display startup information
    print(f"📊 Dashboard URL: http://localhost:5000")
    print(f"📁 Data directory: {Config.DATA_FOLDER}")
    print(f"🤖 Models directory: {Config.MODELS_PATH}")
    print(f"💾 Backups directory: {Config.BACKUPS_PATH}")
    print("=" * 60)
    
    # Check if this is first run
    historical_data_exists = os.path.exists(Config.HISTORICAL_DATA_PATH)
    
    if not historical_data_exists:
        print("ℹ️  FIRST RUN DETECTED")
        print("📋 To get started:")
        print("   1. Go to http://localhost:5000")
        print("   2. Click 'Upload Data' to upload your CSV file")
        print("   3. The system will automatically train models and generate forecasts")
        print("=" * 60)
    else:
        print("✅ Historical data found - Dashboard ready!")
        print("=" * 60)
    
    # Run the application
    try:
        app.run(
            debug=Config.DEBUG,
            host='0.0.0.0',
            port=5000,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Application error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()