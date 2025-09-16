import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app
from app import app

# This is the WSGI handler for Vercel
def handler(environ, start_response):
    return app(environ, start_response)

# For Vercel serverless functions
application = app

# Make sure this works for direct execution too
if __name__ == '__main__':
    app.run(debug=False)