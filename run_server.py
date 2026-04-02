
import os
import sys

# Set environment variables
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

from app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=False)
