from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from traffic_processor import TrafficProcessor
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Processor with Firebase and Gemini credentials
# Use the same credentials as main.py
CRED_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'firebase_key.json')
DB_URL = 'https://traffic-analyser-fad30-default-rtdb.firebaseio.com/'
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')  # Set this environment variable with your Gemini API key
processor = TrafficProcessor(cred_path=CRED_PATH, db_url=DB_URL, gemini_api_key=GEMINI_API_KEY)

# Start Firebase listener for auto-update
processor.start_listener()

# Sample data for demonstration if no Firebase connection
SAMPLE_DATA = """
{
  "06-01-2026": {
    "-0WaCEPkqzD7SO4TLVzM": {
      "date": "06-01-2026",
      "time": "14:30",
      "timestamp": 1736155200.5369713,
      "status": "Congestion",
      "reason": "vehicles moving slowly",
      "suggestion": "Check traffic lights or road conditions",
      "vehicle_count": 12
    },
    "-0WaCEPkqzD7SO4TLVzN": {
      "date": "06-01-2026",
      "time": "14:35",
      "timestamp": 1736155500.5369713,
      "status": "Congestion",
      "reason": "Heavy traffic detected",
      "suggestion": "Check traffic lights or road conditions",
      "vehicle_count": 8
    },
    "-0WaCEPkqzD7SO4TLVzO": {
      "date": "06-01-2026",
      "time": "14:40",
      "timestamp": 1736155800.5369713,
      "status": "Congestion",
      "reason": "vehicles moving slowly",
      "suggestion": "Check traffic lights or road conditions",
      "vehicle_count": 15
    },
    "-0WaCEPkqzD7SO4TLVzP": {
      "date": "06-01-2026",
      "time": "14:45",
      "timestamp": 1736156100.5369713,
      "status": "Congestion",
      "reason": "Road construction ahead",
      "suggestion": "Check traffic lights or road conditions",
      "vehicle_count": 5
    }
  }
}
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/traffic-data')
def get_traffic_data():
    # If we have firebase initialized, use it
    if processor.db_ref:
        try:
            # Fetch from traffic_data path where main.py pushes congestion data
            raw_data = processor.get_data_from_firebase("traffic_data")
            data = processor.process_data(raw_data)
            
            # Save processed data to Firebase for frontend to read directly
            try:
                processor.db_ref.child("processed_data").set(data)
            except Exception as e:
                print(f"Warning: Could not save processed data to Firebase: {e}")
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        # Fallback to sample data for demo purposes
        # Or allow passing JSON via query param for testing?
        # Let's just use the hardcoded sample for a "demo mode"
        raw_data = processor.get_data_from_json(SAMPLE_DATA)
        data = processor.process_data(raw_data)
        
    return jsonify(data)

@app.route('/api/congestion-status')
def get_congestion_status():
    """Get current congestion status from Firebase"""
    if processor.db_ref:
        try:
            is_congestion = processor.get_data_from_firebase("isCongestion")
            return jsonify({"isCongestion": is_congestion or False})
        except Exception as e:
            return jsonify({"error": str(e), "isCongestion": False}), 500
    return jsonify({"isCongestion": False})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
