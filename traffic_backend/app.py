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

# Start AI request listener for Firebase-based AI chat
processor.start_ai_request_listener(GEMINI_API_KEY)

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

@app.route('/api/test-gemini', methods=['GET'])
def test_gemini():
    """Test if Gemini API key is valid"""
    try:
        if not GEMINI_API_KEY:
            return jsonify({"success": False, "error": "GEMINI_API_KEY not set in environment"}), 500
        
        # Check if key looks valid (basic format check)
        if len(GEMINI_API_KEY) < 10:
            return jsonify({"success": False, "error": "API key too short - may be invalid"}), 500
        
        # Try to use Gemini with a simple test
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        print("[TEST] Calling Gemini API with simple test...")
        response = model.generate_content("Say 'Hello Traffic-chan!' in one word")
        print(f"[TEST] Response: {response.text}")
        
        return jsonify({
            "success": True,
            "message": "Gemini API key is valid!",
            "test_response": response.text[:100] if response.text else "No response",
            "api_key_prefix": GEMINI_API_KEY[:8] + "..." if GEMINI_API_KEY else "None"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/ai-report', methods=['POST'])
def generate_ai_report():
    """Generate an AI-powered HTML report using Gemini from all traffic data and save to Firebase"""
    try:
        # Get user query from request
        data = request.get_json()
        user_query = data.get('query', 'Generate a comprehensive traffic analysis report')
        
        # Fetch all traffic data from Firebase
        if not processor.db_ref:
            return jsonify({"error": "Database not connected"}), 500
            
        raw_data = processor.get_data_from_firebase("traffic_data")
        processed_data = processor.process_data(raw_data)
        
        # Prepare data summary for AI
        data_summary = {
            "total_events": len(processed_data.get("graph", [])),
            "congestion_breakdown": processed_data.get("congestion", []),
            "recent_reports": processed_data.get("report", [])[:10],
            "daily_summary": processed_data.get("daily_summary", []),
            "stats": processed_data.get("stats", {}),
            "detailed_history": processed_data.get("detailed_history", [])[:20]
        }
        
        # Use Gemini to generate HTML report
        if not GEMINI_API_KEY:
            return jsonify({"error": "Gemini API key not configured"}), 500
            
        import google.generativeai as genai
        from datetime import datetime
        import time
        
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""You are Traffic-chan, a cute kawaii AI traffic analyst assistant! 🌸
        
Based on the following traffic data, generate a beautiful HTML report that answers: "{user_query}"

TRAFFIC DATA:
{json.dumps(data_summary, indent=2)}

REQUIREMENTS:
1. Generate ONLY valid HTML content (no markdown, no code blocks)
2. Use inline CSS styles with a kawaii pastel theme (pinks, purples, soft colors)
3. Include cute emojis throughout 🚗💖✨🌸
4. Structure the report with:
   - A cute header with title
   - Key statistics in colorful cards
   - Analysis summary
   - Recommendations if applicable
5. Use these colors: pink (#f9a8d4), purple (#c4b5fd), mint (#a7f3d0), cream (#fde68a)
6. Make it visually appealing with rounded corners, shadows, and gradients
7. Include simple ASCII/emoji-based visualizations if showing data trends
8. Keep responses focused and informative while being cute
9. Sign off as "Traffic-chan 🌸"

Generate the HTML now:"""

        response = model.generate_content(prompt)
        html_content = response.text
        
        # Clean up response if it has markdown code blocks
        if html_content.startswith("```html"):
            html_content = html_content[7:]
        if html_content.startswith("```"):
            html_content = html_content[3:]
        if html_content.endswith("```"):
            html_content = html_content[:-3]
        html_content = html_content.strip()
        
        # Save to Firebase
        timestamp = time.time()
        date_str = datetime.now().strftime("%d-%m-%Y")
        time_str = datetime.now().strftime("%H:%M:%S")
        
        report_data = {
            "query": user_query,
            "html": html_content,
            "timestamp": timestamp,
            "date": date_str,
            "time": time_str,
            "created_at": datetime.now().isoformat()
        }
        
        # Push to Firebase and get the report ID
        new_report_ref = processor.db_ref.child("ai_reports").push(report_data)
        report_id = new_report_ref.key
        
        # Also save as latest report for quick access
        processor.db_ref.child("latest_ai_report").set({
            "report_id": report_id,
            **report_data
        })
        
        return jsonify({
            "success": True,
            "html": html_content,
            "report_id": report_id,
            "query": user_query,
            "saved_to_firebase": True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/ai-report/<report_id>', methods=['GET'])
def get_ai_report(report_id):
    """Get a specific AI report from Firebase by ID"""
    try:
        if not processor.db_ref:
            return jsonify({"error": "Database not connected"}), 500
            
        report_data = processor.db_ref.child("ai_reports").child(report_id).get()
        
        if report_data:
            return jsonify({
                "success": True,
                "report_id": report_id,
                **report_data
            })
        else:
            return jsonify({"error": "Report not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ai-reports', methods=['GET'])
def get_all_ai_reports():
    """Get all AI reports from Firebase (list view)"""
    try:
        if not processor.db_ref:
            return jsonify({"error": "Database not connected"}), 500
            
        reports_data = processor.db_ref.child("ai_reports").get()
        
        if reports_data:
            # Convert to list and sort by timestamp (newest first)
            reports_list = []
            for report_id, report in reports_data.items():
                reports_list.append({
                    "report_id": report_id,
                    "query": report.get("query", ""),
                    "date": report.get("date", ""),
                    "time": report.get("time", ""),
                    "timestamp": report.get("timestamp", 0)
                })
            reports_list.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            
            return jsonify({
                "success": True,
                "reports": reports_list[:20]  # Return last 20 reports
            })
        else:
            return jsonify({"success": True, "reports": []})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/latest-ai-report', methods=['GET'])
def get_latest_ai_report():
    """Get the latest AI report from Firebase"""
    try:
        if not processor.db_ref:
            return jsonify({"error": "Database not connected"}), 500
            
        latest_report = processor.db_ref.child("latest_ai_report").get()
        
        if latest_report:
            return jsonify({
                "success": True,
                **latest_report
            })
        else:
            return jsonify({"success": False, "message": "No reports found"})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
