import json
import os
import time
import threading
from collections import Counter
from datetime import datetime

try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("Warning: firebase_admin module not found. Only local JSON processing will work.")

# Gemini import removed as requested to avoid rate limiting
GEMINI_AVAILABLE = False

class TrafficProcessor:
    def __init__(self, cred_path=None, db_url=None, gemini_api_key=None):
        self.db_ref = None
        self.gemini_client = None
        self.suggestion_cache = {}  # Cache to avoid repeated API calls
        self._last_process_time = 0  # For debouncing Firebase listener
        self._debounce_seconds = 30  # Minimum seconds between processing
        
        # Initialize Firebase
        if cred_path and db_url:
            if not FIREBASE_AVAILABLE:
                raise ImportError("firebase_admin not installed. Run: pip install firebase-admin")
            try:
                if not firebase_admin._apps:
                    cred = credentials.Certificate(cred_path)
                    firebase_admin.initialize_app(cred, {'databaseURL': db_url})
                self.db_ref = db.reference()
            except Exception as e:
                print(f"Warning: Failed to initialize Firebase: {e}")
        
        # Initialize Gemini - Disabled
        # api_key = gemini_api_key or os.environ.get('GEMINI_API_KEY')
        # if api_key and GEMINI_AVAILABLE:
        #     try:
        #         self.gemini_client = genai.Client(api_key=api_key)
        #         print("Gemini AI initialized successfully")
        #     except Exception as e:
        #         print(f"Warning: Failed to initialize Gemini: {e}")
    
    def start_listener(self):
        """Start listening for changes to traffic_data and auto-update processed_data"""
        if not self.db_ref:
            print("Warning: Firebase not initialized. Cannot start listener.")
            return
        
        def on_data_change(event):
            """Callback when traffic_data changes"""
            try:
                # event.data contains the data at the path
                # For 'put' events, we need to fetch the full data
                if event.event_type in ('put', 'patch'):
                    current_time = time.time()
                    time_since_last = current_time - self._last_process_time
                    
                    # Debounce: skip if processed recently
                    if time_since_last < self._debounce_seconds:
                        print(f"[Firebase Listener] Debounce: skipping (wait {self._debounce_seconds - time_since_last:.0f}s)")
                        return
                    
                    print(f"[Firebase Listener] Data changed at {event.path}")
                    self._last_process_time = current_time
                    
                    # Fetch fresh data from traffic_data
                    raw_data = self.db_ref.child("traffic_data").get()
                    
                    if raw_data:
                        # Process the data
                        processed = self.process_data(raw_data)
                        
                        # Save to processed_data
                        self.db_ref.child("processed_data").set(processed)
                        print(f"[Firebase Listener] Updated processed_data: {processed.get('vehicleCount', 0)} vehicles")
                    else:
                        print("[Firebase Listener] No data to process")
            except Exception as e:
                print(f"[Firebase Listener] Error processing data: {e}")
        
        # Start listening to traffic_data path
        print("[Firebase Listener] Starting listener on traffic_data...")
        self.db_ref.child("traffic_data").listen(on_data_change)
    
    def generate_suggestions(self, reason):
        """Generate static suggestions for traffic congestion (AI disabled)"""
        # Default fallback suggestions
        default_suggestions = [
            "Monitor the situation closely",
            "Consider alternative routes",
            "Check traffic updates regularly"
        ]
        
        print(f"[Processor] Using static suggestions for: {reason[:30]}...")
        return default_suggestions

    def get_data_from_firebase(self, path="/"):
        if not self.db_ref:
            raise ValueError("Firebase not initialized. Provide cred_path and db_url.")
        return self.db_ref.child(path).get()

    def get_data_from_json(self, json_str):
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to fix common user input error where they use [] for dicts
            # This is a basic repair attempt for the specific sample pattern
            print(f"JSON Parse Error: {e}. Attempting loose parsing...")
            try:
                # Replace incorrect array brackets with dict braces if they look like dicts
                fixed_str = json_str.replace('": [', '": {').replace(']"', '}"')
                if fixed_str.strip().endswith(']'):
                    fixed_str = fixed_str.strip()[:-1] + '}'
                return json.loads(fixed_str)
            except:
                raise ValueError(f"Invalid JSON format: {e}")

    def process_data(self, data):
        """
        Processes raw Firebase data into the target API format.
        Expected Input Format: {"DATE": {"PUSH_ID": {...data...}}}
        """
        all_records = []
        print("process_data")
        # Normalize input to a flat list of records
        if isinstance(data, dict):
            for date_key, valid_data in data.items():
                if isinstance(valid_data, dict):
                    # Standard Firebase: {"id": {data}, "id2": {data}}
                    for record in valid_data.values():
                        if isinstance(record, dict):
                            # Use date_key if record doesn't have its own date
                            if 'date' not in record:
                                record['date'] = date_key
                            all_records.append(record)
                elif isinstance(valid_data, list):
                    # Array-like: [{"id": {data}}, ...] or [{data}, {data}]
                    for item in valid_data:
                        if isinstance(item, dict):
                            # Check if it's a wrapper {"id": {data}} or just {data}
                            # A simple heuristic: check for known keys like 'vehicle_count'
                            if 'vehicle_count' in item:
                                if 'date' not in item:
                                    item['date'] = date_key
                                all_records.append(item)
                            else:
                                for sub_item in item.values():
                                     if isinstance(sub_item, dict):
                                         if 'date' not in sub_item:
                                             sub_item['date'] = date_key
                                         all_records.append(sub_item)

        if not all_records:
            return {
                "vehicleCount": 0,
                "time": "N/A",
                "congestion": [],
                "report": [],
                "graph_data": []
            }

        # Aggregation
        total_vehicles = 0
        reasons = []
        reports_map = {} # deduplicate by reason
        timestamps = []

        for record in all_records:
            # Vehicle Count
            v_count = record.get('vehicle_count', 0)
            if isinstance(v_count, (int, float)):
                 total_vehicles += int(v_count)
            
            # Reasons for congestion stats - use 'reason' field from Firebase
            # This will be the name displayed in the pie chart
            reason = record.get('reason', 'Unknown')
            
            # Clean up reason for better display (truncate very long text)
            if len(reason) > 50:
                reason = reason[:47] + "..."
            
            # Use reason directly as the category for pie chart name
            category = reason
            reasons.append(category)

            # Report generation - use Gemini to generate suggestions
            if category not in reports_map:
                # Generate 3 AI-powered suggestions based on the reason
                suggestions = self.generate_suggestions(category)
                reports_map[category] = {
                    "reason": category,
                    "suggestions": suggestions  # Now an array of 3 suggestions
                }

            # Graph Data - include date and congestion reason
            ts = record.get('timestamp')
            if ts:
                timestamps.append({
                    't': ts,
                    'v': v_count,
                    'date': record.get('date', 'Unknown'),
                    'reason': category
                })

        # Calculate Percentages
        congestion_stats = []
        reason_counts = Counter(reasons)
        total_reasons = sum(reason_counts.values())
        
        for r, count in reason_counts.items():
            percentage = round((count / total_reasons) * 100, 1) if total_reasons > 0 else 0
            congestion_stats.append({
                "name": r,
                "percentage": percentage
            })

        # Format Time (Latest) - use 12-hour format like "3:30"
        timestamps.sort(key=lambda x: x['t'])
        latest_time_str = "N/A"
        if timestamps:
            latest_ts = timestamps[-1]['t']
            dt_obj = datetime.fromtimestamp(latest_ts)
            # Use %-H for 12-hour without leading zero on Windows use %#H
            hour = dt_obj.hour % 12 or 12  # Convert to 12-hour format
            minute = dt_obj.strftime("%M")
            latest_time_str = f"{hour}:{minute}"

        # Graph Data Formatting - use percentage from congestion stats
        # Create a lookup for reason percentages
        reason_percentages = {c['name']: c['percentage'] for c in congestion_stats}
        
        graph_data = []
         
        for point in timestamps:
            dt = datetime.fromtimestamp(point['t'])
            # Use 24-hour format to ensure unique sorting and correct AM/PM distinction
            time_label = dt.strftime('%H:%M:%S')
            # Combine time and date into single timestamp field
            timestamp_str = f"{time_label},{point['date']}"
            graph_data.append({
                "timestamp": timestamp_str,
                "percentage": reason_percentages.get(point['reason'], 0),
                "reason": point['reason']
            })

        # Filter for Last 10 Days
        # Extract all unique dates and sort them
        unique_dates = sorted(list(set(r.get('date', 'Unknown') for r in all_records)), 
                              key=lambda d: datetime.strptime(d, "%d-%m-%Y") if d != 'Unknown' else datetime.min)
        
        # Keep only the last 10 dates
        last_10_dates = unique_dates[-10:] if unique_dates else []
        
        # Filter records to only include those from the last 10 dates
        recent_records = [r for r in all_records if r.get('date') in last_10_dates]

        # Detailed History (Flat list for "All Events" view)
        detailed_history = []
        for r in recent_records:
            detailed_history.append({
                "date": r.get('date', 'Unknown'),
                "time": r.get('time', 'N/A'),
                "vehicleCount": int(r.get('vehicle_count', 0)),
                "reason": r.get('reason', 'Unknown'),
                "congestion_status": r.get('status', 'Unknown'),
                "timestamp": r.get('timestamp', 0)
            })
        
        # Sort detailed history by timestamp descending (newest first)
        detailed_history.sort(key=lambda x: x['timestamp'], reverse=True)

        # Daily Summary Calculation (aggregated from recent records)
        daily_stats = {}
        for record in recent_records:
            date_key = record.get('date', 'Unknown')
            if date_key not in daily_stats:
                daily_stats[date_key] = {
                    "date": date_key,
                    "totalEvents": 0,
                    "peakVehicleCount": 0
                }
            
            daily_stats[date_key]["totalEvents"] += 1
            v_count = int(record.get('vehicle_count', 0))
            if v_count > daily_stats[date_key]["peakVehicleCount"]:
                daily_stats[date_key]["peakVehicleCount"] = v_count
        
        daily_summary = list(daily_stats.values())
        # Sort daily summary by date descending (newest dates first for the list view)
        daily_summary.sort(key=lambda x: datetime.strptime(x['date'], "%d-%m-%Y") if x['date'] != 'Unknown' else datetime.min, reverse=True)

        return {
            "vehicleCount": total_vehicles,
            "time": latest_time_str,
            "congestion": congestion_stats,
            "report": list(reports_map.values()),
            "graph": graph_data,
            "daily_summary": daily_summary,
            "detailed_history": detailed_history
        }

if __name__ == "__main__":
    # Test with the sample provided (cleaned up)
    sample_json = """
    {
      "01-08-2025": {
        "-0WaCEPkqzD7SO4TLVzM": {
          "date": "01-08-2025",
          "reason": "Observed: Illegal Parking",
          "status": "Wrong Parking",
          "suggestion": "Remove parked vehicles",
          "time": "20:46",
          "timestamp": 1754061402.5369713,
          "vehicle_count": 1
        },
        "-0WaCEPkqzD7SO4TLVzN": {
          "date": "01-08-2025",
          "reason": "Signal malfunction",
          "status": "Signal Delay",
          "suggestion": "Optimize signal",
          "time": "20:47",
          "timestamp": 1754061462.5369713,
          "vehicle_count": 5
        }
      }
    }
    """
    processor = TrafficProcessor()
    data = processor.get_data_from_json(sample_json)
    result = processor.process_data(data)
    print(json.dumps(result, indent=2))
