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
    
    def start_listener(self):
        """Start listening for changes to traffic_data and auto-update processed_data"""
        if not self.db_ref:
            print("Warning: Firebase not initialized. Cannot start listener.")
            return
        
        def on_data_change(event):
            """Callback when traffic_data changes"""
            try:
                if event.event_type in ('put', 'patch'):
                    # Only process if there's actual new data
                    if not event.data:
                        return
                    
                    # Check if data actually changed using hash
                    import hashlib
                    import json
                    data_str = json.dumps(event.data, sort_keys=True)
                    data_hash = hashlib.md5(data_str.encode()).hexdigest()
                    
                    if hasattr(self, '_last_data_hash') and self._last_data_hash == data_hash:
                        # Data hasn't actually changed, skip
                        return
                    
                    self._last_data_hash = data_hash
                    
                    print(f"[Firebase Listener] New data detected at {event.path}")
                    
                    raw_data = self.db_ref.child("traffic_data").get()
                    
                    if raw_data:
                        processed = self.process_data(raw_data)
                        self.db_ref.child("processed_data").set(processed)
                        print(f"[Firebase Listener] Updated processed_data: {processed.get('vehicleCount', 0)} vehicles")
                    else:
                        print("[Firebase Listener] No data to process")
            except Exception as e:
                print(f"[Firebase Listener] Error processing data: {e}")
        
        print("[Firebase Listener] Starting listener on traffic_data...")
        self.db_ref.child("traffic_data").listen(on_data_change)
    
    def start_ai_request_listener(self, gemini_api_key):
        """Start listening for AI report requests in Firebase"""
        if not self.db_ref:
            print("Warning: Firebase not initialized. Cannot start AI request listener.")
            return
        
        import google.generativeai as genai
        # Ollama Configuration
        ollama_model = "qwen2.5-coder:3b"
        ollama_url = "http://localhost:11434/api/generate"
        
        print(f"[AI Listener] Using local Ollama model: {ollama_model}")
        
        # Request queue for batching
        self.request_queue = []
        self.queue_lock = threading.Lock()
        
        def process_queue_worker():
            """Background worker to process queued requests in batches"""
            while True:
                time.sleep(0.3)  # Process every 0.3 seconds (fast response)
                
                batch_to_process = []
                with self.queue_lock:
                    if not self.request_queue:
                        continue
                    
                    # Take all pending requests
                    batch_to_process = self.request_queue[:]
                    self.request_queue = []
                
                if batch_to_process:
                    print(f"[AI Batch] Processing batch of {len(batch_to_process)} requests via Ollama")
                    self._process_batch_ai_requests(batch_to_process, ollama_model)

        # Start background worker
        threading.Thread(target=process_queue_worker, daemon=True).start()
        
        def on_ai_request(event):
            """Callback when ai_requests changes"""
            try:
                if event.event_type == 'put' and event.data:
                    request_id = event.path.strip('/')
                    
                    # Handle initial load - path is empty or '/'
                    if not request_id or request_id == '':
                        # Initial load - iterate through all requests
                        if isinstance(event.data, dict):
                            with self.queue_lock:
                                for req_id, req_data in event.data.items():
                                    if isinstance(req_data, dict) and req_data.get('status') == 'pending':
                                        print(f"[AI Request] Queuing pending request on startup: {req_id}")
                                        self.request_queue.append((req_id, req_data))
                        return
                    
                    request_data = event.data
                    if not isinstance(request_data, dict):
                        return
                    
                    # Check if already processed
                    if request_data.get('status') != 'pending':
                        return
                    
                    # Queue the request
                    print(f"[AI Request] Queuing new request: {request_id}")
                    with self.queue_lock:
                        self.request_queue.append((request_id, request_data))
                        
            except Exception as e:
                print(f"[AI Request Listener] Error: {e}")
        
        print("[Firebase Listener] Starting listener on ai_requests...")
        self.db_ref.child("ai_requests").listen(on_ai_request)

    def _process_batch_ai_requests(self, requests_batch, model_name):
        """Process a batch of AI requests in a single API call using Ollama"""
        try:
            from datetime import datetime
            import time
            import json
            import urllib.request
            
            # Fetch traffic data once for the batch
            try:
                raw_data = self.db_ref.child("traffic_data").get()
                processed_data = self.process_data(raw_data) if raw_data else {}
            except Exception as e:
                print(f"[AI Batch] Error fetching traffic data: {e}")
                processed_data = {}
            
            # Prepare data summary
            data_summary = {
                "total_events": len(processed_data.get("graph", [])),
                "congestion_breakdown": processed_data.get("congestion", []),
                "stats": processed_data.get("stats", {}),
                "detailed_history": processed_data.get("detailed_history", [])[:5]
            }
            
            # Construct Batch Prompt
            batch_prompt = "You are Traffic-chan, a cute kawaii AI traffic analyst! 🌸\n\n"
            batch_prompt += f"TRAFFIC DATA SUMMARY:\n{json.dumps(data_summary, indent=2)}\n\n"
            batch_prompt += "TASKS:\n"
            
            for i, (req_id, req_data) in enumerate(requests_batch):
                query = req_data.get('query', 'Traffic report')
                batch_prompt += f"Request {i+1} (ID: {req_id}): {query}\n"
                
                # Mark as processing
                self.db_ref.child("ai_requests").child(req_id).update({"status": "processing"})
            
            batch_prompt += """
REQUIREMENTS:
1. Return a JSON object where keys are the Request IDs and values are COMPLETE HTML pages.
2. Example format: { "req_id_1": "<div>...</div>" }

EACH HTML REPORT MUST INCLUDE:
=================================

A) HEADER SECTION:
   - Title with cute emoji (e.g., "🚗 Traffic Analysis Report 🌸")
   - Current date/time styled nicely

B) SUMMARY TEXT SECTION:
   - 2-3 paragraph analysis of the traffic data
   - Key insights written in friendly language

C) DATA TABLE:
   - Use <table> with borders and styling
   - Show congestion breakdown with columns: Type | Count | Percentage
   - Alternating row colors (pastel pink/white)
   - Example:
     <table style="width:100%; border-collapse:collapse; margin:20px 0;">
       <tr style="background:#f9a8d4;"><th>Type</th><th>Count</th><th>%</th></tr>
       <tr><td>Heavy Traffic</td><td>5</td><td>38%</td></tr>
     </table>

D) VISUAL BAR CHART (ASCII/Emoji based):
   - Use colored divs or emoji bars to show data visually
   - Example: "Heavy: ████████ 40%"
   - Use inline CSS for bar widths

E) STATISTICS CARDS:
   - 3-4 colorful stat boxes showing key metrics
   - Rounded corners, shadows, pastel colors

F) RECOMMENDATIONS:
   - Bullet list of 3-5 actionable suggestions

STYLING REQUIREMENTS:
- Use inline CSS only (no external stylesheets)
- Colors: pink (#f9a8d4), lavender (#c4b5fd), mint (#a7f3d0), cream (#fde68a)
- Font: Arial or system-ui
- Rounded corners (border-radius: 12px)
- Box shadows for depth
- Include cute emojis 🚗💖✨🌸🚦

OUTPUT FORMAT:
- Return ONLY valid JSON
- Keys = Request IDs, Values = HTML string
- NO markdown, NO code blocks in values
"""
            
            # Call Ollama API
            ollama_url = "http://localhost:11434/api/generate"
            print(f"[AI Batch] Sending batch to Ollama ({len(requests_batch)} requests)...")
            
            payload = {
                "model": model_name,
                "prompt": batch_prompt,
                "stream": False,
                "format": "json"  # Enforce JSON output if model supports it
            }
            
            req = urllib.request.Request(
                ollama_url, 
                data=json.dumps(payload).encode('utf-8'), 
                headers={'Content-Type': 'application/json'}
            )
            
            try:
                with urllib.request.urlopen(req) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    response_text = result.get('response', '')
                    print(f"[AI Batch] Ollama raw result: {result}")
                    print(f"[AI Batch] Ollama response text: {response_text[:500]}...")
            except Exception as e:
                print(f"[AI Batch] Ollama API Error: {e}")
                raise e
            
            # Clean up response logic (remove code blocks if any)
            response_text = response_text.strip()
            print(f"[AI Batch] Cleaned response: {response_text[:500]}...")
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            try:
                # Parse JSON response - may be {"id": "html"} or just {"html": "..."}
                results = json.loads(response_text)
                
                # Process results - use actual request_ids from batch
                timestamp = time.time()
                current_time_iso = datetime.now().isoformat()
                
                # Handle single request case (most common)
                if len(requests_batch) == 1:
                    req_id, req_data = requests_batch[0]
                    
                    # Get HTML from response - could be keyed by request_id or just first value
                    if req_id in results:
                        html_content = results[req_id]
                    elif isinstance(results, dict) and len(results) > 0:
                        # Take first value if request_id not found as key
                        html_content = list(results.values())[0]
                    else:
                        html_content = str(results)
                    
                    response_data = {
                        "request_id": req_id,  # Use actual request_id
                        "query": req_data.get('query', 'Traffic Analysis'),
                        "html": html_content,
                        "status": "completed",
                        "timestamp": timestamp,
                        "created_at": current_time_iso
                    }
                    
                    # Update Firebase
                    print(f"[AI Batch] Saving response for: {req_id}")
                    self.db_ref.child("latest_ai_report").set(response_data)
                    print(f"[AI Batch] Completed request: {req_id}")
                    
                    # Delete the result AND request after 30 seconds
                    def delete_after_delay(request_id):
                        time.sleep(30)
                        try:
                            self.db_ref.child("latest_ai_report").delete()
                            self.db_ref.child("ai_requests").child(request_id).delete()
                            print(f"[AI Batch] Deleted latest_ai_report and ai_request/{request_id} after 30 seconds")
                        except Exception as e:
                            print(f"[AI Batch] Error deleting: {e}")
                    
                    threading.Thread(target=delete_after_delay, args=(req_id,), daemon=True).start()
                else:
                    # Multiple requests - try to match by key
                    for req_id, req_data in requests_batch:
                        html_content = results.get(req_id, "")
                        if html_content:
                            response_data = {
                                "request_id": req_id,
                                "query": req_data.get('query', 'Traffic Analysis'),
                                "html": html_content,
                                "status": "completed",
                                "timestamp": timestamp,
                                "created_at": current_time_iso
                            }
                            self.db_ref.child("latest_ai_report").set(response_data)
                            self.db_ref.child("ai_requests").child(req_id).delete()
                            print(f"[AI Batch] Completed request: {req_id}")
                        
            except json.JSONDecodeError as e:
                print(f"[AI Batch] JSON Parse Error: {e}")
                print(f"Response was: {response_text[:100]}...")
                for req_id, _ in requests_batch:
                    self.db_ref.child("ai_requests").child(req_id).update({
                        "status": "error",
                        "error": "AI response format error"
                    })
                    
        except Exception as e:
            print(f"[AI Batch Processing] Error: {e}")
            import time
            for req_id, _ in requests_batch:
                self.db_ref.child("ai_requests").child(req_id).update({
                    "status": "error",
                    "error": str(e)
                })
    
    def generate_suggestions(self, reason):
        """Generate AI-powered suggestions based on congestion reason to rectify the congestion """
        try:
            from suggestion_generator import get_suggestions_for_reason
            return get_suggestions_for_reason(reason)
        except ImportError:
            # Fallback if suggestion generator not available
            return [
                "Monitor the situation closely",
                "Consider alternative routes",
                "Check traffic updates regularly"
            ]

    def get_data_from_firebase(self, path="/"):
        if not self.db_ref:
            raise ValueError("Firebase not initialized. Provide cred_path and db_url.")
        return self.db_ref.child(path).get()

    def get_data_from_json(self, json_str):
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}. Attempting loose parsing...")
            try:
                fixed_str = json_str.replace('": [', '": {').replace(']"', '}"')
                if fixed_str.strip().endswith(']'):
                    fixed_str = fixed_str.strip()[:-1] + '}'
                return json.loads(fixed_str)
            except:
                raise ValueError(f"Invalid JSON format: {e}")

    def process_data(self, data):
        """
        Processes raw Firebase data into the target API format.
        Now includes: confidence, average_speed, stuck_ratio, status levels
        Expected Input Format: {"DATE": {"PUSH_ID": {...data...}}}
        """
        all_records = []
        print("process_data")
        
        # Normalize input to a flat list of records
        if isinstance(data, dict):
            for date_key, valid_data in data.items():
                if isinstance(valid_data, dict):
                    for record in valid_data.values():
                        if isinstance(record, dict):
                            if 'date' not in record:
                                record['date'] = date_key
                            all_records.append(record)
                elif isinstance(valid_data, list):
                    for item in valid_data:
                        if isinstance(item, dict):
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
                "graph": [],
                "daily_summary": [],
                "detailed_history": [],
                "stats": {
                    "avg_confidence": 0,
                    "avg_speed": 0,
                    "avg_stuck_ratio": 0,
                    "total_events": 0,
                    "status_breakdown": {}
                }
            }

        # Get today's date for filtering
        from datetime import datetime
        now = datetime.now()
        today_str = now.strftime("%d-%m-%Y")
        
        # Aggregation with new fields
        latest_vehicle_count = 0  # Will be set to most recent event's count
        today_vehicle_count = 0   # Sum of today's vehicle counts only
        reasons = []
        reports_map = {}
        timestamps = []
        
        # New aggregations
        confidence_values = []
        speed_values = []
        stuck_ratio_values = []
        status_counts = Counter()
        
        # Filter for today's records for current stats
        today_records = [r for r in all_records if r.get('date') == today_str]

        for record in all_records:
            # Track vehicle count from each record
            v_count = record.get('vehicle_count', 0)
            if isinstance(v_count, (int, float)):
                v_count = int(v_count)
                # Track for today only
                if record.get('date') == today_str:
                    today_vehicle_count = v_count  # Use latest, not sum
            
            # Reason processing
            reason = record.get('reason', 'Unknown')
            if len(reason) > 50:
                reason = reason[:47] + "..."
            
            category = reason
            reasons.append(category)

            # Report generation
            if category not in reports_map:
                suggestions = self.generate_suggestions(category)
                reports_map[category] = {
                    "reason": category,
                    "suggestions": suggestions
                }

            # Graph Data
            ts = record.get('timestamp')
            if ts:
                timestamps.append({
                    't': ts,
                    'v': v_count,
                    'date': record.get('date', 'Unknown'),
                    'reason': category,
                    'confidence': record.get('confidence', 0),
                    'average_speed': record.get('average_speed', 0),
                    'stuck_ratio': record.get('stuck_ratio', 0),
                    'status': record.get('status', 'Unknown')
                })
            
            # New field aggregations
            if 'confidence' in record:
                confidence_values.append(float(record['confidence']))
            if 'average_speed' in record:
                speed_values.append(float(record['average_speed']))
            if 'stuck_ratio' in record:
                stuck_ratio_values.append(float(record['stuck_ratio']))
            if 'status' in record:
                status_counts[record['status']] += 1

        # Calculate Percentages for congestion pie chart
        congestion_stats = []
        reason_counts = Counter(reasons)
        total_reasons = sum(reason_counts.values())
        
        for r, count in reason_counts.items():
            percentage = round((count / total_reasons) * 100, 1) if total_reasons > 0 else 0
            congestion_stats.append({
                "name": r,
                "percentage": percentage,
                "count": count
            })

        # Format Time (Latest)
        timestamps.sort(key=lambda x: x['t'])
        latest_time_str = "N/A"
        if timestamps:
            latest_ts = timestamps[-1]['t']
            dt_obj = datetime.fromtimestamp(latest_ts)
            hour = dt_obj.hour % 12 or 12
            minute = dt_obj.strftime("%M")
            latest_time_str = f"{hour}:{minute}"

        # Reason percentages lookup
        reason_percentages = {c['name']: c['percentage'] for c in congestion_stats}
        
        # Graph Data Formatting with new fields
        graph_data = []
        for point in timestamps:
            dt = datetime.fromtimestamp(point['t'])
            time_label = dt.strftime('%H:%M:%S')
            timestamp_str = f"{time_label},{point['date']}"
            graph_data.append({
                "timestamp": timestamp_str,
                "percentage": reason_percentages.get(point['reason'], 0),
                "reason": point['reason'],
                "confidence": point['confidence'],
                "average_speed": point['average_speed'],
                "stuck_ratio": point['stuck_ratio'],
                "status": point['status']
            })

        # Filter for Last 10 Days
        unique_dates = sorted(list(set(r.get('date', 'Unknown') for r in all_records)), 
                              key=lambda d: datetime.strptime(d, "%d-%m-%Y") if d != 'Unknown' else datetime.min)
        last_10_dates = unique_dates[-10:] if unique_dates else []
        recent_records = [r for r in all_records if r.get('date') in last_10_dates]

        # Detailed History with new fields
        detailed_history = []
        for r in recent_records:
            detailed_history.append({
                "date": r.get('date', 'Unknown'),
                "time": r.get('time', 'N/A'),
                "vehicleCount": int(r.get('vehicle_count', 0)),
                "reason": r.get('reason', 'Unknown'),
                "congestion_status": r.get('status', 'Unknown'),
                "timestamp": r.get('timestamp', 0),
                "confidence": r.get('confidence', 0),
                "average_speed": r.get('average_speed', 0),
                "stuck_ratio": r.get('stuck_ratio', 0)
            })
        
        detailed_history.sort(key=lambda x: x['timestamp'], reverse=True)

        # Daily Summary with enhanced stats
        daily_stats = {}
        for record in recent_records:
            date_key = record.get('date', 'Unknown')
            if date_key not in daily_stats:
                daily_stats[date_key] = {
                    "date": date_key,
                    "totalEvents": 0,
                    "peakVehicleCount": 0,
                    "avgConfidence": [],
                    "avgSpeed": [],
                    "avgStuckRatio": []
                }
            
            daily_stats[date_key]["totalEvents"] += 1
            v_count = int(record.get('vehicle_count', 0))
            if v_count > daily_stats[date_key]["peakVehicleCount"]:
                daily_stats[date_key]["peakVehicleCount"] = v_count
            
            # Collect values for averaging
            if 'confidence' in record:
                daily_stats[date_key]["avgConfidence"].append(float(record['confidence']))
            if 'average_speed' in record:
                daily_stats[date_key]["avgSpeed"].append(float(record['average_speed']))
            if 'stuck_ratio' in record:
                daily_stats[date_key]["avgStuckRatio"].append(float(record['stuck_ratio']))
        
        # Calculate daily averages
        daily_summary = []
        for date_key, stats in daily_stats.items():
            daily_summary.append({
                "date": stats["date"],
                "totalEvents": stats["totalEvents"],
                "peakVehicleCount": stats["peakVehicleCount"],
                "avgConfidence": round(sum(stats["avgConfidence"]) / len(stats["avgConfidence"]), 2) if stats["avgConfidence"] else 0,
                "avgSpeed": round(sum(stats["avgSpeed"]) / len(stats["avgSpeed"]), 2) if stats["avgSpeed"] else 0,
                "avgStuckRatio": round(sum(stats["avgStuckRatio"]) / len(stats["avgStuckRatio"]), 3) if stats["avgStuckRatio"] else 0
            })
        
        daily_summary.sort(key=lambda x: datetime.strptime(x['date'], "%d-%m-%Y") if x['date'] != 'Unknown' else datetime.min, reverse=True)

        # Status breakdown for pie chart
        status_breakdown = []
        total_status = sum(status_counts.values())
        for status, count in status_counts.items():
            percentage = round((count / total_status) * 100, 1) if total_status > 0 else 0
            status_breakdown.append({
                "name": status,
                "count": count,
                "percentage": percentage
            })

        # Overall statistics
        stats = {
            "avg_confidence": round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else 0,
            "avg_speed": round(sum(speed_values) / len(speed_values), 2) if speed_values else 0,
            "avg_stuck_ratio": round(sum(stuck_ratio_values) / len(stuck_ratio_values), 3) if stuck_ratio_values else 0,
            "total_events": len(all_records),
            "status_breakdown": status_breakdown
        }

        return {
            "vehicleCount": today_vehicle_count,  # Current/latest event's count, not cumulative
            "time": latest_time_str,
            "congestion": congestion_stats,
            "report": list(reports_map.values()),
            "graph": graph_data,
            "daily_summary": daily_summary,
            "detailed_history": detailed_history,
            "stats": stats
        }

if __name__ == "__main__":
    # Test with sample including new fields
    sample_json = """
    {
      "07-01-2026": {
        "-OAbc123xyz": {
          "date": "07-01-2026",
          "time": "01:55",
          "timestamp": 1736193350.123,
          "status": "Moderate Congestion",
          "reason": "stop and go traffic pattern",
          "vehicle_count": 6,
          "confidence": 0.58,
          "average_speed": 3.4,
          "stuck_ratio": 0.42
        },
        "-OAbc123abc": {
          "date": "07-01-2026",
          "time": "01:56",
          "timestamp": 1736193410.123,
          "status": "Heavy Congestion",
          "reason": "bumper to bumper traffic",
          "vehicle_count": 8,
          "confidence": 0.75,
          "average_speed": 1.2,
          "stuck_ratio": 0.68
        }
      }
    }
    """
    processor = TrafficProcessor()
    data = processor.get_data_from_json(sample_json)
    result = processor.process_data(data)
    print(json.dumps(result, indent=2))
