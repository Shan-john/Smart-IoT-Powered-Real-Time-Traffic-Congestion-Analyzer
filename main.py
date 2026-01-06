import tempfile
import cv2
import time
import os
import threading
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
from reason_analyzer import analyze_congestion_reason
from tracker import SimpleTracker
from ultralytics import YOLO
from PIL import Image

# Firebase config
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://traffic-analyser-fad30-default-rtdb.firebaseio.com/'
})

# Load YOLOv5
model = YOLO("yolov5nu.pt")   

# Tracker
tracker = SimpleTracker(distance_threshold=40, stuck_seconds=5)

# Video source
VIDEO_PATH = "video.mp4"  # or use 0 for webcam
cap = cv2.VideoCapture(0)

# Global State for Threading
last_analysis_time = 0
ANALYSIS_COOLDOWN = 10.0  # Seconds between analysis (increased to prevent lag)
current_status = "Normal"
current_reason = ""
is_analyzing = False
analysis_thread = None

def process_congestion_async(image_pil, detections_count):
    """Background task to analyze reason and update Firebase"""
    global current_status, current_reason, is_analyzing, last_analysis_time
    
    try:
        print("[Async] Starting analysis...")
        # 1. AI Analysis (Heavy Task)
        reason = analyze_congestion_reason(image_pil)
        current_reason = reason
        print(f"[Async] Analysis complete: {reason}")

        # 2. Firebase Push (Network Task)
        now = datetime.now()
        current_date = now.strftime("%d-%m-%Y")
        current_time = now.strftime("%H:%M")
        
        suggestion = "Check traffic lights or road conditions"
        
        db.reference(f"traffic_data/{current_date}").push({
            'date': current_date,
            'time': current_time,
            'timestamp': time.time(),
            'status': "Congestion",
            'reason': reason,
            'suggestion': suggestion,
            'vehicle_count': detections_count
        })
        db.reference("isCongestion").set(True)
        print("[Async] Firebase updated.")

    except Exception as e:
        print(f"[Async] Error: {e}")
        current_reason = "Error analyzing"
    finally:
        is_analyzing = False
        last_analysis_time = time.time()
        # If analysis finished and we are still "Analyzing...", update status
        if current_status == "Analyzing...":
             current_status = "Congestion"

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Detect vehicles
        results = model.predict(frame, verbose=False)[0]
        detections = []
        
        # Draw detections
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] in ['car', 'truck', 'bus', 'motorbike','bicycle',"scooter"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                detections.append((cx, cy))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Update tracker
        tracked = tracker.update(detections)
        stuck = tracker.get_stuck_vehicles()

        # Logic for Congestion Detection & Analysis
        if stuck:
            # If we are not currently analyzing and cooldown has passed
            if not is_analyzing and (time.time() - last_analysis_time > ANALYSIS_COOLDOWN):
                is_analyzing = True
                current_status = "Analyzing..."
                current_reason = "Identifying reason..."
                
                # Copy frame for thread to avoid race conditions
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(frame_rgb)
                
                # Start background thread
                analysis_thread = threading.Thread(
                    target=process_congestion_async,
                    args=(image_pil, len(detections))
                )
                analysis_thread.daemon = True
                analysis_thread.start()
            
            elif is_analyzing:
                current_status = "Analyzing..."
            else:
                current_status = "Congestion"

        else:
            # No stuck vehicles
            # Only switch to Normal if not in the middle of analyzing a valid stuck event
            if not is_analyzing:
                current_status = "Normal"
                current_reason = ""
                # Optional: Update Firebase to clear congestion status occasionally
                # To prevent spam, do this less frequently or relying on the 'Congestion' pulses is often enough
                # db.reference("isCongestion").set(False) 

        # --- Drawing Overlay Info ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Determine color based on status
        if current_status == "Congestion":
            color = (0, 0, 255) # Red
        elif current_status == "Analyzing...":
            color = (0, 165, 255) # Orange
        else:
            color = (0, 255, 0) # Green

        # Draw text
        cv2.putText(frame, f"Status: {current_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Vehicles: {len(detections)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if current_reason:
            cv2.putText(frame, f"Reason: {current_reason}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

        # Display
        cv2.imshow("Traffic Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
