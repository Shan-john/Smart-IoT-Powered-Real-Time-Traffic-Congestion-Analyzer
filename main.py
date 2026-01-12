import cv2
import time
import threading
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
from tracker import EnhancedTracker
from congestion_analyzer import CongestionAnalyzer
from ultralytics import YOLO
from PIL import Image

# Firebase config
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://traffic-analyser-fad30-default-rtdb.firebaseio.com/'
})

# Load YOLOv5
model = YOLO("yolov5nu.pt")

# Enhanced Tracker with speed/flow metrics
tracker = EnhancedTracker(
    distance_threshold=50,      # Max pixels between frames to match
    stuck_speed_threshold=5.0,  # Below 5cd tra  px/s = stuck
    stuck_seconds=3.0           # 3 seconds stuck = congestion
)

# Congestion Analyzer with optical flow + rules (CLIP disabled for speed)
congestion_analyzer = CongestionAnalyzer(use_clip=False)

# Video source
VIDEO_PATH = "video.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)  # 0 for webcam, VIDEO_PATH for file

# Global State for Threading
last_analysis_time = 0
ANALYSIS_COOLDOWN = 10.0  # Seconds between Firebase updates
is_analyzing = False
current_analysis = {
    'level': 'Normal',
    'reason': '',
    'confidence': 0.0,
    'metrics': {}
}

def update_firebase_async(analysis_result: dict, detections_count: int):
    """Background task to update Firebase with congestion data."""
    global is_analyzing, last_analysis_time, current_analysis
    
    try:
        print(f"[Async] Updating Firebase: {analysis_result['reason']}")
        
        now = datetime.now()
        current_date = now.strftime("%d-%m-%Y")
        current_time = now.strftime("%H:%M")
        
        # Push to Firebase
        db.reference(f"traffic_data/{current_date}").push({
            'date': current_date,
            'time': current_time,
            'timestamp': time.time(),
            'status': analysis_result['level'],
            'reason': analysis_result['reason'],
            'vehicle_count': detections_count,
            'confidence': analysis_result['confidence'],
            'average_speed': analysis_result['metrics'].get('average_speed', 0),
            'stuck_ratio': analysis_result['metrics'].get('stuck_ratio', 0)
        })
        
        # Update congestion flag
        db.reference("isCongestion").set(analysis_result['is_congested'])
        print("[Async] Firebase updated successfully.")

    except Exception as e:
        print(f"[Async] Firebase error: {e}")
    finally:
        is_analyzing = False
        last_analysis_time = time.time()


def draw_overlay(frame, analysis, detections_count, metrics):
    """Draw enhanced status overlay on frame."""
    overlay = frame.copy()
    
    # Status colors
    level = analysis['level']
    if 'Severe' in level or 'Heavy' in level:
        color = (0, 0, 255)  # Red
        bg_color = (0, 0, 100)
    elif 'Moderate' in level or 'Light' in level:
        color = (0, 165, 255)  # Orange
        bg_color = (0, 80, 100)
    else:
        color = (0, 255, 0)  # Green
        bg_color = (0, 80, 0)
    
    # Draw background
    cv2.rectangle(overlay, (0, 0), (640, 130), bg_color, -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Status text
    cv2.putText(frame, f"Status: {level}", (10, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Metrics row
    avg_speed = metrics.get('average_speed', 0)
    stuck_ratio = metrics.get('stuck_ratio', 0) * 100
    cv2.putText(frame, f"Vehicles: {detections_count}  |  Avg Speed: {avg_speed:.1f} px/s  |  Stuck: {stuck_ratio:.0f}%", 
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Reason (if congested)
    if analysis['is_congested'] and analysis['reason']:
        cv2.putText(frame, f"Reason: {analysis['reason']}", 
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1)
        
        # Confidence bar
        conf = analysis['confidence']
        bar_width = int(150 * conf)
        cv2.rectangle(frame, (10, 95), (160, 110), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 95), (10 + bar_width, 110), color, -1)
        cv2.putText(frame, f"Conf: {conf:.0%}", (170, 107), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Optical flow indicator
    flow_mag = metrics.get('optical_flow', 0)
    flow_text = "High" if flow_mag > 3 else "Medium" if flow_mag > 1 else "Low"
    cv2.putText(frame, f"Flow: {flow_text}", (550, 125), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return frame


try:
    print("[Main] Starting enhanced traffic monitoring...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video if using file
            if VIDEO_PATH:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # Resize for consistent processing
        frame = cv2.resize(frame, (640, 480))
        
        # Detect vehicles with YOLO
        results = model.predict(frame, verbose=False)[0]
        detections = []
        
        # Process detections
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] in ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'scooter']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                detections.append((cx, cy))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Update tracker with detections
        tracked = tracker.update(detections)
        
        # Get comprehensive traffic metrics
        traffic_metrics = tracker.get_traffic_metrics(640, 480)
        
        # Analyze congestion using rules + optical flow
        analysis = congestion_analyzer.analyze(frame, traffic_metrics)
        current_analysis = analysis
        
        # Draw stuck vehicles
        for obj_id, pos in traffic_metrics['stuck_vehicles']:
            cv2.circle(frame, pos, 8, (0, 0, 255), 2)
        
        # Firebase update logic
        if analysis['is_congested']:
            if not is_analyzing and (time.time() - last_analysis_time > ANALYSIS_COOLDOWN):
                is_analyzing = True
                
                # Start background thread for Firebase
                thread = threading.Thread(
                    target=update_firebase_async,
                    args=(analysis, len(detections))
                )
                thread.daemon = True
                thread.start()
        
        # Draw overlay with enhanced metrics
        frame = draw_overlay(frame, analysis, len(detections), analysis['metrics'])

        # Display
        cv2.imshow("Traffic Monitor (Enhanced)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("[Main] Stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[Main] Cleanup complete")
