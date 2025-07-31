import tempfile
import cv2
import time
import os
import firebase_admin
from firebase_admin import credentials, db
from sympy import true
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
 
model = YOLO("yolov5nu.pt")  # use 'yolov5n.pt' or yolov8 if preferred
 

 

 
# Tracker
tracker = SimpleTracker(distance_threshold=40, stuck_seconds=5)

# Video source
VIDEO_PATH = "video.mp4"  # or use 0 for webcam
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
   
    # Detect vehicles
    results = model.predict(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if model.names[cls_id] in ['car', 'truck', 'bus', 'motorbike']:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            detections.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Update tracker
    tracked = tracker.update(detections)
    stuck = tracker.get_stuck_vehicles()

    if stuck:
        
        # Capture current frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb) 

        try:            
             reason = analyze_congestion_reason(image_pil)  
                         
        except Exception as e:            
             reason = "Unknown (AI error)"           
             print("Reason analyzer error:", e)

        status = "Congestion"
    
        color = (0, 0, 255)
          
        # Firebase log only during congestion
        try:
            db.reference("traffic_data").push({
                'timestamp': time.time(),
                'status': status,
                'reason': reason,
                 
                'vehicle_count': len(detections)
            })
            db.reference("isCongestion").set(True)
        except Exception as e:
            print("Firebase error:", e)

    else:
        try: 
            db.reference("isCongestion").set(False)
        except:
            pass
        status = "Normal"
        color = (0, 255, 0)

    # Draw info
    for (i, (x, y)) in tracked:
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(frame, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Vehicles: {len(detections)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Display
    cv2.imshow("Traffic Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
