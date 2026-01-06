import firebase_admin
from firebase_admin import credentials, db
import json
import os
from traffic_processor import TrafficProcessor

# Initialize Firebase
cred = credentials.Certificate("../firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://traffic-analyser-fad30-default-rtdb.firebaseio.com/'
})

# Sample Data (2 Days)
sample_data = {
    "01-08-2025": {
        "id1": { 
            "vehicle_count": 5, 
            "reason": "Signal Issue", 
            "status": "Signal Delay",
            "time": "08:30",
            "timestamp": 1754061642.536 
        },
        "id2": { 
            "vehicle_count": 2, 
            "reason": "Illegal Parking", 
            "status": "Congestion",
            "time": "09:15",
            "timestamp": 1754064342.536 
        }
    },
    "02-08-2025": {
        "id3": { 
            "vehicle_count": 8, 
            "reason": "Accident", 
            "status": "Severe Congestion",
            "time": "17:45",
            "timestamp": 1754148942.536 
        }
    }
}

# Process
processor = TrafficProcessor()
# Manually inject db_ref since we initialized it here
processor.db_ref = db.reference()

# Process data
processed = processor.process_data(sample_data)

# Write to Firebase
print("Writing processed data to Firebase 'processed_data' path...")
db.reference("processed_data").set(processed)
print("Done! Dashboard should now show content.")
