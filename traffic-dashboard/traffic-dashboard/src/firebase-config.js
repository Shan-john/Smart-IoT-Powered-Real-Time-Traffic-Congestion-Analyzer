// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getDatabase, ref, onValue } from "firebase/database";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCahTAAw0-CZus_-b0LqE9K_t6n3TWV350",
  authDomain: "traffic-analyser-fad30.firebaseapp.com",
  databaseURL: "https://traffic-analyser-fad30-default-rtdb.firebaseio.com",
  projectId: "traffic-analyser-fad30",
  storageBucket: "traffic-analyser-fad30.firebasestorage.app",
  messagingSenderId: "442603118634",
  appId: "1:442603118634:web:cf71a305e4fe4ca0a8a353"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const database = getDatabase(app);


/**
 * Subscribe to processed_data updates from Firebase
 * Reads pre-processed data from backend (no client-side processing needed)
 * Now includes enhanced fields: stats, confidence, speed, stuck_ratio
 */
export const subscribeToProcessedData = (callback) => {
  // Listen directly to processed_data (pre-processed by backend)
  const processedDataRef = ref(database, 'processed_data');
  console.log('[Firebase] Setting up listener for processed_data...');
  
  return onValue(processedDataRef, (snapshot) => {
    console.log('[Firebase] Snapshot received:', snapshot.exists());
    const data = snapshot.val();
    console.log('[Firebase] Data from processed_data:', data);
    
    if (data) {
      // Data is already processed by backend, pass directly
      callback({
        vehicleCount: data.vehicleCount || 0,
        time: data.time || "N/A",
        congestion: data.congestion || [],
        report: data.report || [],
        graph: data.graph || [],
        daily_summary: data.daily_summary || [],
        detailed_history: data.detailed_history || [],
        // New stats field with enhanced metrics
        stats: data.stats || {
          avg_confidence: 0,
          avg_speed: 0,
          avg_stuck_ratio: 0,
          total_events: 0,
          status_breakdown: []
        }
      });
    } else {
      console.log('[Firebase] No data found at processed_data path');
      // Return empty structure with stats
      callback({
        vehicleCount: 0,
        time: "N/A",
        congestion: [],
        report: [],
        graph: [],
        daily_summary: [],
        detailed_history: [],
        stats: {
          avg_confidence: 0,
          avg_speed: 0,
          avg_stuck_ratio: 0,
          total_events: 0,
          status_breakdown: []
        }
      });
    }
  }, (error) => {
    console.error('[Firebase] Error reading data:', error);
  });
};

export default database;