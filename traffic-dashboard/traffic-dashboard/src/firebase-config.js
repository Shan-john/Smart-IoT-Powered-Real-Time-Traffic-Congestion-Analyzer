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

/**
 * Delete all data at the specified Firebase path
 * @param {string} path - The path to delete (e.g., 'traffic_data' or 'processed_data')
 */
export const deleteFirebaseData = async (path) => {
  const { set } = await import('firebase/database');
  const dataRef = ref(database, path);
  console.log(`[Firebase] Deleting data at path: ${path}`);
  await set(dataRef, null);
  console.log(`[Firebase] Successfully deleted data at path: ${path}`);
};

/**
 * Upload/set data at the specified Firebase path (overwrites existing data)
 * @param {string} path - The path to upload to
 * @param {object} data - The data to upload
 */
export const uploadFirebaseData = async (path, data) => {
  const { set } = await import('firebase/database');
  const dataRef = ref(database, path);
  console.log(`[Firebase] Uploading data to path: ${path}`, data);
  await set(dataRef, data);
  console.log(`[Firebase] Successfully uploaded data to path: ${path}`);
};

/**
 * Merge/update data at the specified Firebase path (preserves existing data)
 * @param {string} path - The path to update
 * @param {object} data - The data to merge
 */
export const uploadMergeFirebaseData = async (path, data) => {
  const { update } = await import('firebase/database');
  const dataRef = ref(database, path);
  console.log(`[Firebase] Merging data at path: ${path}`, data);
  await update(dataRef, data);
  console.log(`[Firebase] Successfully merged data at path: ${path}`);
};

export default database;