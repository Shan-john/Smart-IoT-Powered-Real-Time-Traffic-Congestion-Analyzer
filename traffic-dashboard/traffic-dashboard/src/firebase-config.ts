// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getDatabase, ref, onValue, set, update, push, get, child, Database, Unsubscribe } from "firebase/database";

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

// Type definitions
interface TrafficStats {
    avg_confidence: number;
    avg_speed: number;
    avg_stuck_ratio: number;
    total_events: number;
    status_breakdown: Array<{ status: string; count: number }>;
}

interface ProcessedData {
    vehicleCount: number;
    time: string;
    congestion: Array<{ name: string; percentage: number }>;
    report: Array<{ reason: string; suggestions: string[] }>;
    graph: Array<{
        average_speed: number;
        confidence: number;
        percentage: number;
        reason: string;
        status: string;
        stuck_ratio: number;
        timestamp: string;
    }>;
    daily_summary: Array<unknown>;
    detailed_history: Array<unknown>;
    stats: TrafficStats;
}

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const database: Database = getDatabase(app);


/**
 * Subscribe to processed_data updates from Firebase
 * Reads pre-processed data from backend (no client-side processing needed)
 * Now includes enhanced fields: stats, confidence, speed, stuck_ratio
 */
export const subscribeToProcessedData = (callback: (data: ProcessedData) => void): Unsubscribe => {
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
 * @param path - The path to delete (e.g., 'traffic_data' or 'processed_data')
 */
export const deleteFirebaseData = async (path: string): Promise<void> => {
    const dataRef = ref(database, path);
    console.log(`[Firebase] Deleting data at path: ${path}`);
    await set(dataRef, null);
    console.log(`[Firebase] Successfully deleted data at path: ${path}`);
};

/**
 * Upload/set data at the specified Firebase path (overwrites existing data)
 * @param path - The path to upload to
 * @param data - The data to upload
 */
export const uploadFirebaseData = async (path: string, data: unknown): Promise<void> => {
    const dataRef = ref(database, path);
    console.log(`[Firebase] Uploading data to path: ${path}`, data);
    await set(dataRef, data);
    console.log(`[Firebase] Successfully uploaded data to path: ${path}`);
};

/**
 * Merge/update data at the specified Firebase path (preserves existing data)
 * @param path - The path to update
 * @param data - The data to merge
 */
export const uploadMergeFirebaseData = async (path: string, data: Record<string, unknown>): Promise<void> => {
    const dataRef = ref(database, path);
    console.log(`[Firebase] Merging data at path: ${path}`, data);
    await update(dataRef, data);
    console.log(`[Firebase] Successfully merged data at path: ${path}`);
};

// ============== AI Chat Firebase Functions ==============

interface AiRequest {
    query: string;
    status: 'pending' | 'processing' | 'completed' | 'error';
    timestamp: number;
    created_at: string;
}

interface AiResponse {
    request_id: string;
    query: string;
    html: string;
    status: 'completed' | 'error';
    error?: string;
    timestamp: number;
    created_at: string;
}

/**
 * Push a new AI request to Firebase for backend processing
 * @returns The request ID
 */
export const pushAiRequest = async (query: string): Promise<string> => {
    const requestsRef = ref(database, 'ai_requests');
    const newRequest: AiRequest = {
        query,
        status: 'pending',
        timestamp: Date.now() / 1000,
        created_at: new Date().toISOString()
    };

    console.log('[Firebase AI] Pushing request:', query);
    const newRef = await push(requestsRef, newRequest);
    const requestId = newRef.key!;
    console.log('[Firebase AI] Request ID:', requestId);
    return requestId;
};

/**
 * Subscribe to latest AI report updates (real-time listener)
 */
export const subscribeToAiResponse = (
    requestId: string,
    callback: (response: AiResponse | null) => void
): Unsubscribe => {
    // Listen to latest_ai_report for real-time updates
    const latestRef = ref(database, 'latest_ai_report');
    console.log('[Firebase AI] Subscribing to latest_ai_report');

    return onValue(latestRef, (snapshot) => {
        const data = snapshot.val();
        // Trigger callback for any completed report
        if (data && data.status === 'completed' && data.html) {
            console.log('[Firebase AI] Report received:', data.request_id);
            callback(data);
        }
    });
};

/**
 * Get latest AI report from Firebase
 */
export const getLatestAiReport = async (): Promise<AiResponse | null> => {
    const latestRef = ref(database, 'latest_ai_report');
    const snapshot = await get(latestRef);
    return snapshot.val();
};

/**
 * Get AI report history (last 20 reports)
 */
export const getAiReportHistory = async (): Promise<Array<{ report_id: string; query: string; timestamp: number }>> => {
    const responsesRef = ref(database, 'ai_responses');
    const snapshot = await get(responsesRef);
    const data = snapshot.val();

    if (!data) return [];

    const reports = Object.entries(data).map(([id, report]: [string, any]) => ({
        report_id: id,
        query: report.query || '',
        timestamp: report.timestamp || 0,
        date: report.created_at ? new Date(report.created_at).toLocaleDateString() : '',
        time: report.created_at ? new Date(report.created_at).toLocaleTimeString() : ''
    }));

    // Sort by timestamp descending
    reports.sort((a, b) => b.timestamp - a.timestamp);
    return reports.slice(0, 20);
};

/**
 * Get a specific AI report by ID
 */
export const getAiReport = async (reportId: string): Promise<AiResponse | null> => {
    const reportRef = ref(database, `ai_responses/${reportId}`);
    const snapshot = await get(reportRef);
    return snapshot.val();
};

export default database;
