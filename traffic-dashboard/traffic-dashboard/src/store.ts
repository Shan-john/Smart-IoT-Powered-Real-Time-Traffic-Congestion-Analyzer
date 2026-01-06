import { configureStore, createSlice, PayloadAction } from '@reduxjs/toolkit';

// Define types for better type safety
interface CongestionItem {
  name: string;
  percentage: number;
  count?: number;
}

interface ReportItem {
  reason: string;
  suggestions: string[];
}

interface GraphItem {
  timestamp: string;
  percentage: number;
  reason: string;
  confidence?: number;
  average_speed?: number;
  stuck_ratio?: number;
  status?: string;
}

interface DailySummaryItem {
  date: string;
  totalEvents: number;
  peakVehicleCount: number;
  avgConfidence?: number;
  avgSpeed?: number;
  avgStuckRatio?: number;
}

interface DetailedHistoryItem {
  date: string;
  time: string;
  vehicleCount: number;
  reason: string;
  congestion_status: string;
  timestamp: number;
  confidence?: number;
  average_speed?: number;
  stuck_ratio?: number;
}

interface StatusBreakdownItem {
  name: string;
  count: number;
  percentage: number;
}

interface Stats {
  avg_confidence: number;
  avg_speed: number;
  avg_stuck_ratio: number;
  total_events: number;
  status_breakdown: StatusBreakdownItem[];
}

// Define the initial state
const initialState = {
  vehicleCount: 0,
  time: "N/A",
  // Current active view state
  currentView: 'dashboard', // 'dashboard', 'live', 'summary', 'settings'
  showMobileMenu: false,
  // Loading and error states
  loading: false,
  error: null as string | null,
  // Data
  congestion: [] as CongestionItem[],
  report: [] as ReportItem[],
  graph: [] as GraphItem[],
  daily_summary: [] as DailySummaryItem[],
  detailed_history: [] as DetailedHistoryItem[],
  // New stats field
  stats: {
    avg_confidence: 0,
    avg_speed: 0,
    avg_stuck_ratio: 0,
    total_events: 0,
    status_breakdown: [] as StatusBreakdownItem[]
  } as Stats,
};

const trafficSlice = createSlice({
  name: 'traffic',
  initialState,
  reducers: {
    updateTime: (state, action: PayloadAction<string>) => {
      state.time = action.payload;
    },
    setView: (state, action: PayloadAction<string>) => {
      state.currentView = action.payload;
    },
    toggleMobileMenu: (state) => {
      state.showMobileMenu = !state.showMobileMenu;
    },
    closeMobileMenu: (state) => {
      state.showMobileMenu = false;
    },
    setTrafficData: (state, action: PayloadAction<any>) => {
      const data = action.payload;
      state.vehicleCount = data.vehicleCount || 0;
      state.time = data.time || "N/A";
      state.congestion = data.congestion || [];
      state.report = data.report || [];
      state.graph = data.graph || [];
      state.daily_summary = data.daily_summary || [];
      state.detailed_history = data.detailed_history || [];
      state.stats = data.stats || {
        avg_confidence: 0,
        avg_speed: 0,
        avg_stuck_ratio: 0,
        total_events: 0,
        status_breakdown: []
      };
      state.loading = false;
      state.error = null;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
      state.loading = false;
    },
  },
});

export const { updateTime, setView, toggleMobileMenu, closeMobileMenu, setTrafficData, setLoading, setError } = trafficSlice.actions;

export const store = configureStore({
  reducer: {
    traffic: trafficSlice.reducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
