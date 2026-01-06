import { configureStore, createSlice, PayloadAction } from '@reduxjs/toolkit';

// Define the initial state based on the provided JSON
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
  congestion: [] as { name: string; percentage: number }[],
  report: [] as { reason: string; suggestions: string[] }[],
  graph: [] as { timestamp: string; percentage: number; reason: string }[],
  daily_summary: [] as { date: string; totalEvents: number; peakVehicleCount: number }[],
  detailed_history: [] as { date: string; time: string; vehicleCount: number; reason: string; congestion_status: string; timestamp: number }[],
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

