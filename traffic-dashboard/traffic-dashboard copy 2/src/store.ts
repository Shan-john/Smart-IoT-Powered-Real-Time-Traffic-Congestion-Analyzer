import { configureStore, createSlice, PayloadAction } from '@reduxjs/toolkit';

// Define the initial state based on the provided JSON
const initialState = {
  vehicleCount: 15,
  time: "3:30",
  // Current active view state
  currentView: 'dashboard', // 'dashboard', 'live', 'summary', 'settings'
  showMobileMenu: false,
  // Data
  congestion: [
    { name: "Wrong Parking", percentage: 50 },
    { name: "Signal Delay", percentage: 33.3 },
    { name: "Road Block", percentage: 16.7 }
  ],
  report: [
    {
      reason: "Wrong Parking (Illegal Or Unplanned Vehicle Stops)",
      suggestion: "Remove parked vehicles to restore traffic flow."
    },
    {
      reason: "Signal Delay",
      suggestion: "Optimize signal timing for smoother traffic."
    }
  ]
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
    }
  },
});

export const { updateTime, setView, toggleMobileMenu, closeMobileMenu } = trafficSlice.actions;

export const store = configureStore({
  reducer: {
    traffic: trafficSlice.reducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
