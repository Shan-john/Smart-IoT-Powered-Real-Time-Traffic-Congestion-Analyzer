import React, { useState, useMemo } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState, AppDispatch, setView, toggleMobileMenu, closeMobileMenu } from '../store';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, LineChart, Line, XAxis, YAxis, CartesianGrid, Legend, BarChart, Bar } from 'recharts';
import { LayoutDashboard, Activity, FileText, Settings, Menu, X, MapPin, Clock, Car, Loader2, AlertCircle, RefreshCw, ChevronRight, ChevronLeft, ArrowLeft, Filter, Calendar, Search } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useToast } from '../hooks/use-toast';

// --- Components ---

const Sidebar = () => {
  const dispatch = useDispatch();
  const currentView = useSelector((state: RootState) => state.traffic.currentView);

  const menuItems = [
    { id: 'dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { id: 'live', icon: Activity, label: 'Live Monitoring', subItem: true },
    { id: 'summary', icon: FileText, label: 'Recent Summary', subItem: true },
    { id: 'settings', icon: Settings, label: 'Settings' },
  ];

  return (
    <div className="w-64 h-full bg-[#54606e] text-white flex flex-col p-6 hidden md:flex shadow-xl relative overflow-hidden z-20" data-testid="sidebar">
      <div className="mb-12 cursor-pointer" onClick={() => dispatch(setView('dashboard'))}>
        <h1 className="text-sm font-bold font-sans tracking-wide opacity-90">Smart Traffic Monitor</h1>
      </div>

      <div className="flex-1 flex flex-col">
        <div
          className={`mb-6 flex items-center text-white/90 cursor-pointer hover:text-white transition-colors ${currentView === 'dashboard' ? 'text-white font-bold' : ''}`}
          onClick={() => dispatch(setView('dashboard'))}
        >
          <LayoutDashboard className="w-5 h-5 mr-3" />
          <span className="text-sm">Dashboard</span>
        </div>

        {/* Toggle-like Menu */}
        <div className="bg-white/10 rounded-2xl p-1.5 mb-auto flex flex-col gap-1">
          <div
            className={`rounded-xl px-4 py-2 text-xs font-bold shadow-sm text-center cursor-pointer transition-all duration-200 flex items-center justify-center gap-2
              ${currentView === 'live' ? 'bg-white text-slate-800 scale-105' : 'text-white/70 hover:bg-white/5 hover:text-white'}`}
            onClick={() => dispatch(setView('live'))}
          >
            {currentView === 'live' && <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>}
            live monitoring
          </div>
          <div
            className={`rounded-xl px-4 py-2 text-xs font-bold shadow-sm text-center cursor-pointer transition-all duration-200
              ${currentView === 'summary' ? 'bg-white text-slate-800 scale-105' : 'text-white/70 hover:bg-white/5 hover:text-white'}`}
            onClick={() => dispatch(setView('summary'))}
          >
            recent summary
          </div>
        </div>

        {/* Map Section at Bottom — FIXED */}
        <div
          className="mt-auto w-full rounded-2xl overflow-hidden relative aspect-square border-2 border-white/20 group cursor-pointer shadow-lg hover:shadow-2xl transition-all duration-300 hover:-translate-y-1"
          onClick={() => dispatch(setView('map'))}
        >
          <iframe
            className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity duration-500 hover:scale-110"

            src="https://www.google.com/maps/embed?pb=!1m17!1m12!1m3!1d4494.924008878361!2d76.90539207501351!3d8.57107499147314!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m2!1m1!2zOMKwMzQnMTUuOSJOIDc2wrA1NCcyOC43IkU!5e1!3m2!1sen!2sin!4v1767719153017!5m2!1sen!2sin"
            title="Live Traffic Map"
            loading="lazy"
            style={{ border: 0 }}
            allowFullScreen
            referrerPolicy="no-referrer-when-downgrade"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent flex items-end p-4">
            <div className="flex items-center text-white">
              <MapPin className="w-3 h-3 mr-1.5" />
              <span className="text-xs font-bold">View Live Map</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};


const MobileMenu = () => {
  const dispatch = useDispatch();
  const { showMobileMenu, currentView } = useSelector((state: RootState) => state.traffic);

  const handleNav = (view: string) => {
    dispatch(setView(view));
    dispatch(closeMobileMenu());
  };

  return (
    <AnimatePresence>
      {showMobileMenu && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 z-40 md:hidden backdrop-blur-sm"
            onClick={() => dispatch(closeMobileMenu())}
          />
          <motion.div
            initial={{ x: '-100%' }}
            animate={{ x: 0 }}
            exit={{ x: '-100%' }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="fixed inset-y-0 left-0 w-[80%] max-w-xs bg-[#54606e] z-50 p-6 flex flex-col shadow-2xl md:hidden"
          >
            <div className="flex justify-between items-center mb-10 text-white">
              <h1 className="text-lg font-bold">Smart Traffic</h1>
              <button onClick={() => dispatch(closeMobileMenu())} className="p-2 hover:bg-white/10 rounded-full">
                <X className="w-6 h-6" />
              </button>
            </div>

            <nav className="space-y-4">
              <div
                className={`p-3 rounded-xl flex items-center cursor-pointer ${currentView === 'dashboard' ? 'bg-white text-slate-800' : 'text-white hover:bg-white/10'}`}
                onClick={() => handleNav('dashboard')}
              >
                <LayoutDashboard className="w-5 h-5 mr-3" />
                <span className="font-bold">Dashboard</span>
              </div>
              <div
                className={`p-3 rounded-xl flex items-center cursor-pointer ${currentView === 'live' ? 'bg-white text-slate-800' : 'text-white hover:bg-white/10'}`}
                onClick={() => handleNav('live')}
              >
                <Activity className="w-5 h-5 mr-3" />
                <span className="font-bold">Live Monitoring</span>
              </div>
              <div
                className={`p-3 rounded-xl flex items-center cursor-pointer ${currentView === 'summary' ? 'bg-white text-slate-800' : 'text-white hover:bg-white/10'}`}
                onClick={() => handleNav('summary')}
              >
                <FileText className="w-5 h-5 mr-3" />
                <span className="font-bold">Recent Summary</span>
              </div>
            </nav>

            <div className="mt-auto">
              <div className="p-4 bg-white/10 rounded-2xl">
                <p className="text-xs text-white/60 mb-2">Current Status</p>
                <div className="flex items-center text-green-400 text-sm font-bold">
                  <span className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></span>
                  System Online
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

const StatCard = ({ value, label, colorClass, delay, icon: Icon }: { value: string | number, label: string, colorClass: string, delay: number, icon?: any }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      whileHover={{ scale: 1.05, y: -5 }}
      className={`rounded-3xl p-4 w-32 h-40 flex flex-col justify-center items-center shadow-lg ${colorClass} cursor-pointer relative overflow-hidden group`}
      data-testid={`stat-card-${label.toLowerCase()}`}
    >
      {/* Background decoration */}
      <div className="absolute -right-4 -top-4 w-16 h-16 rounded-full bg-white/20 blur-xl group-hover:bg-white/30 transition-colors"></div>

      {Icon && <Icon className="w-6 h-6 text-slate-800/50 mb-2" />}
      <span className="text-4xl font-bold text-slate-900 mb-2 tracking-tight relative z-10">{value}</span>
      <span className="text-[10px] uppercase tracking-wider font-bold text-slate-800/70 relative z-10">{label}</span>
    </motion.div>
  );
};

const CongestionChart = ({ data }: { data: any[] }) => {
  // Colors from the image: Red/Orange, Yellow/Orange, Blue
  const COLORS = ['#ef4444', '#f59e0b', '#3b82f6'];
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  return (
    <div className="h-64 w-full flex items-center justify-between px-4">
      <div className="h-full w-1/2 relative">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={0}
              outerRadius={80}
              paddingAngle={0}
              dataKey="percentage"
              stroke="white"
              strokeWidth={2}
              onMouseEnter={(_, index) => setActiveIndex(index)}
              onMouseLeave={() => setActiveIndex(null)}
            >
              {data.map((entry: any, index: number) => (
                <Cell
                  key={`cell-${index}`}
                  fill={COLORS[index % COLORS.length]}
                  stroke={activeIndex === index ? "white" : "none"}
                  strokeWidth={activeIndex === index ? 4 : 0}
                  className="transition-all duration-300"
                  style={{
                    filter: activeIndex === index ? 'drop-shadow(0px 4px 8px rgba(0,0,0,0.2))' : 'none',
                    transform: activeIndex === index ? 'scale(1.05)' : 'scale(1)',
                    transformOrigin: 'center'
                  }}
                />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)', padding: '8px 12px' }}
              itemStyle={{ fontSize: '12px', fontWeight: 600, color: '#334155' }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Custom Legend matching the image style */}
      <div className="flex flex-col space-y-3 pl-4">
        {data.map((entry: any, index: number) => (
          <motion.div
            key={index}
            className={`flex items-center text-xs cursor-pointer p-2 rounded-lg transition-colors ${activeIndex === index ? 'bg-slate-100' : ''}`}
            onMouseEnter={() => setActiveIndex(index)}
            onMouseLeave={() => setActiveIndex(null)}
          >
            <div className="w-3 h-3 rounded-[2px] mr-3 shadow-sm" style={{ backgroundColor: COLORS[index % COLORS.length] }}></div>
            <span className="text-slate-500 font-bold tracking-wide">{entry.name}</span>
            <span className="ml-auto font-mono text-slate-400 text-[10px] pl-2">{entry.percentage}%</span>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

const ReportCard = ({ reason, suggestions, index }: { reason: string, suggestions: string[], index: number }) => {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay: 0.2 + (index * 0.1) }}
      className="mb-6 last:mb-0 group cursor-default"
      data-testid={`report-item-${index}`}
    >
      <div className="flex items-center mb-2">
        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wide">Reason & Suggestions</h4>
        <div className="ml-2 h-px bg-slate-200 flex-1 group-hover:bg-slate-300 transition-colors"></div>
      </div>
      <div className="mb-3">
        <p className="text-sm font-semibold text-slate-800 group-hover:text-blue-600 transition-colors">{reason}</p>
      </div>
      <div className="bg-slate-50 p-3 rounded-lg border border-transparent group-hover:border-slate-100 transition-all space-y-2">
        {suggestions && suggestions.map((suggestion, idx) => (
          <div key={idx} className="flex items-start gap-2">
            <span className="text-blue-500 font-bold text-xs mt-0.5">{idx + 1}.</span>
            <p className="text-xs text-slate-600 font-medium leading-relaxed">{suggestion}</p>
          </div>
        ))}
      </div>
    </motion.div>
  );
};

// Recent Summary View
const RecentSummaryView = ({ dailySummary, detailedHistory }: { dailySummary: any[], detailedHistory: any[] }) => {
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [filterReason, setFilterReason] = useState<string>('All');

  // Get unique reasons for filter
  const uniqueReasons = useMemo(() => {
    const reasons = new Set<string>();
    detailedHistory.forEach((item: any) => reasons.add(item.reason || 'Unknown'));
    return ['All', ...Array.from(reasons)];
  }, [detailedHistory]);

  // Filtered detailed history
  const filteredHistory = useMemo(() => {
    if (!selectedDate) return [];
    return detailedHistory.filter((item: any) => {
      if (item.date !== selectedDate) return false;
      if (filterReason !== 'All' && item.reason !== filterReason) return false;
      return true;
    });
  }, [selectedDate, detailedHistory, filterReason]);

  // Handle click on list item
  const handleDateClick = (date: string) => {
    setSelectedDate(date);
    setFilterReason('All'); // Reset filter when switching days
  };

  return (
    <motion.div
      key="recent-summary"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">Recent Summary</h2>
          <p className="text-sm text-slate-500 mt-1">
            {dailySummary && dailySummary.length === 1
              ? "Today's traffic activity"
              : `Last ${dailySummary ? dailySummary.length : 0} days of traffic activity`}
          </p>
        </div>
        {selectedDate && (
          <button
            onClick={() => setSelectedDate(null)}
            className="flex items-center px-4 py-2 bg-slate-200 hover:bg-slate-300 rounded-xl text-slate-700 font-bold text-sm transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to List
          </button>
        )}
      </div>

      <AnimatePresence mode="wait">
        {!selectedDate ? (
          <motion.div
            key="list-view"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="grid gap-4"
          >
            {dailySummary && dailySummary.length > 0 ? (
              dailySummary.map((day: any, index: number) => (
                <div
                  key={day.date}
                  onClick={() => handleDateClick(day.date)}
                  className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 hover:shadow-md cursor-pointer transition-all hover:scale-[1.01] group"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-full bg-blue-50 flex items-center justify-center text-blue-600 font-bold group-hover:bg-blue-500 group-hover:text-white transition-colors">
                        {index + 1}
                      </div>
                      <div>
                        <h3 className="font-bold text-slate-800 text-lg flex items-center gap-2">
                          <Calendar className="w-4 h-4 text-slate-400" />
                          {day.date}
                        </h3>
                        <div className="flex gap-4 mt-1 text-sm text-slate-500">
                          <span>Events: <strong className="text-slate-700">{day.totalEvents}</strong></span>
                          <span>Peak Traffic: <strong className="text-slate-700">{day.peakVehicleCount} cars</strong></span>
                        </div>
                      </div>
                    </div>
                    <ChevronRight className="w-5 h-5 text-slate-300 group-hover:text-blue-500 transition-colors" />
                  </div>
                </div>
              ))
            ) : (
              <div className="p-8 text-center text-slate-400 bg-white rounded-2xl border border-slate-100 border-dashed">
                No summary data available.
              </div>
            )}
          </motion.div>
        ) : (
          <motion.div
            key="detail-view"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="bg-white rounded-[2rem] p-8 shadow-sm border border-slate-100"
          >
            <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 gap-4">
              <h3 className="text-xl font-bold text-slate-800 flex items-center gap-2">
                <FileText className="w-5 h-5 text-blue-500" />
                Details for {selectedDate}
              </h3>

              {/* Filters */}
              <div className="flex items-center gap-3">
                <div className="relative">
                  <Filter className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                  <select
                    value={filterReason}
                    onChange={(e) => setFilterReason(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-slate-50 rounded-xl text-sm font-semibold text-slate-700 border-none focus:ring-2 focus:ring-blue-500 outline-none"
                  >
                    {uniqueReasons.map(r => (
                      <option key={r} value={r}>{r}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-2">
              {filteredHistory.length > 0 ? (
                filteredHistory.map((item: any, idx: number) => (
                  <div key={idx} className="flex items-start p-4 rounded-xl bg-slate-50 hover:bg-slate-100 transition-colors">
                    <div className={`mt-1 w-2 h-2 rounded-full mr-4 flex-shrink-0 ${item.vehicleCount > 5 ? 'bg-red-500' : 'bg-green-500'}`}></div>
                    <div className="flex-1">
                      <div className="flex justify-between items-start">
                        <h4 className="font-bold text-slate-800 text-sm">{item.reason}</h4>
                        <span className="text-xs font-mono text-slate-400 bg-white px-2 py-1 rounded-md shadow-sm">{item.time}</span>
                      </div>
                      <p className="text-xs text-slate-500 mt-1">Vehicle Count: {item.vehicleCount}</p>
                      <p className="text-xs text-slate-400 mt-0.5">Status: {item.congestion_status}</p>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-10 text-slate-400">
                  No events found matching filters.
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

// Live Monitoring View with Time-Series Graph
const LiveMonitoringView = ({ graph, loading }: { graph: any[], loading: boolean }) => {
  // Filter for today's data to ensure the graph only shows relevant daily events
  const todayGraph = useMemo(() => {
    if (!graph || graph.length === 0) return [];

    const now = new Date();
    const day = String(now.getDate()).padStart(2, '0');
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const year = now.getFullYear();
    const todayStr = `${day}-${month}-${year}`;

    return graph.filter((item: any) => {
      const parts = item.timestamp?.split(',') || [];
      // Check if the date part matches today
      return parts.length >= 2 && parts[1] === todayStr;
    });
  }, [graph]);

  // Process graph data for the chart with 10-minute binning (more detailed)
  const chartData = useMemo(() => {
    if (!todayGraph || todayGraph.length === 0) return [];

    const BIN_MINUTES = 10;

    // Group data by time interval
    const groupedByBin: { [key: string]: { [reason: string]: number } } = {};

    todayGraph.forEach((item: any) => {
      const [timeStr] = item.timestamp?.split(',') || [];
      if (!timeStr) return;

      // Parse HH:MM:SS
      const parts = timeStr.split(':');
      let hours = 0, minutes = 0;

      if (parts.length >= 2) {
        hours = parseInt(parts[0]);
        minutes = parseInt(parts[1]);
      }

      // Calculate bin
      const totalMinutes = hours * 60 + minutes;
      const binIndex = Math.floor(totalMinutes / BIN_MINUTES);
      const binStartMinutes = binIndex * BIN_MINUTES;

      const binH = Math.floor(binStartMinutes / 60);
      const binM = binStartMinutes % 60;

      // Format: HH:MM
      const timeBinLabel = `${binH}:${binM.toString().padStart(2, '0')}`;

      if (!groupedByBin[timeBinLabel]) {
        groupedByBin[timeBinLabel] = {};
      }

      const reason = item.reason || 'Unknown';
      groupedByBin[timeBinLabel][reason] = (groupedByBin[timeBinLabel][reason] || 0) + 1;
    });

    // Convert to chart format and sort by time
    return Object.entries(groupedByBin).map(([time, reasons]) => ({
      time,
      ...reasons,
    })).sort((a, b) => {
      const [h1, m1] = a.time.split(':').map(Number);
      const [h2, m2] = b.time.split(':').map(Number);
      return h1 * 60 + m1 - (h2 * 60 + m2);
    });
  }, [todayGraph]);

  // Get unique reasons for the legend based on TODAY'S data
  const uniqueReasons = useMemo(() => {
    if (!todayGraph || todayGraph.length === 0) return [];
    return Array.from(new Set(todayGraph.map((item: any) => item.reason || 'Unknown')));
  }, [todayGraph]);

  // Colors for different congestion types
  const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899'];

  // Summary stats for TODAY
  const totalEvents = todayGraph.length;
  const reasonCounts = useMemo(() => {
    return todayGraph.reduce((acc: { [key: string]: number }, item: any) => {
      const reason = item.reason || 'Unknown';
      acc[reason] = (acc[reason] || 0) + 1;
      return acc;
    }, {});
  }, [todayGraph]);

  return (
    <motion.div
      key="live-monitoring"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="space-y-8"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
            <span className="text-xs font-bold uppercase tracking-[0.2em] text-green-600">Live Monitoring</span>
            {loading && <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />}
          </div>
          <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">Today's Congestion Timeline</h2>
          <p className="text-sm text-slate-500 mt-1">Real-time tracking of traffic congestion events</p>
        </div>
        <div className="text-right">
          <p className="text-4xl font-bold text-slate-900">{totalEvents}</p>
          <p className="text-xs text-slate-500 uppercase tracking-wide">Total Events</p>
        </div>
      </div>

      {/* Time-Series Chart */}
      <div className="bg-white rounded-[2rem] p-8 shadow-sm border border-slate-100">
        <h3 className="text-sm font-bold text-slate-900 mb-6 flex items-center">
          <Activity className="w-4 h-4 mr-2 text-blue-500" />
          Congestion Events Over Time
        </h3>

        {chartData.length > 0 ? (
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 10, fill: '#64748b' }}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis
                  tick={{ fontSize: 12, fill: '#64748b' }}
                  allowDecimals={false}
                  label={{ value: 'Count', angle: -90, position: 'insideLeft', style: { fontSize: 12, fill: '#64748b' } }}
                />
                <Tooltip
                  contentStyle={{
                    borderRadius: '12px',
                    border: 'none',
                    boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
                    padding: '12px'
                  }}
                />
                <Legend
                  wrapperStyle={{ paddingTop: '20px' }}
                  formatter={(value) => <span className="text-xs font-medium text-slate-600">{value}</span>}
                />
                {uniqueReasons.map((reason, index) => (
                  <Bar
                    key={reason}
                    dataKey={reason}
                    stackId="a"
                    fill={COLORS[index % COLORS.length]}
                    radius={index === uniqueReasons.length - 1 ? [4, 4, 0, 0] : [0, 0, 0, 0]}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-80 flex items-center justify-center text-slate-400">
            <p>No congestion data available for today</p>
          </div>
        )}
      </div>

      {/* Reason Breakdown Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Object.entries(reasonCounts).map(([reason, count], index) => (
          <motion.div
            key={reason}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 * index }}
            className="bg-white rounded-2xl p-4 shadow-sm border border-slate-100 hover:shadow-md transition-shadow"
          >
            <div
              className="w-3 h-3 rounded-full mb-3"
              style={{ backgroundColor: COLORS[index % COLORS.length] }}
            ></div>
            <p className="text-2xl font-bold text-slate-900">{count as number}</p>
            <p className="text-xs text-slate-500 font-medium truncate" title={reason}>{reason}</p>
          </motion.div>
        ))}
      </div>

      {/* Recent Events List */}
      <div className="bg-white rounded-[2rem] p-8 shadow-sm border border-slate-100">
        <h3 className="text-sm font-bold text-slate-900 mb-4 flex items-center">
          <Clock className="w-4 h-4 mr-2 text-slate-400" />
          Recent Congestion Events
        </h3>
        <div className="space-y-3 max-h-60 overflow-y-auto">
          {todayGraph?.slice(-10).reverse().map((item: any, index: number) => (
            <div key={index} className="flex items-center justify-between py-2 border-b border-slate-100 last:border-0">
              <div className="flex items-center gap-3">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: COLORS[uniqueReasons.indexOf(item.reason) % COLORS.length] }}
                ></div>
                <span className="text-sm font-medium text-slate-700">{item.reason}</span>
              </div>
              <span className="text-xs text-slate-400 font-mono">{item.timestamp?.split(',')[0]}</span>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
};

export const TrafficDashboard = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { vehicleCount, time, congestion, report, graph, daily_summary, detailed_history, currentView, loading, error } = useSelector((state: RootState) => state.traffic);
  const { toast } = useToast();

  // Data is now fetched automatically via useFirebaseData hook in App.tsx
  // Firebase provides real-time updates, no polling needed

  const handleStatClick = (type: string) => {
    toast({
      title: `${type} Details`,
      description: `Viewing detailed analytics for ${type}.`,
    });
  };

  const handleRefresh = () => {
    // Data refreshes automatically via Firebase real-time subscription
    toast({
      title: "Live Data",
      description: "Data updates automatically from Firebase.",
    });
  };

  return (
    <div className="flex h-screen bg-[#f2f3f5] overflow-hidden font-sans">
      <Sidebar />
      <MobileMenu />

      <main className="flex-1 flex flex-col h-full overflow-y-auto p-4 md:p-10 relative scroll-smooth">
        {/* Mobile Header */}
        <div className="md:hidden flex justify-between items-center mb-6 sticky top-0 bg-[#f2f3f5]/90 backdrop-blur-sm z-30 py-2">
          <h1 className="text-lg font-bold text-slate-800">Smart Traffic</h1>
          <button onClick={() => dispatch(toggleMobileMenu())} className="p-2 active:bg-slate-200 rounded-full">
            <Menu className="w-6 h-6 text-slate-600" />
          </button>
        </div>

        <div className="max-w-6xl mx-auto w-full">
          {/* Content Switcher based on View */}
          <AnimatePresence mode="wait">
            {currentView === 'dashboard' && (
              <motion.div
                key="dashboard"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <header className="mb-12 flex flex-col md:flex-row md:items-start justify-between gap-8">
                  <div className="pt-4">
                    <div className="flex items-center gap-3 mb-3">
                      <span className="text-xs font-bold uppercase tracking-[0.2em] text-slate-400">overview</span>
                      {loading && <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />}
                      <button
                        onClick={handleRefresh}
                        className="p-1 hover:bg-slate-200 rounded-full transition-colors"
                        title="Refresh data"
                      >
                        <RefreshCw className="w-4 h-4 text-slate-400 hover:text-slate-600" />
                      </button>
                    </div>
                    <h2 className="text-4xl font-extrabold text-slate-900 mb-3 tracking-tight">Analyze Your <br />Congestion</h2>
                    <p className="text-xs text-slate-400 font-bold tracking-wide uppercase">From Chaos To Control — Monitor, Analyze, Improve</p>
                    {error && (
                      <div className="mt-3 flex items-center gap-2 text-red-500 text-sm">
                        <AlertCircle className="w-4 h-4" />
                        <span>{error}</span>
                      </div>
                    )}
                  </div>

                  <div className="flex gap-5">
                    <div onClick={() => handleStatClick('Vehicle Count')}>
                      <StatCard
                        value={vehicleCount}
                        label="V-Count"
                        colorClass="bg-[#9bcab3]" /* Muted Green */
                        delay={0.1}
                        icon={Car}
                      />
                    </div>
                    <div onClick={() => handleStatClick('Time')}>
                      <StatCard
                        value={time}
                        label="Time"
                        colorClass="bg-[#ff4d4d]" /* Vibrant Red */
                        delay={0.2}
                        icon={Clock}
                      />
                    </div>
                  </div>
                </header>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                  {/* Left Column: Chart (Larger) */}
                  <div className="lg:col-span-7">
                    <motion.div
                      whileHover={{ scale: 1.01 }}
                      className="flex items-center justify-center min-h-[320px] bg-white/50 rounded-[2.5rem] p-4 border border-white/50 shadow-sm transition-all"
                    >
                      <CongestionChart data={congestion} />
                    </motion.div>
                  </div>

                  {/* Right Column: Report (Smaller card look) */}
                  <div className="lg:col-span-5 bg-white rounded-[2rem] p-8 shadow-sm border border-slate-100 h-full min-h-[300px] relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-400 to-purple-500"></div>
                    <div className="flex justify-between items-center mb-8">
                      <h3 className="text-sm font-bold text-slate-900 flex items-center">
                        <FileText className="w-4 h-4 mr-2 text-slate-400" />
                        Current Report
                      </h3>
                      <span className="text-[10px] bg-red-100 text-red-600 px-2 py-1 rounded-full font-bold">Live Alerts</span>
                    </div>

                    <div className="space-y-6">
                      {report.map((item: any, idx: number) => (
                        <ReportCard key={idx} index={idx} reason={item.reason} suggestions={item.suggestions || []} />
                      ))}
                    </div>
                  </div>
                </div>

                {/* Bottom Section - Additional Stats */}
                <div className="mt-12 bg-white rounded-[2rem] p-10 shadow-sm border border-slate-100 hover:shadow-md transition-shadow duration-300 cursor-default">
                  <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-6">Quick, High-Level Stats:</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                    <ul className="space-y-4 text-sm text-slate-600 font-medium">
                      <li className="flex items-center group">
                        <span className="w-1.5 h-1.5 rounded-full bg-slate-800 mr-3 group-hover:scale-150 transition-transform bg-blue-500"></span>
                        Total Congestion Events (E.G. 128)
                      </li>
                      <li className="flex items-center group">
                        <span className="w-1.5 h-1.5 rounded-full bg-slate-800 mr-3 group-hover:scale-150 transition-transform bg-red-500"></span>
                        Most Common Reason (E.G. Wrong Parking)
                      </li>
                    </ul>
                    <ul className="space-y-4 text-sm text-slate-600 font-medium">
                      <li className="flex items-center group">
                        <span className="w-1.5 h-1.5 rounded-full bg-slate-800 mr-3 group-hover:scale-150 transition-transform bg-orange-500"></span>
                        Average Jam Duration (E.G. 6.4 Mins)
                      </li>
                      <li className="flex items-center group">
                        <span className="w-1.5 h-1.5 rounded-full bg-slate-800 mr-3 group-hover:scale-150 transition-transform bg-purple-500"></span>
                        Peak Congestion Time (E.G. 6:00 PM - 7:30 PM)
                      </li>
                    </ul>
                  </div>
                </div>
              </motion.div>
            )}

            {currentView === 'live' && (
              <LiveMonitoringView graph={graph} loading={loading} />
            )}

            {currentView === 'summary' && (
              <RecentSummaryView dailySummary={daily_summary} detailedHistory={detailed_history} />
            )}

            {currentView === 'settings' && (
              <motion.div
                key="settings"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="flex flex-col items-center justify-center h-[60vh] text-center"
              >
                <div className="w-24 h-24 bg-slate-200 rounded-full flex items-center justify-center mb-6 animate-pulse">
                  <Settings className="w-10 h-10 text-slate-400" />
                </div>
                <h2 className="text-2xl font-bold text-slate-800 mb-2 capitalize">
                  Settings View
                </h2>
                <p className="text-slate-500 max-w-md">
                  This module is currently being simulated. In a production environment, this would display real-time settings.
                </p>
                <button
                  onClick={() => dispatch(setView('dashboard'))}
                  className="mt-8 px-6 py-2 bg-slate-800 text-white rounded-xl font-bold text-sm hover:bg-slate-700 transition-colors shadow-lg hover:shadow-xl"
                >
                  Return to Dashboard
                </button>
              </motion.div>
            )}

            {currentView === 'map' && (
              <motion.div
                key="map"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="flex flex-col items-center justify-center h-[60vh] text-center"
              >
                <div className="w-24 h-24 bg-slate-200 rounded-full flex items-center justify-center mb-6 animate-pulse">
                  <MapPin className="w-10 h-10 text-slate-400" />
                </div>
                <h2 className="text-2xl font-bold text-slate-800 mb-2 capitalize">
                  Live Map View
                </h2>
                <p className="text-slate-500 max-w-md">
                  This module is currently being simulated. In a production environment, this would display real-time map data.
                </p>
                <button
                  onClick={() => dispatch(setView('dashboard'))}
                  className="mt-8 px-6 py-2 bg-slate-800 text-white rounded-xl font-bold text-sm hover:bg-slate-700 transition-colors shadow-lg hover:shadow-xl"
                >
                  Return to Dashboard
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
};
