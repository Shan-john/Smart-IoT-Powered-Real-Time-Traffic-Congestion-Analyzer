import React, { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState, setView, toggleMobileMenu, closeMobileMenu, updateTime } from '../store';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { LayoutDashboard, Activity, FileText, Settings, Menu, X, MapPin, Clock, Car } from 'lucide-react';
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
          <img 
            src="https://via.placeholder.com/400x400?text=Map+Preview"
            alt="Traffic Map Placeholder"
            className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity duration-500 hover:scale-110"
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

const ReportCard = ({ reason, suggestion, index }: { reason: string, suggestion: string, index: number }) => {
  return (
    <motion.div 
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay: 0.2 + (index * 0.1) }}
      className="mb-6 last:mb-0 group cursor-default"
      data-testid={`report-item-${index}`}
    >
      <div className="flex items-center mb-2">
        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wide">Reason And Suggestion</h4>
        <div className="ml-2 h-px bg-slate-200 flex-1 group-hover:bg-slate-300 transition-colors"></div>
      </div>
      <div className="mb-2">
        <p className="text-sm font-semibold text-slate-800 group-hover:text-blue-600 transition-colors">{reason}</p>
      </div>
      <p className="text-xs text-slate-500 font-medium leading-relaxed bg-slate-50 p-3 rounded-lg border border-transparent group-hover:border-slate-100 transition-all">
        {suggestion}
      </p>
    </motion.div>
  );
};

export const TrafficDashboard = () => {
  const dispatch = useDispatch();
  const { vehicleCount, time, congestion, report, currentView } = useSelector((state: RootState) => state.traffic);
  const { toast } = useToast();

  // Simulate real-time clock update
  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date();
      const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      // Only update if the minute changes to avoid too many dispatches
      // dispatch(updateTime(timeString)); 
    }, 60000);
    
    return () => clearInterval(interval);
  }, [dispatch]);

  const handleStatClick = (type: string) => {
    toast({
      title: `${type} Details`,
      description: `Viewing detailed analytics for ${type}.`,
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
                    <span className="text-xs font-bold uppercase tracking-[0.2em] text-slate-400 mb-3 block">overview</span>
                    <h2 className="text-4xl font-extrabold text-slate-900 mb-3 tracking-tight">Analyze Your <br/>Congestion</h2>
                    <p className="text-xs text-slate-400 font-bold tracking-wide uppercase">From Chaos To Control — Monitor, Analyze, Improve</p>
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
                        <ReportCard key={idx} index={idx} reason={item.reason} suggestion={item.suggestion} />
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

            {currentView !== 'dashboard' && (
              <motion.div
                key="other-views"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="flex flex-col items-center justify-center h-[60vh] text-center"
              >
                <div className="w-24 h-24 bg-slate-200 rounded-full flex items-center justify-center mb-6 animate-pulse">
                  {currentView === 'live' && <Activity className="w-10 h-10 text-slate-400" />}
                  {currentView === 'summary' && <FileText className="w-10 h-10 text-slate-400" />}
                  {currentView === 'settings' && <Settings className="w-10 h-10 text-slate-400" />}
                  {currentView === 'map' && <MapPin className="w-10 h-10 text-slate-400" />}
                </div>
                <h2 className="text-2xl font-bold text-slate-800 mb-2 capitalize">
                  {currentView === 'map' ? 'Live Map View' : `${currentView} View`}
                </h2>
                <p className="text-slate-500 max-w-md">
                  This module is currently being simulated. In a production environment, this would display real-time {currentView} data streams.
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
