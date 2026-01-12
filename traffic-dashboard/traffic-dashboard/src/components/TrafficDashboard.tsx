import React, { useState, useMemo } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState, AppDispatch, setView, toggleMobileMenu, closeMobileMenu } from '../store';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, LineChart, Line, XAxis, YAxis, CartesianGrid, Legend, BarChart, Bar } from 'recharts';
import { LayoutDashboard, Activity, FileText, Settings, Menu, X, MapPin, Clock, Car, Loader2, AlertCircle, RefreshCw, ChevronRight, ChevronLeft, ArrowLeft, Filter, Calendar, Search, Shield, Heart, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useToast } from '../hooks/use-toast';
import { AdminPanel } from './AdminPanel';

// --- 🌸 Anime Character Mascot Component ---
const AnimeMascot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [currentTip, setCurrentTip] = useState(0);

  const tips = [
    { text: "Welcome to the dashboard, senpai! 💕", emoji: "🎀" },
    { text: "Click on me for traffic tips, nya~!", emoji: "🐱" },
    { text: "Drive safely and watch for congestion! ✨", emoji: "🚗" },
    { text: "The live monitoring shows real-time data!", emoji: "📊" },
    { text: "Check the summary for daily reports~", emoji: "📋" },
  ];

  const nextTip = () => {
    setCurrentTip((prev) => (prev + 1) % tips.length);
  };

  return (
    <div className="fixed bottom-6 right-6 z-50">
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.8 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.8 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            className="absolute bottom-24 right-0 w-72 bg-white/95 backdrop-blur-xl rounded-3xl shadow-2xl border-2 border-pink-200 p-5 kawaii-shadow"
          >
            {/* Close button */}
            <button
              onClick={() => setIsOpen(false)}
              className="absolute top-3 right-3 w-6 h-6 bg-pink-100 hover:bg-pink-200 rounded-full flex items-center justify-center text-pink-500 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>

            {/* Header */}
            <div className="flex items-center gap-3 mb-4 pb-3 border-b border-pink-100">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-pink-400 to-purple-400 flex items-center justify-center text-lg shadow-lg">
                🌸
              </div>
              <div>
                <h3 className="font-extrabold text-purple-800 text-sm">Traffic-chan</h3>
                <p className="text-[10px] text-pink-400">Your kawaii assistant! ✨</p>
              </div>
            </div>

            {/* Tip content */}
            <motion.div
              key={currentTip}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-gradient-to-r from-pink-50 to-purple-50 rounded-2xl p-4 mb-4"
            >
              <div className="flex items-start gap-3">
                <span className="text-2xl animate-bounce-cute">{tips[currentTip].emoji}</span>
                <p className="text-sm text-purple-700 font-medium leading-relaxed">
                  {tips[currentTip].text}
                </p>
              </div>
            </motion.div>

            {/* Actions */}
            <div className="flex gap-2">
              <button
                onClick={nextTip}
                className="flex-1 py-2 px-4 bg-gradient-to-r from-pink-400 to-purple-400 text-white rounded-xl font-bold text-xs hover:from-pink-500 hover:to-purple-500 transition-all shadow-lg hover:shadow-xl kawaii-btn"
              >
                Next Tip ✨
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className="py-2 px-4 bg-pink-100 text-pink-600 rounded-xl font-bold text-xs hover:bg-pink-200 transition-colors"
              >
                Close
              </button>
            </div>

            {/* Decorative elements */}
            <div className="absolute -top-2 -right-2 text-lg animate-sparkle">✦</div>
            <div className="absolute -bottom-1 -left-1 text-sm animate-twinkle">💕</div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Mascot Character Button */}
      <motion.button
        onClick={() => setIsOpen(!isOpen)}
        whileHover={{ scale: 1.1, rotate: 5 }}
        whileTap={{ scale: 0.95 }}
        className="relative w-20 h-20 rounded-full bg-gradient-to-br from-pink-400 via-purple-400 to-pink-500 shadow-2xl flex items-center justify-center cursor-pointer border-4 border-white kawaii-shadow group"
        initial={{ scale: 0, rotate: -180 }}
        animate={{ scale: 1, rotate: 0 }}
        transition={{ type: "spring", stiffness: 260, damping: 20, delay: 0.5 }}
      >
        {/* Anime character face */}
        <div className="relative">
          <motion.div
            animate={isOpen ? { scale: 1.1 } : { scale: 1 }}
            transition={{ repeat: Infinity, repeatType: "reverse", duration: 0.8 }}
            className="text-4xl"
          >
            {isOpen ? "😊" : "🌸"}
          </motion.div>

          {/* Sparkle decorations */}
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ repeat: Infinity, duration: 8, ease: "linear" }}
            className="absolute -top-2 -right-2"
          >
            <Sparkles className="w-4 h-4 text-yellow-300 drop-shadow-lg" />
          </motion.div>
        </div>

        {/* Pulsing ring effect */}
        <motion.div
          className="absolute inset-0 rounded-full border-2 border-pink-300"
          animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0, 0.5] }}
          transition={{ repeat: Infinity, duration: 2 }}
        />

        {/* Notification dot when closed */}
        {!isOpen && (
          <motion.div
            className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center text-white text-[10px] font-bold border-2 border-white"
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ repeat: Infinity, duration: 1 }}
          >
            !
          </motion.div>
        )}

        {/* Floating hearts on hover */}
        <div className="absolute -top-4 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
          <motion.span
            animate={{ y: [-5, -15, -5] }}
            transition={{ repeat: Infinity, duration: 1.5 }}
            className="text-sm"
          >
            💕
          </motion.span>
        </div>
      </motion.button>

      {/* Speech bubble when not open */}
      {!isOpen && (
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 1.5 }}
          className="absolute bottom-16 right-20 bg-white rounded-xl px-3 py-2 shadow-lg border border-pink-200 text-xs text-purple-700 font-medium whitespace-nowrap"
        >
          Click me! ✨
          <div className="absolute bottom-2 -right-2 w-3 h-3 bg-white border-r border-b border-pink-200 transform rotate-45"></div>
        </motion.div>
      )}
    </div>
  );
};

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
    <div className="w-64 h-full bg-gradient-to-b from-pink-300 via-purple-300 to-pink-400 text-white flex flex-col p-6 hidden md:flex shadow-xl relative overflow-hidden z-20" data-testid="sidebar">
      {/* ✨ Floating decorative elements */}
      <div className="absolute top-4 right-4 text-white/40 text-lg animate-sparkle">✦</div>
      <div className="absolute top-20 right-8 text-white/30 text-sm animate-twinkle" style={{ animationDelay: '0.5s' }}>✧</div>
      <div className="absolute bottom-40 left-4 text-white/25 text-xs animate-sparkle" style={{ animationDelay: '1s' }}>✦</div>

      {/* 💖 Floating hearts */}
      <div className="absolute top-32 right-3 text-lg opacity-30 animate-float" style={{ animationDelay: '0.3s' }}>💗</div>
      <div className="absolute top-48 left-2 text-sm opacity-25 animate-float" style={{ animationDelay: '1.5s' }}>💕</div>

      {/* 🌸 Floating petals */}
      <div className="absolute top-16 left-8 text-lg opacity-20 animate-sway">🌸</div>
      <div className="absolute bottom-60 right-6 text-sm opacity-15 animate-sway" style={{ animationDelay: '2s' }}>🌼</div>

      {/* ⭐ Twinkling stars */}
      <div className="absolute top-8 left-12 text-xs text-yellow-200/40 animate-twinkle">⭐</div>
      <div className="absolute bottom-32 right-10 text-xs text-yellow-200/30 animate-twinkle" style={{ animationDelay: '1s' }}>✨</div>

      <div className="mb-8 cursor-pointer group" onClick={() => dispatch(setView('dashboard'))}>
        <h1 className="text-sm font-extrabold font-sans tracking-wide drop-shadow-sm flex items-center gap-2">
          <span className="animate-wiggle inline-block">🌸</span> Smart Traffic Monitor
        </h1>
        <p className="text-[10px] text-white/60 mt-1 font-medium">✨ Kawaii Dashboard ✨</p>
      </div>

      <div className="flex-1 flex flex-col">
        <div
          className={`mb-6 flex items-center cursor-pointer hover:bg-white/20 rounded-xl px-3 py-2 transition-all duration-300 ${currentView === 'dashboard' ? 'bg-white/25 text-white font-bold shadow-lg' : 'text-white/90 hover:text-white'}`}
          onClick={() => dispatch(setView('dashboard'))}
        >
          <LayoutDashboard className="w-5 h-5 mr-3" />
          <span className="text-sm font-semibold">Dashboard</span>
        </div>

        {/* Toggle-like Menu - Kawaii Style */}
        <div className="bg-white/20 backdrop-blur-sm rounded-2xl p-2 mb-6 flex flex-col gap-1.5 shadow-inner">
          <div
            className={`rounded-xl px-4 py-2.5 text-xs font-bold text-center cursor-pointer transition-all duration-300 flex items-center justify-center gap-2 kawaii-btn
              ${currentView === 'live' ? 'bg-white text-pink-600 scale-105 shadow-lg kawaii-shadow' : 'text-white/80 hover:bg-white/15 hover:text-white'}`}
            onClick={() => dispatch(setView('live'))}
          >
            {currentView === 'live' && <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse shadow-lg shadow-green-400/50"></span>}
            <span className="animate-bounce-cute inline-block">✨</span> live monitoring
          </div>
          <div
            className={`rounded-xl px-4 py-2.5 text-xs font-bold text-center cursor-pointer transition-all duration-300 kawaii-btn
              ${currentView === 'summary' ? 'bg-white text-pink-600 scale-105 shadow-lg kawaii-shadow' : 'text-white/80 hover:bg-white/15 hover:text-white'}`}
            onClick={() => dispatch(setView('summary'))}
          >
            📋 recent summary
          </div>
        </div>

        {/* 🐾 Cute Mascot Section */}
        <div className="mb-4 p-3 bg-white/15 rounded-2xl backdrop-blur-sm border border-white/20">
          <div className="flex items-center gap-2">
            <span className="text-2xl animate-bounce-cute">🐱</span>
            <div>
              <p className="text-[10px] font-bold text-white">Traffic-chan says:</p>
              <p className="text-[9px] text-white/80 italic">"Drive safely, nya~!" 💗</p>
            </div>
          </div>
        </div>

        {/* Admin Link - Kawaii Style */}
        <div
          className={`mb-auto flex items-center cursor-pointer hover:bg-white/20 rounded-xl px-3 py-2 transition-all duration-300 ${currentView === 'admin' ? 'bg-white/25 text-white font-bold shadow-lg' : 'text-white/90 hover:text-white'}`}
          onClick={() => dispatch(setView('admin'))}
        >
          <Shield className="w-5 h-5 mr-3" />
          <span className="text-sm font-semibold">🔐 Admin Panel</span>
        </div>

        {/* Map Section at Bottom — Kawaii Style */}
        <div
          className="mt-auto w-full rounded-3xl overflow-hidden relative aspect-square border-2 border-white/40 group cursor-pointer shadow-lg hover:shadow-2xl transition-all duration-300 hover:-translate-y-1 kawaii-glow"
          onClick={() => dispatch(setView('map'))}
        >
          <iframe
            className="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity duration-500"
            src="https://www.google.com/maps/embed?pb=!1m17!1m12!1m3!1d4494.924008878361!2d76.90539207501351!3d8.57107499147314!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m2!1m1!2zOMKwMzQnMTUuOSJOIDc2wrA1NCcyOC43IkU!5e1!3m2!1sen!2sin!4v1767719153017!5m2!1sen!2sin"
            title="Live Traffic Map"
            loading="lazy"
            style={{ border: 0 }}
            allowFullScreen
            referrerPolicy="no-referrer-when-downgrade"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-purple-500/70 via-pink-400/30 to-transparent flex items-end p-4">
            <div className="flex items-center text-white drop-shadow-lg">
              <MapPin className="w-3 h-3 mr-1.5" />
              <span className="text-xs font-bold">📍 View Live Map</span>
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
            className="fixed inset-y-0 left-0 w-[80%] max-w-xs bg-gradient-to-b from-pink-300 via-purple-300 to-pink-400 z-50 p-6 flex flex-col shadow-2xl md:hidden"
          >
            <div className="flex justify-between items-center mb-10 text-white">
              <h1 className="text-lg font-extrabold flex items-center gap-2">🌸 Smart Traffic</h1>
              <button onClick={() => dispatch(closeMobileMenu())} className="p-2 hover:bg-white/20 rounded-full transition-colors">
                <X className="w-6 h-6" />
              </button>
            </div>

            <nav className="space-y-3">
              <div
                className={`p-3 rounded-xl flex items-center cursor-pointer transition-all duration-300 ${currentView === 'dashboard' ? 'bg-white text-pink-600 shadow-lg kawaii-shadow' : 'text-white hover:bg-white/20'}`}
                onClick={() => handleNav('dashboard')}
              >
                <LayoutDashboard className="w-5 h-5 mr-3" />
                <span className="font-bold">Dashboard</span>
              </div>
              <div
                className={`p-3 rounded-xl flex items-center cursor-pointer transition-all duration-300 ${currentView === 'live' ? 'bg-white text-pink-600 shadow-lg kawaii-shadow' : 'text-white hover:bg-white/20'}`}
                onClick={() => handleNav('live')}
              >
                <Activity className="w-5 h-5 mr-3" />
                <span className="font-bold">✨ Live Monitoring</span>
              </div>
              <div
                className={`p-3 rounded-xl flex items-center cursor-pointer transition-all duration-300 ${currentView === 'summary' ? 'bg-white text-pink-600 shadow-lg kawaii-shadow' : 'text-white hover:bg-white/20'}`}
                onClick={() => handleNav('summary')}
              >
                <FileText className="w-5 h-5 mr-3" />
                <span className="font-bold">📋 Recent Summary</span>
              </div>
              <div
                className={`p-3 rounded-xl flex items-center cursor-pointer transition-all duration-300 ${currentView === 'admin' ? 'bg-white text-pink-600 shadow-lg kawaii-shadow' : 'text-white hover:bg-white/20'}`}
                onClick={() => handleNav('admin')}
              >
                <Shield className="w-5 h-5 mr-3" />
                <span className="font-bold">🔐 Admin Panel</span>
              </div>
            </nav>

            <div className="mt-auto">
              <div className="p-4 bg-white/20 backdrop-blur-sm rounded-2xl border border-white/30">
                <p className="text-xs text-white/70 mb-2 font-semibold">✨ Current Status</p>
                <div className="flex items-center text-green-300 text-sm font-bold">
                  <span className="w-2.5 h-2.5 bg-green-400 rounded-full mr-2 animate-pulse shadow-lg shadow-green-400/50"></span>
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
      initial={{ opacity: 0, y: 20, scale: 0.8 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, delay, type: "spring" }}
      whileHover={{ scale: 1.08, y: -8, rotate: 1 }}
      className={`rounded-3xl p-6 w-36 h-44 flex flex-col justify-center items-center shadow-xl ${colorClass} cursor-pointer relative overflow-visible group kawaii-shadow`}
      data-testid={`stat-card-${label.toLowerCase()}`}
    >
      {/* 🚗 Anime car decoration on hover */}
      <div className="absolute -top-6 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-all duration-300 transform group-hover:-translate-y-2">
        <span className="text-2xl animate-bounce-cute">🚗</span>
        <span className="text-sm absolute -right-4 top-0 animate-float">💨</span>
      </div>

      {/* Kawaii decorations */}
      <div className="absolute -right-4 -top-4 w-20 h-20 rounded-full bg-white/25 blur-2xl group-hover:bg-white/40 transition-colors"></div>
      <div className="absolute -left-2 -bottom-2 w-12 h-12 rounded-full bg-white/20 blur-xl"></div>

      {/* Sparkle decorations */}
      <div className="absolute top-3 right-4 text-white/50 text-sm animate-sparkle">✦</div>
      <div className="absolute bottom-4 left-3 text-white/40 text-xs animate-twinkle" style={{ animationDelay: '0.5s' }}>✧</div>

      {Icon && <Icon className="w-7 h-7 text-white/80 mb-2 drop-shadow-lg group-hover:animate-wiggle" />}
      <span className="text-5xl font-extrabold text-white mb-2 tracking-tight relative z-10 drop-shadow-lg">{value}</span>
      <span className="text-[10px] uppercase tracking-widest font-bold text-white/90 relative z-10">{label}</span>

      {/* Traffic light indicator */}
      <div className="absolute bottom-2 right-2 w-2 h-2 rounded-full traffic-light-indicator"></div>
    </motion.div>
  );
};

const CongestionChart = ({ data }: { data: any[] }) => {
  // 🌸 Kawaii Pastel Colors
  const COLORS = ['#f9a8d4', '#c4b5fd', '#93c5fd', '#86efac', '#fcd34d', '#fda4af'];
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

      {/* Custom Legend - Scrollable */}
      <div className="flex flex-col space-y-2 pl-4 max-h-56 overflow-y-auto w-1/2">
        {data.map((entry: any, index: number) => (
          <motion.div
            key={index}
            className={`flex items-center text-xs cursor-pointer p-2 rounded-lg transition-colors ${activeIndex === index ? 'bg-slate-100' : ''}`}
            onMouseEnter={() => setActiveIndex(index)}
            onMouseLeave={() => setActiveIndex(null)}
            title={entry.name}
          >
            <div className="w-3 h-3 rounded-[2px] mr-2 flex-shrink-0 shadow-sm" style={{ backgroundColor: COLORS[index % COLORS.length] }}></div>
            <span className="text-slate-500 font-medium truncate max-w-[120px]">{entry.name}</span>
            <span className="ml-auto font-mono text-slate-400 text-[10px] pl-2 flex-shrink-0">{entry.percentage}%</span>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

const ReportCard = ({ reason, suggestions, index }: { reason: string, suggestions: string[], index: number }) => {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, x: 10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: 0.1 * index }}
      className="relative group"
      data-testid={`report-item-${index}`}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      {/* Compact Reason Display - Kawaii Style */}
      <div className="flex items-center gap-2 p-3 rounded-xl hover:bg-pink-50 cursor-pointer transition-all duration-200 border border-transparent hover:border-pink-200">
        <span className="w-2.5 h-2.5 rounded-full bg-gradient-to-br from-pink-400 to-purple-400 flex-shrink-0 shadow-sm"></span>
        <p className="text-sm font-semibold text-purple-800 truncate flex-1" title={reason}>
          {reason.length > 40 ? reason.substring(0, 40) + '...' : reason}
        </p>
        <span className="text-xs text-pink-400 opacity-0 group-hover:opacity-100 transition-opacity font-medium">
          ✨ hover for tips
        </span>
      </div>

      {/* Tooltip Popup - Kawaii Style */}
      <AnimatePresence>
        {showTooltip && suggestions && suggestions.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 5, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 5, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute left-0 right-0 top-full mt-1 z-50 bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl border border-pink-200 p-4 kawaii-shadow"
          >
            <div className="absolute -top-2 left-6 w-4 h-4 bg-white border-l border-t border-pink-200 transform rotate-45"></div>
            <h4 className="text-xs font-bold text-purple-700 uppercase tracking-wide mb-3 flex items-center gap-1">💡 Suggestions</h4>
            <div className="space-y-2">
              {suggestions.map((suggestion, idx) => (
                <div key={idx} className="flex items-start gap-2">
                  <span className="text-pink-500 text-xs mt-0.5">✔</span>
                  <p className="text-xs text-purple-600 leading-relaxed">{suggestion}</p>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
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
          <h2 className="text-3xl font-extrabold text-purple-900 tracking-tight">Recent Summary 🌸</h2>
          <p className="text-sm text-purple-400 mt-1">
            {dailySummary && dailySummary.length === 1
              ? "✨ Today's traffic activity"
              : `✨ Last ${dailySummary ? dailySummary.length : 0} days of traffic activity`}
          </p>
        </div>
        {selectedDate && (
          <button
            onClick={() => setSelectedDate(null)}
            className="flex items-center px-4 py-2 bg-gradient-to-r from-pink-400 to-purple-400 hover:from-pink-500 hover:to-purple-500 rounded-xl text-white font-bold text-sm transition-all shadow-lg hover:shadow-xl"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            ← Back to List
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
                  className="bg-white/80 backdrop-blur-sm p-6 rounded-2xl shadow-lg border border-pink-200/50 hover:shadow-xl cursor-pointer transition-all hover:scale-[1.01] group kawaii-shadow"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-pink-400 to-purple-400 flex items-center justify-center text-white font-bold shadow-lg group-hover:scale-110 transition-transform">
                        {index + 1}
                      </div>
                      <div>
                        <h3 className="font-bold text-purple-800 text-lg flex items-center gap-2">
                          <Calendar className="w-4 h-4 text-pink-400" />
                          {day.date}
                        </h3>
                        <div className="flex gap-4 mt-1 text-sm text-purple-500">
                          <span>Events: <strong className="text-purple-700">{day.totalEvents}</strong></span>
                          <span>Peak: <strong className="text-purple-700">{day.peakVehicleCount}</strong></span>
                        </div>
                        {(day.avgSpeed || day.avgConfidence || day.avgStuckRatio) && (
                          <div className="flex gap-3 mt-2 text-xs">
                            {day.avgSpeed !== undefined && day.avgSpeed > 0 && (
                              <span className="bg-green-100 text-green-600 px-2 py-0.5 rounded-full font-medium">⚡ {day.avgSpeed.toFixed(1)} px/s</span>
                            )}
                            {day.avgStuckRatio !== undefined && day.avgStuckRatio > 0 && (
                              <span className="bg-amber-100 text-amber-600 px-2 py-0.5 rounded-full font-medium">🛑 {(day.avgStuckRatio * 100).toFixed(0)}%</span>
                            )}
                            {day.avgConfidence !== undefined && day.avgConfidence > 0 && (
                              <span className="bg-purple-100 text-purple-600 px-2 py-0.5 rounded-full font-medium">📊 {(day.avgConfidence * 100).toFixed(0)}%</span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                    <ChevronRight className="w-5 h-5 text-pink-300 group-hover:text-pink-500 group-hover:translate-x-1 transition-all" />
                  </div>
                </div>
              ))
            ) : (
              <div className="p-8 text-center text-purple-400 bg-white/80 backdrop-blur-sm rounded-2xl border border-pink-200 border-dashed">
                ✨ No summary data available.
              </div>
            )}
          </motion.div>
        ) : (
          <motion.div
            key="detail-view"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="bg-white/80 backdrop-blur-sm rounded-[2rem] p-8 shadow-lg border border-pink-200/50 kawaii-shadow"
          >
            <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 gap-4">
              <h3 className="text-xl font-bold text-purple-800 flex items-center gap-2">
                <FileText className="w-5 h-5 text-pink-500" />
                📋 Details for {selectedDate}
              </h3>

              {/* Filters - Kawaii Style */}
              <div className="flex items-center gap-3">
                <div className="relative">
                  <Filter className="w-4 h-4 text-pink-400 absolute left-3 top-1/2 -translate-y-1/2" />
                  <select
                    value={filterReason}
                    onChange={(e) => setFilterReason(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-pink-50 rounded-xl text-sm font-semibold text-purple-700 border border-pink-200 focus:ring-2 focus:ring-pink-400 outline-none cursor-pointer"
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
                  <div key={idx} className="flex items-start p-4 rounded-xl bg-pink-50/50 hover:bg-pink-100/50 transition-colors border border-pink-100">
                    <div className={`mt-1 w-2.5 h-2.5 rounded-full mr-4 flex-shrink-0 shadow-sm ${item.vehicleCount > 5 ? 'bg-rose-400' : 'bg-green-400'}`}></div>
                    <div className="flex-1">
                      <div className="flex justify-between items-start">
                        <h4 className="font-bold text-purple-800 text-sm">{item.reason}</h4>
                        <span className="text-xs font-mono text-purple-400 bg-white/80 px-2 py-1 rounded-lg border border-pink-100">{item.time}</span>
                      </div>
                      <div className="flex flex-wrap gap-3 mt-2">
                        <span className="text-xs text-purple-500 font-medium">🚗 {item.vehicleCount} vehicles</span>
                        {item.average_speed !== undefined && (
                          <span className="text-xs text-green-600 font-medium">⚡ {item.average_speed.toFixed(1)} px/s</span>
                        )}
                        {item.stuck_ratio !== undefined && (
                          <span className="text-xs text-amber-600 font-medium">🛑 {(item.stuck_ratio * 100).toFixed(0)}% stuck</span>
                        )}
                        {item.confidence !== undefined && (
                          <span className="text-xs text-purple-600 font-medium">📊 {(item.confidence * 100).toFixed(0)}% conf</span>
                        )}
                      </div>
                      <p className="text-xs text-pink-400 mt-1 font-medium">Status: {item.congestion_status}</p>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-10 text-purple-400">
                  ✨ No events found matching filters.
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
const LiveMonitoringView = ({ graph, loading, stats }: { graph: any[], loading: boolean, stats: any }) => {
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
  // 🌸 Kawaii Pastel Colors for Bar Chart
  const COLORS = ['#f9a8d4', '#c4b5fd', '#93c5fd', '#a7f3d0', '#fde68a', '#fda4af', '#d8b4fe', '#67e8f9'];

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
      className="space-y-8 relative"
    >
      {/* 🚗 Floating anime cars decoration */}
      <div className="absolute -top-4 left-20 text-3xl animate-car-drive z-10 pointer-events-none">🚗</div>
      <div className="absolute -top-2 right-32 text-2xl animate-float pointer-events-none" style={{ animationDelay: '1s' }}>🚙</div>
      <div className="absolute top-20 right-10 text-xl animate-float pointer-events-none" style={{ animationDelay: '2s' }}>🏎️</div>

      {/* Header with Anime Character - Kawaii Style */}
      <div className="flex items-center justify-between relative">
        <div className="flex items-center gap-6">
          {/* Anime Traffic Girl Mascot */}
          <motion.div
            className="hidden md:block"
            animate={{ y: [0, -5, 0] }}
            transition={{ repeat: Infinity, duration: 2 }}
          >
            <img
              src="/anime_traffic_girl.png"
              alt="Traffic-chan"
              className="w-24 h-24 rounded-full border-4 border-white shadow-xl object-cover kawaii-shadow"
            />
          </motion.div>
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span className="w-2.5 h-2.5 bg-green-400 rounded-full animate-pulse shadow-lg shadow-green-400/50"></span>
              <span className="text-xs font-bold uppercase tracking-[0.2em] text-pink-500">✨ Live Monitoring</span>
              {loading && <Loader2 className="w-4 h-4 text-pink-500 animate-spin" />}
            </div>
            <h2 className="text-3xl font-extrabold text-purple-900 tracking-tight">Today's Congestion Timeline 🌸</h2>
            <p className="text-sm text-purple-400 mt-1">Real-time tracking of traffic congestion events</p>
          </div>
        </div>
        <div className="text-right bg-white/70 backdrop-blur-sm rounded-2xl p-4 border border-pink-200/50 kawaii-shadow relative overflow-visible">
          {/* Floating anime car on top */}

          <p className="text-4xl font-extrabold text-pink-500">{totalEvents}</p>
          <p className="text-xs text-purple-400 uppercase tracking-wide font-semibold">Today's Events</p>
        </div>
      </div>

      {/* Real-Time Metrics Grid with Anime Car Icons - Kawaii Style */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-pink-50 to-pink-100 rounded-2xl p-4 border border-pink-200 hover:scale-105 transition-transform cursor-pointer kawaii-shadow relative overflow-visible group">
          <div className="absolute -top-3 -right-2 text-lg opacity-0 group-hover:opacity-100 transition-opacity animate-bounce-cute">🚗</div>
          <p className="text-3xl font-extrabold text-pink-600">{stats?.total_events || 0}</p>
          <p className="text-xs text-pink-500 font-semibold mt-1">Total Events</p>
        </div>
        <div className="bg-gradient-to-br from-green-50 to-emerald-100 rounded-2xl p-4 border border-green-200 hover:scale-105 transition-transform cursor-pointer kawaii-shadow relative overflow-visible group">
          <div className="absolute -top-3 -right-2 text-lg opacity-0 group-hover:opacity-100 transition-opacity animate-bounce-cute">🏎️</div>
          <p className="text-3xl font-extrabold text-emerald-600">{stats?.avg_speed?.toFixed(1) || '0.0'}</p>
          <p className="text-xs text-emerald-500 font-semibold mt-1">Avg Speed (px/s)</p>
        </div>
        <div className="bg-gradient-to-br from-amber-50 to-orange-100 rounded-2xl p-4 border border-amber-200 hover:scale-105 transition-transform cursor-pointer kawaii-shadow relative overflow-visible group">
          <div className="absolute -top-3 -right-2 text-lg opacity-0 group-hover:opacity-100 transition-opacity animate-bounce-cute">🚧</div>
          <p className="text-3xl font-extrabold text-amber-600">{((stats?.avg_stuck_ratio || 0) * 100).toFixed(0)}%</p>
          <p className="text-xs text-amber-500 font-semibold mt-1">Avg Stuck Ratio</p>
        </div>
        <div className="bg-gradient-to-br from-purple-50 to-violet-100 rounded-2xl p-4 border border-purple-200 hover:scale-105 transition-transform cursor-pointer kawaii-shadow relative overflow-visible group">
          <div className="absolute -top-3 -right-2 text-lg opacity-0 group-hover:opacity-100 transition-opacity animate-bounce-cute">✨</div>
          <p className="text-3xl font-extrabold text-violet-600">{((stats?.avg_confidence || 0) * 100).toFixed(0)}%</p>
          <p className="text-xs text-violet-500 font-semibold mt-1">Avg Confidence</p>
        </div>
      </div>

      {/* Status Breakdown - Kawaii Style */}
      {stats?.status_breakdown && stats.status_breakdown.length > 0 && (
        <div className="kawaii-bg-cream backdrop-blur-sm rounded-2xl p-6 shadow-lg border border-amber-200/50 kawaii-shadow">
          <h3 className="text-sm font-bold text-purple-800 mb-4 flex items-center gap-2">🌸 Congestion Level Distribution</h3>
          <div className="flex flex-wrap gap-3">
            {stats.status_breakdown.map((item: any, idx: number) => {
              const colors = ['bg-green-400', 'bg-yellow-400', 'bg-orange-400', 'bg-rose-400', 'bg-purple-400'];
              const bgColors = ['bg-green-50', 'bg-yellow-50', 'bg-orange-50', 'bg-rose-50', 'bg-purple-50'];
              return (
                <div key={idx} className={`flex items-center gap-2 ${bgColors[idx % bgColors.length]} px-4 py-2 rounded-full border border-pink-100 hover:scale-105 transition-transform cursor-pointer`}>
                  <span className={`w-2.5 h-2.5 rounded-full ${colors[idx % colors.length]} shadow-sm`}></span>
                  <span className="text-sm font-semibold text-purple-700">{item.name}</span>
                  <span className="text-xs bg-white/80 px-2 py-0.5 rounded-full text-purple-500 font-medium">{item.count} ({item.percentage}%)</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Time-Series Chart - Kawaii Style */}
      <div className="kawaii-bg-sky backdrop-blur-sm rounded-[2rem] p-8 shadow-lg border border-blue-200/50 kawaii-shadow">
        <h3 className="text-sm font-bold text-purple-800 mb-6 flex items-center">
          <Activity className="w-4 h-4 mr-2 text-pink-500" />
          ✨ Congestion Events Over Time
        </h3>

        {chartData.length > 0 ? (
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f9a8d4" opacity={0.3} />
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 10, fill: '#9333ea' }}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis
                  tick={{ fontSize: 12, fill: '#9333ea' }}
                  allowDecimals={false}
                  label={{ value: 'Count', angle: -90, position: 'insideLeft', style: { fontSize: 12, fill: '#9333ea' } }}
                />
                <Tooltip
                  contentStyle={{
                    borderRadius: '16px',
                    border: '1px solid #f9a8d4',
                    boxShadow: '0 10px 25px -5px rgba(249, 168, 212, 0.3)',
                    padding: '12px',
                    background: 'rgba(255, 255, 255, 0.95)'
                  }}
                />
                <Legend
                  wrapperStyle={{ paddingTop: '20px' }}
                  formatter={(value) => <span className="text-xs font-semibold text-purple-600">{value}</span>}
                />
                {uniqueReasons.map((reason, index) => (
                  <Bar
                    key={reason}
                    dataKey={reason}
                    stackId="a"
                    fill={COLORS[index % COLORS.length]}
                    radius={index === uniqueReasons.length - 1 ? [8, 8, 0, 0] : [0, 0, 0, 0]}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-80 flex items-center justify-center text-purple-400">
            <p>✨ No congestion data available for today</p>
          </div>
        )}
      </div>

      {/* Reason Breakdown Cards - Kawaii Style */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Object.entries(reasonCounts).map(([reason, count], index) => (
          <motion.div
            key={reason}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 * index }}
            className="bg-white/80 backdrop-blur-sm rounded-2xl p-4 shadow-lg border border-pink-200/50 hover:shadow-xl hover:scale-105 transition-all cursor-pointer kawaii-shadow"
          >
            <div
              className="w-4 h-4 rounded-full mb-3 shadow-sm"
              style={{ backgroundColor: COLORS[index % COLORS.length] }}
            ></div>
            <p className="text-2xl font-extrabold text-purple-800">{count as number}</p>
            <p className="text-xs text-purple-500 font-semibold truncate" title={reason}>{reason}</p>
          </motion.div>
        ))}
      </div>

      {/* Recent Events List - Kawaii Style */}
      <div className="bg-white/80 backdrop-blur-sm rounded-[2rem] p-8 shadow-lg border border-pink-200/50 kawaii-shadow">
        <h3 className="text-sm font-bold text-purple-800 mb-4 flex items-center">
          <Clock className="w-4 h-4 mr-2 text-pink-400" />
          🕒 Recent Congestion Events
        </h3>
        <div className="space-y-3 max-h-80 overflow-y-auto">
          {todayGraph?.slice(-10).reverse().map((item: any, index: number) => (
            <div key={index} className="p-3 rounded-xl bg-pink-50/50 hover:bg-pink-100/50 transition-colors border border-pink-100">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <div
                    className="w-2.5 h-2.5 rounded-full shadow-sm"
                    style={{ backgroundColor: COLORS[uniqueReasons.indexOf(item.reason) % COLORS.length] }}
                  ></div>
                  <span className="text-sm font-semibold text-purple-700">{item.reason}</span>
                </div>
                <span className="text-xs text-purple-400 font-mono bg-white/80 px-2 py-1 rounded-lg border border-pink-100">{item.timestamp?.split(',')[0]}</span>
              </div>
              <div className="flex flex-wrap gap-2 ml-5">
                {item.status && (
                  <span className="text-xs bg-pink-100 text-pink-600 px-2 py-0.5 rounded-full font-medium">{item.status}</span>
                )}
                {item.average_speed !== undefined && (
                  <span className="text-xs bg-green-100 text-green-600 px-2 py-0.5 rounded-full font-medium">⚡ {item.average_speed.toFixed(1)} px/s</span>
                )}
                {item.stuck_ratio !== undefined && (
                  <span className="text-xs bg-amber-100 text-amber-600 px-2 py-0.5 rounded-full font-medium">🛑 {(item.stuck_ratio * 100).toFixed(0)}%</span>
                )}
                {item.confidence !== undefined && (
                  <span className="text-xs bg-purple-100 text-purple-600 px-2 py-0.5 rounded-full font-medium">📊 {(item.confidence * 100).toFixed(0)}%</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
};

export const TrafficDashboard = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { vehicleCount, time, congestion, report, graph, daily_summary, detailed_history, currentView, loading, error, stats } = useSelector((state: RootState) => state.traffic);
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
    <div className="flex h-screen overflow-hidden font-sans" style={{ background: 'linear-gradient(135deg, hsl(330 40% 96%) 0%, hsl(280 40% 96%) 50%, hsl(200 40% 96%) 100%)' }}>
      <Sidebar />
      <MobileMenu />

      <main className="flex-1 flex flex-col h-full overflow-y-auto p-4 md:p-10 relative scroll-smooth">
        {/* Mobile Header - Kawaii Style */}
        <div className="md:hidden flex justify-between items-center mb-6 sticky top-0 backdrop-blur-md z-30 py-2 px-3 rounded-2xl" style={{ background: 'rgba(255,255,255,0.7)' }}>
          <h1 className="text-lg font-extrabold text-purple-700 flex items-center gap-2">🌸 Smart Traffic</h1>
          <button onClick={() => dispatch(toggleMobileMenu())} className="p-2 active:bg-pink-100 rounded-full transition-colors">
            <Menu className="w-6 h-6 text-pink-500" />
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
                <header className="mb-12 flex flex-col md:flex-row md:items-start justify-between gap-8 relative">
                  {/* Floating anime cars */}
                  <div className="absolute -top-6 left-40 text-2xl animate-car-drive z-10 pointer-events-none hidden md:block">🚗</div>
                  <div className="absolute -top-4 right-40 text-xl animate-float pointer-events-none hidden md:block" style={{ animationDelay: '1.5s' }}>🚙</div>

                  <div className="pt-4 flex items-start gap-6">
                    {/* Anime Car Mascot */}
                    <motion.div
                      className="hidden lg:block"
                      animate={{ y: [0, -8, 0] }}
                      transition={{ repeat: Infinity, duration: 2.5 }}
                    >
                      <img
                        src="/anime_car_mascot.png"
                        alt="Car-chan"
                        className="w-20 h-20 object-contain drop-shadow-lg"
                      />
                    </motion.div>

                    <div>
                      <div className="flex items-center gap-3 mb-3">
                        <span className="text-xs font-bold uppercase tracking-[0.2em] text-pink-400">✨ overview</span>
                        {loading && <Loader2 className="w-4 h-4 text-pink-500 animate-spin" />}
                        <button
                          onClick={handleRefresh}
                          className="p-1.5 hover:bg-pink-100 rounded-full transition-colors"
                          title="Refresh data"
                        >
                          <RefreshCw className="w-4 h-4 text-pink-400 hover:text-pink-600" />
                        </button>
                      </div>
                      <h2 className="text-4xl font-extrabold text-purple-900 mb-3 tracking-tight">Analyze Your <br />Congestion 🌸</h2>
                      <p className="text-xs text-purple-400 font-bold tracking-wide uppercase">Pookie the traffic analyser ✨</p>
                      {error && (
                        <div className="mt-3 flex items-center gap-2 text-rose-500 text-sm">
                          <AlertCircle className="w-4 h-4" />
                          <span>{error}</span>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex gap-5">
                    <div onClick={() => handleStatClick('Vehicle Count')}>
                      <StatCard
                        value={vehicleCount}
                        label="V-Count"
                        colorClass="bg-gradient-to-br from-[#a8e6cf] via-[#88d8b0] to-[#7ad9a4]" /* Kawaii Mint */
                        delay={0.1}
                        icon={Car}
                      />
                    </div>
                    <div onClick={() => handleStatClick('Time')}>
                      <StatCard
                        value={time}
                        label="Time"
                        colorClass="bg-gradient-to-br from-[#f9a8d4] via-[#f472b6] to-[#ec4899]" /* Kawaii Pink */
                        delay={0.2}
                        icon={Clock}
                      />
                    </div>
                  </div>
                </header>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                  {/* Left Column: Chart with Anime Car Decorations */}
                  <div className="lg:col-span-7 relative">
                    {/* Floating anime cars decoration */}
                    <div className="absolute -top-8 left-10 text-2xl animate-car-drive z-10 pointer-events-none">🚗</div>
                    <div className="absolute -top-6 right-20 text-xl animate-float" style={{ animationDelay: '1s' }}>🚙</div>

                    <motion.div
                      whileHover={{ scale: 1.01 }}
                      className="flex items-center justify-center min-h-[320px] kawaii-bg-lavender backdrop-blur-sm rounded-3xl p-6 border border-purple-200/50 shadow-xl kawaii-shadow transition-all relative overflow-hidden"
                    >
                      {/* Road lines decoration */}
                      <div className="absolute bottom-0 left-0 right-0 h-1.5 bg-gradient-to-r from-transparent via-yellow-300/30 to-transparent rounded-b-3xl"></div>
                      <CongestionChart data={congestion} />
                    </motion.div>
                  </div>

                  {/* Right Column: Report with Traffic Decorations */}
                  <div className="lg:col-span-5 kawaii-bg-pink backdrop-blur-sm rounded-3xl p-6 shadow-xl border border-pink-200/50 max-h-[380px] relative overflow-hidden kawaii-shadow">
                    {/* Gradient stripe top */}
                    <div className="absolute top-0 left-0 w-full h-1.5 bg-gradient-to-r from-pink-400 via-purple-400 to-pink-300 rounded-t-3xl"></div>

                    {/* Traffic light corner decoration */}
                    <div className="absolute top-4 right-4 flex flex-col gap-1">
                      <div className="w-2 h-2 rounded-full bg-rose-400 animate-pulse"></div>
                      <div className="w-2 h-2 rounded-full bg-amber-400"></div>
                      <div className="w-2 h-2 rounded-full bg-green-400"></div>
                    </div>

                    <div className="flex justify-between items-center mb-4">
                      <h3 className="text-sm font-bold text-purple-800 flex items-center">
                        <FileText className="w-4 h-4 mr-2 text-pink-400" />
                        📝 Traffic Report
                      </h3>
                      <span className="text-[10px] bg-gradient-to-r from-pink-400 to-rose-400 text-white px-3 py-1.5 rounded-full font-bold shadow-sm flex items-center gap-1">
                        <span className="w-1.5 h-1.5 bg-white rounded-full animate-pulse"></span>
                        Live
                      </span>
                    </div>

                    <div className="space-y-1 max-h-[280px] overflow-y-auto pr-1">
                      {report.map((item: any, idx: number) => (
                        <ReportCard key={idx} index={idx} reason={item.reason} suggestions={item.suggestions || []} />
                      ))}
                      {report.length === 0 && (
                        <p className="text-sm text-purple-400 text-center py-8">✨ No reports yet</p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Bottom Section - Enhanced Stats with Real Data - Kawaii Style */}
                <div className="mt-6 kawaii-bg-mint backdrop-blur-sm rounded-[2rem] p-6 shadow-lg border border-green-200/50 hover:shadow-xl transition-shadow duration-300 kawaii-shadow">
                  <h3 className="text-sm font-bold text-pink-500 uppercase tracking-wider mb-4 flex items-center gap-2">✨ Real-Time Analytics</h3>

                  {/* Stats Grid - Kawaii Colors */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div className="bg-gradient-to-br from-pink-50 to-pink-100 rounded-2xl p-4 border border-pink-200 hover:scale-105 transition-transform cursor-pointer">
                      <p className="text-2xl font-extrabold text-pink-600">{stats.total_events}</p>
                      <p className="text-[10px] text-pink-500 font-semibold mt-0.5">Total Events</p>
                    </div>
                    <div className="bg-gradient-to-br from-green-50 to-emerald-100 rounded-2xl p-4 border border-green-200 hover:scale-105 transition-transform cursor-pointer">
                      <p className="text-2xl font-extrabold text-emerald-600">{stats.avg_speed.toFixed(1)}</p>
                      <p className="text-[10px] text-emerald-500 font-semibold mt-0.5">Avg Speed</p>
                    </div>
                    <div className="bg-gradient-to-br from-amber-50 to-orange-100 rounded-2xl p-4 border border-amber-200 hover:scale-105 transition-transform cursor-pointer">
                      <p className="text-2xl font-extrabold text-amber-600">{(stats.avg_stuck_ratio * 100).toFixed(0)}%</p>
                      <p className="text-[10px] text-amber-500 font-semibold mt-0.5">Stuck Ratio</p>
                    </div>
                    <div className="bg-gradient-to-br from-purple-50 to-violet-100 rounded-2xl p-4 border border-purple-200 hover:scale-105 transition-transform cursor-pointer">
                      <p className="text-2xl font-extrabold text-violet-600">{(stats.avg_confidence * 100).toFixed(0)}%</p>
                      <p className="text-[10px] text-violet-500 font-semibold mt-0.5">Confidence</p>
                    </div>
                  </div>

                  {/* Status Breakdown - Kawaii Style */}
                  {stats.status_breakdown && stats.status_breakdown.length > 0 && (
                    <div>
                      <h4 className="text-xs font-bold text-purple-400 uppercase tracking-wider mb-4">🌸 Congestion Level Breakdown</h4>
                      <div className="flex flex-wrap gap-3">
                        {stats.status_breakdown.map((item: any, idx: number) => {
                          const colors = ['bg-green-400', 'bg-yellow-400', 'bg-orange-400', 'bg-rose-400', 'bg-purple-400'];
                          const bgColors = ['bg-green-50', 'bg-yellow-50', 'bg-orange-50', 'bg-rose-50', 'bg-purple-50'];
                          return (
                            <div key={idx} className={`flex items-center gap-2 ${bgColors[idx % bgColors.length]} px-4 py-2 rounded-full border border-pink-100 hover:scale-105 transition-transform cursor-pointer`}>
                              <span className={`w-2.5 h-2.5 rounded-full ${colors[idx % colors.length]} shadow-sm`}></span>
                              <span className="text-sm font-semibold text-purple-700">{item.name}</span>
                              <span className="text-xs text-purple-400 font-medium">({item.count} - {item.percentage}%)</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Most Common Reason - Kawaii Style */}
                  {congestion && congestion.length > 0 && (
                    <div className="mt-6 pt-6 border-t border-pink-200">
                      <p className="text-xs text-pink-400 uppercase tracking-wider mb-2 font-semibold">✨ Most Common Reason</p>
                      <p className="text-lg font-extrabold text-purple-700">{congestion[0]?.name || 'N/A'}</p>
                      <p className="text-sm text-purple-500">{congestion[0]?.percentage || 0}% of all events</p>
                    </div>
                  )}
                </div>
              </motion.div>
            )}

            {currentView === 'live' && (
              <LiveMonitoringView graph={graph} loading={loading} stats={stats} />
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
                <div className="w-24 h-24 bg-gradient-to-br from-pink-200 to-purple-200 rounded-full flex items-center justify-center mb-6 animate-pulse kawaii-shadow">
                  <Settings className="w-10 h-10 text-purple-500" />
                </div>
                <h2 className="text-2xl font-extrabold text-purple-800 mb-2 capitalize">
                  ✨ Settings View
                </h2>
                <p className="text-purple-400 max-w-md">
                  This module is currently being simulated. In a production environment, this would display real-time settings.
                </p>
                <button
                  onClick={() => dispatch(setView('dashboard'))}
                  className="mt-8 px-6 py-3 bg-gradient-to-r from-pink-400 to-purple-400 text-white rounded-2xl font-bold text-sm hover:from-pink-500 hover:to-purple-500 transition-all shadow-lg hover:shadow-xl kawaii-shadow"
                >
                  🌸 Return to Dashboard
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
                <div className="w-24 h-24 bg-gradient-to-br from-pink-200 to-purple-200 rounded-full flex items-center justify-center mb-6 animate-pulse kawaii-shadow">
                  <MapPin className="w-10 h-10 text-purple-500" />
                </div>
                <h2 className="text-2xl font-extrabold text-purple-800 mb-2 capitalize">
                  📍 Live Map View
                </h2>
                <p className="text-purple-400 max-w-md">
                  This module is currently being simulated. In a production environment, this would display real-time map data.
                </p>
                <button
                  onClick={() => dispatch(setView('dashboard'))}
                  className="mt-8 px-6 py-3 bg-gradient-to-r from-pink-400 to-purple-400 text-white rounded-2xl font-bold text-sm hover:from-pink-500 hover:to-purple-500 transition-all shadow-lg hover:shadow-xl kawaii-shadow"
                >
                  🌸 Return to Dashboard
                </button>
              </motion.div>
            )}

            {currentView === 'admin' && (
              <AdminPanel />
            )}
          </AnimatePresence>
        </div>
      </main>

      {/* 🌸 Anime Character Mascot - Floating Helper */}
      <AnimeMascot />
    </div>
  );
};
