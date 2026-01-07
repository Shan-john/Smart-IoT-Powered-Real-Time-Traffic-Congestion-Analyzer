# 🚦 Smart IoT-Powered Real-Time Traffic Congestion Analyzer

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org)
[![Firebase](https://img.shields.io/badge/Firebase-Realtime_DB-FFCA28.svg)](https://firebase.google.com)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-Object_Detection-00FFFF.svg)](https://ultralytics.com)

An intelligent traffic monitoring system that uses computer vision and AI to detect, analyze, and report traffic congestion in real-time. Designed for both desktop PCs and Raspberry Pi 5 IoT deployments.

![Traffic Dashboard Preview](https://via.placeholder.com/800x400?text=Traffic+Dashboard+Preview)

---

## ✨ Features

- **🎯 Real-Time Vehicle Detection** - YOLOv5 powered vehicle detection and tracking
- **📊 Congestion Analysis** - Optical flow + rule-based intelligent congestion classification
- **🧠 540+ Congestion Reasons** - Comprehensive reason database for accurate classification
- **💡 AI-Powered Suggestions** - Context-specific recommendations for each congestion type
- **🔥 Firebase Integration** - Real-time data sync to cloud dashboard
- **📱 Modern React Dashboard** - Beautiful, responsive web interface
- **🍇 Raspberry Pi Ready** - Optimized for IoT edge deployment

---

## 🏗️ System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Camera/Video  │────▶│   main.py        │────▶│    Firebase     │
│                 │     │  (Detection Hub) │     │  Realtime DB    │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                │                         │
                                ▼                         ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │ congestion_      │     │ traffic_backend │
                        │ analyzer.py      │     │ (Flask API)     │
                        └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │ React Dashboard │
                                                 │ (Web UI)        │
                                                 └─────────────────┘
```

---

## 📋 Requirements

### Hardware Requirements

| Component | PC (Recommended) | Raspberry Pi 5 |
|-----------|-----------------|----------------|
| CPU | Intel i5+ / Ryzen 5+ | Quad-core ARM Cortex-A76 |
| RAM | 8GB+ | 8GB |
| GPU | Optional (NVIDIA for faster inference) | N/A |
| Camera | USB Webcam / IP Camera | USB Webcam / Pi Camera |
| Storage | 10GB+ | 32GB+ SD Card |

### Software Requirements

- Python 3.9+
- Node.js 16+
- npm or yarn
- Git

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Shan-john/Smart-IoT-Powered-Real-Time-Traffic-Congestion-Analyzer.git
cd Smart-IoT-Powered-Real-Time-Traffic-Congestion-Analyzer
```

### 2. Setup Firebase

1. Create a Firebase project at [console.firebase.google.com](https://console.firebase.google.com)
2. Enable Realtime Database
3. Download your service account key as `firebase_key.json`
4. Place it in the project root directory

> ⚠️ **Security**: Never commit `firebase_key.json` to version control!

### 3. Choose Your Platform

- [PC Installation](#-pc-installation)
- [Raspberry Pi 5 Installation](#-raspberry-pi-5-installation)

---

## 💻 PC Installation

### Backend Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install opencv-python ultralytics firebase-admin numpy pillow flask flask-cors

# Optional: For CLIP model support (adds ~2GB)
pip install transformers torch
```

### Dashboard Setup

```bash
cd traffic-dashboard/traffic-dashboard

# Install dependencies
npm install

# Start development server
npm run dev
```

### Run the System

```bash
# Terminal 1: Start main detection
python main.py

# Terminal 2: Start backend API
cd traffic_backend
python app.py

# Terminal 3: Start dashboard (if not already running)
cd traffic-dashboard/traffic-dashboard
npm run dev
```

### Access Dashboard

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🍓 Raspberry Pi 5 Installation

### Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip python3-venv -y

# Install OpenCV dependencies
sudo apt install libopencv-dev python3-opencv -y

# Install Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install nodejs -y
```

### Optimized Backend Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Pi-optimized packages
pip install opencv-python-headless  # No GUI overhead
pip install ultralytics firebase-admin numpy pillow flask flask-cors

# DO NOT install transformers/torch (too heavy for Pi)
```

### Performance Optimizations

Edit `main.py` to apply these optimizations:

```python
# 1. Use nano model (smallest, fastest)
model = YOLO("yolov5n.pt")

# 2. Lower resolution
frame = cv2.resize(frame, (320, 240))

# 3. Disable CLIP (already default)
congestion_analyzer = CongestionAnalyzer(use_clip=False)
```

### Pi-Specific Configuration

Create `config_pi.py`:

```python
# Raspberry Pi 5 optimized settings
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
SKIP_FRAMES = 2  # Analyze every 2nd frame
USE_CLIP = False
YOLO_MODEL = "yolov5n.pt"
ANALYSIS_COOLDOWN = 15  # Seconds between Firebase updates
```

### Run on Boot (Optional)

Create a systemd service for auto-start:

```bash
sudo nano /etc/systemd/system/traffic-analyzer.service
```

```ini
[Unit]
Description=Traffic Congestion Analyzer
After=network.target

[Service]
ExecStart=/home/pi/Smart-IoT-Powered-Real-Time-Traffic-Congestion-Analyzer/venv/bin/python /home/pi/Smart-IoT-Powered-Real-Time-Traffic-Congestion-Analyzer/main.py
WorkingDirectory=/home/pi/Smart-IoT-Powered-Real-Time-Traffic-Congestion-Analyzer
User=pi
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable traffic-analyzer
sudo systemctl start traffic-analyzer
```

---

## 📊 Performance Expectations

| Platform | FPS | Latency | RAM Usage | Notes |
|----------|-----|---------|-----------|-------|
| PC (CPU only) | 15-25 | <100ms | ~2GB | Good for development |
| PC (NVIDIA GPU) | 30-60 | <50ms | ~4GB | Best performance |
| Raspberry Pi 5 | 5-12 | 0.5-1s | ~3GB | Good for deployment |
| Pi 5 + Coral TPU | 15-25 | <200ms | ~2GB | Recommended for Pi |

---

## 🎯 Recommendations

### For Development (PC)
- ✅ Use full resolution (640x480)
- ✅ Enable CLIP for detailed reason classification
- ✅ Use GPU acceleration if available
- ✅ Run all components locally

### For Production (Raspberry Pi 5)
- ✅ Use low resolution (320x240)
- ❌ Disable CLIP model
- ✅ Use YOLOv5 Nano model
- ✅ Skip frames for processing
- ✅ Consider adding Google Coral USB TPU
- ✅ Use headless OpenCV
- ✅ Set up auto-start service

### For Best Accuracy
- ✅ Position camera at optimal angle (45° from traffic)
- ✅ Ensure good lighting conditions
- ✅ Calibrate stuck detection thresholds for your location
- ✅ Fine-tune speed thresholds based on road type

---

## 📁 Project Structure

```
Smart-IoT-Powered-Real-Time-Traffic-Congestion-Analyzer/
├── main.py                    # Main detection script
├── tracker.py                 # Enhanced vehicle tracker
├── congestion_analyzer.py     # Congestion detection engine
├── reasons_database.py        # 540+ congestion reasons
├── reason_analyzer.py         # CLIP-based analysis (optional)
├── firebase_key.json          # Firebase credentials (DO NOT COMMIT)
├── traffic_backend/
│   ├── app.py                 # Flask API server
│   ├── traffic_processor.py   # Data processing
│   └── suggestion_generator.py # AI suggestions
└── traffic-dashboard/
    └── traffic-dashboard/
        ├── src/
        │   ├── components/
        │   │   └── TrafficDashboard.tsx
        │   ├── store.ts
        │   └── firebase-config.js
        └── package.json
```

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "No module named cv2" | `pip install opencv-python` |
| "CUDA out of memory" | Reduce batch size or use CPU |
| Firebase connection error | Check `firebase_key.json` path |
| Low FPS on Pi | Apply optimizations above |
| Camera not detected | Check USB connection, try `ls /dev/video*` |

### Pi-Specific Issues

```bash
# If camera permission denied
sudo usermod -a -G video $USER
sudo reboot

# If memory errors
# Add swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [OpenCV](https://opencv.org/)
- [Firebase](https://firebase.google.com/)
- [React](https://reactjs.org/)
- [Recharts](https://recharts.org/)

---

## 📬 Contact

**Shan John** - [@Shan-john](https://github.com/Shan-john)

Project Link: [https://github.com/Shan-john/Smart-IoT-Powered-Real-Time-Traffic-Congestion-Analyzer](https://github.com/Shan-john/Smart-IoT-Powered-Real-Time-Traffic-Congestion-Analyzer)

---

<p align="center">
  Made with ❤️ for smarter traffic management
</p>
