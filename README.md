# 🏭 Tube Counter

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.13">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/YOLOv8-FF4F00?style=for-the-badge&logo=pytorch" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite">
  <img src="https://img.shields.io/badge/Render-Deploy_Ready-46E3B7?style=for-the-badge" alt="Render Ready">
</p>

An intelligent, AI-powered industrial web application for automatically detecting and counting thread tubes in baskets using Computer Vision (YOLOv8) and FastAPI. Designed with a mobile-first UI for factory field workers.

---

## ✨ Features

- **🧠 AI-Powered Detection**: Replaced legacy OpenCV Hough Circles with a custom-trained YOLOv8 model for robust, high-accuracy object detection even with overlapping or tilted tubes.
- **📱 Mobile-First UI**: A responsive, clean interface tailored specifically for mobile usage on the factory floor.
- **📸 Native Camera Integration**: Tap to open the mobile device's native camera immediately without jittery web-camera delays.
- **⚡ Fast Cloud Backend**: FastAPI backend optimized for free-tier cloud deployment (e.g., Render) utilizing memory-efficient PyTorch CPU builds.
- **📊 History & Export**: Keep track of counts, categorize defect types (เช่น ใยพันทับ, ใยแตก), and export data instantly to CSV or JSON.

---

## 🏗️ Architecture

```text
┌─────────────────────────┐        ┌──────────────────────────┐
│   Browser (Mobile/PC)   │        │   FastAPI Server         │
│                         │        │                          │
│  - Vite Dev Server      │  HTTP  │  - YOLOv8 Inference      │
│  - HTML/CSS/JS          │ ─────▶ │  - PyTorch (CPU mode)    │
│  - Native Camera Upload │ ◀───── │  - JSON Data Storage     │
└─────────────────────────┘        └──────────────────────────┘
```

## 🚀 Getting Started

### Prerequisites
- Python 3.13
- Node.js & npm (for frontend development)

### 1. Backend Setup (FastAPI & YOLOv8)

```bash
# Clone the repository
git clone https://github.com/your-username/tube-counter.git
cd tube-counter

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (CPU-optimized PyTorch included)
pip install -r requirements.txt
```

### 2. Frontend Setup (Vite)

```bash
# Install frontend dependencies
npm install
```

### 3. Run the Development Environment

Run both frontend and backend concurrently:
```bash
npm run dev
```
- **Frontend**: `http://localhost:5173`
- **Backend API & Docs**: `http://localhost:8000/docs`

---

## ☁️ Deployment (Render)

This project is configured for seamless deployment on [Render](https://render.com).

1. Connect your GitHub repository to Render.
2. Create a new **Web Service**.
3. Use the following configuration (already provided in `render.yaml`):
   - **Environment**: Python
   - **Build Command**: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt` (Ensures memory efficiency to prevent OOM errors).
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

---

## 📂 Project Structure

```text
tube-counter/
├── model/
│   └── best.pt              # Custom trained YOLOv8 model
├── static/                  # Production static assets
├── index.html               # Main frontend entrypoint
├── main.py                  # FastAPI application & endpoints
├── package.json             # NPM scripts (concurrently, vite)
├── render.yaml              # Render deployment configuration
├── records.json             # File-based database
└── requirements.txt         # Python dependencies
```

---

<p align="center">
  <i>Developed for optimized industrial counting workflows.</i>
</p>
