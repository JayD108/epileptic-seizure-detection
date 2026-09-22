Plaintextneuropulse/
│
├── docker-compose.yml
├── .env.example
├── README.md
│
├── backend/
│   ├── app/
│   │   ├── main.py                   # FastAPI entrypoint (App initialization, CORS, WebSocket router)
│   │   ├── config.py                 # App settings, path configurations, stream tick rates
│   │   │
│   │   ├── api/
│   │   │   ├── websocket.py          # Real-time WebSocket endpoint for 22-patient telemetry
│   │   │   ├── stream_control.py     # Play/Pause, Speed multiplier (1x-60x), Seek endpoints
│   │   │   ├── eeg.py                # Static EDF upload and manual file batch analysis
│   │   │   └── predictions.py        # Historical predictions, model metrics, and reports
│   │   │
│   │   ├── services/
│   │   │   ├── simulator_engine.py   # Multi-patient EDF stream replay worker
│   │   │   ├── feature_pipeline.py   # Real-time 10s sliding window PSD & Hjorth extractor
│   │   │   ├── inference_engine.py   # XGBoost prediction engine + SHAP attribution calculator
│   │   │   └── websocket_manager.py  # Broadcast manager for active client socket connections
│   │   │
│   │   ├── schemas/
│   │   │   ├── stream.py             # Pydantic schemas for live frame telemetry and socket events
│   │   │   └── prediction.py         # Schemas for risk output, XAI metrics, and patient state
│   │   │
│   │   └── models/                   # Serialized ML artifacts
│   │       ├── xgboost_model.pkl     # Trained preictal classifier
│   │       ├── scaler.pkl            # Feature scaler
│   │       └── feature_names.json    # Ordering mapping for standard 18 channels x 5 bands
│   │
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx                  # Central EMU Grid View (All 22 Patients)
│   │   ├── patient/
│   │   │   └── [id]/page.tsx         # Deep-Dive View (3D Brain + EEG Canvas + Live XAI)
│   │   ├── upload/page.tsx           # Manual EDF File Analysis Tool
│   │   ├── analytics/page.tsx        # System-wide performance metrics & cross-patient ROC/AUC
│   │   └── layout.tsx
│   │
│   ├── components/
│   │   ├── emu/
│   │   │   ├── StreamControls.tsx    # Play, Pause, Speed controls (1x, 5x, 10x, 60x), Seek
│   │   │   ├── PatientCard.tsx       # Mini-card for Central Grid with risk gauge and alerts
│   │   │   ├── AlertBanner.tsx       # Global preictal state transition warning bar
│   │   │   └── GridFilter.tsx        # Filter patients by risk state (All, Normal, Preictal)
│   │   │
│   │   ├── deepdive/
│   │   │   ├── BrainViewer3D.tsx     # React Three Fiber 3D interactive mesh with cortical heatmaps
│   │   │   ├── EEGStreamCanvas.tsx   # 60 FPS HTML5 Canvas renderer for continuous multi-channel wave
│   │   │   ├── LiveSHAPChart.tsx     # Dynamic feature attribution bar chart (Top 5 risk drivers)
│   │   │   ├── RiskHorizon.tsx       # Probability trend line over time with 10-30 min preictal window
│   │   │   └── ChannelConnectivity.tsx# Coherence network graph across electrode nodes
│   │   │
│   │   └── ui/                       # shadcn/ui base primitives (Button, Card, Badge, Slider, Dialog)
│   │
│   ├── hooks/
│   │   ├── useEMUSocket.ts           # Global WebSocket hook managing 22-patient stream state
│   │   └── usePatientStream.ts       # Single-patient focused stream hook
│   │
│   ├── lib/
│   │   ├── utils.ts                  # Canvas scaling and mathematical helpers
│   │   └── constants.ts              # 10-20 channel coordinates, frequency band colors
│   │
│   ├── package.json
│   └── Dockerfile
│
└── ml/
    ├── data/
    │   ├── raw/                      # CHB-MIT EDF and summary text files (chb01 - chb24)
    │   └── processed/                # Multiprocessed extracted feature CSV files
    │
    ├── preprocessing/
    │   ├── parser.py                 # Summary text parser extracting exact seizure windows
    │   └── feature_extractor.py     # Multi-threaded Welch PSD & Hjorth extractor
    │
    ├── training/
    │   ├── train_xgboost.py          # XGBoost training script with class imbalance weighting
    │   └── validate.py               # Patient-wise cross-validation & chronological validation
    │
    └── ingest_dataset.py             # Parallel dataset pipeline runner
🖥️ Live EMU Dashboard Design Architecture1. Global Simulation & Control Bar (Top Navigation)Located across the top of every screen to manage the underlying simulation replay loop:System Status: WebSocket Connection Monitor (LIVE, RECONNECTING, OFFLINE).Playback Controls: Play, Pause, Step-Forward (10s), and Speed Multiplier Toggle (1x Real-Time, 5x, 10x, 60x Demo Mode).Global Preictal Alarm Counter: Displays active alerts (e.g., 3 Patients in Preictal State).Global Time Index: Current simulated timestamp relative to the dataset playback.2. Primary View: Central EMU Grid (/app/page.tsx)A high-density monitoring matrix designed to simulate an hospital Epilepsy Monitoring Unit displaying all 22 CHB-MIT patients simultaneously.Plaintext┌────────────────────────────────────────────────────────────────────────────────────────┐
│ [LIVE] EMU Central Monitor  │  [Play] [Pause]  Speed: [1x] [5x] [10x] [60x] │ ⚠️ 2 ALERTS │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐          │
│  │ PATIENT 01    [30s]  │  │ PATIENT 02    [30s]  │  │ PATIENT 03    [30s]  │  ... 22  │
│  │ Risk: 14%    NORMAL  │  │ Risk: 82%  PREICTAL  │  │ Risk: 08%    NORMAL  │  Cards   │
│  │ -------------------- │  │ -------------------- │  │ -------------------- │          │
│  │ [Mini EEG Waves]     │  │ [Mini EEG Waves]     │  │ [Mini EEG Waves]     │          │
│  │ Band: Alpha Dominant │  │ Band: Theta Spike ↑  │  │ Band: Normal Baseline│          │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────────┘          │
└────────────────────────────────────────────────────────────────────────────────────────┘
Patient Card Features:Dynamic Status Badge:🟢 Interictal (Normal): Risk $< 30\%$🟠 Preictal (Elevated Warning): Risk $30\%\text{--}70\%$ (Seizure estimated in 10–30 mins)🔴 High Preictal Risk: Risk $> 70\%$Real-time Seizure Risk Gauge: Radial circular progress visualizer updated every 2 seconds.Mini Waveform Sparkline: Canvas preview showing 3 primary channels (FP1-F7, C3-P3, CZ-PZ).Primary Band Indicator: Shows dominant active band (e.g., Elevated Theta Power).Direct Navigation: Clicking anywhere on a card transitions smoothly into that patient's Deep-Dive View.3. Secondary View: Patient Deep-Dive (/app/patient/[id]/page.tsx)A detailed analytical interface split into a 4-quadrant layout for deep clinical and explainability inspection:Plaintext┌─────────────────────────────────────────┬──────────────────────────────────────────────┐
│ QUADRANT 1: 3D Brain Activity Heatmap   │ QUADRANT 2: Multi-Channel EEG Stream         │
│ (React Three Fiber Canvas)              │ (60 FPS HTML5 Canvas)                        │
│                                         │                                              │
│ • Interactive cortical mesh             │ • Real-time rendering of 18 channels         │
│ • Dynamic lobe highlights based on PSD  │ • Channel selection & gain controls          │
│ • Electrode node markers (10-20 system) │ • Notch & Band-pass filter status indicators │
├─────────────────────────────────────────┼──────────────────────────────────────────────┤
│ QUADRANT 3: Live XAI Engine (SHAP)      │ QUADRANT 4: Risk Horizon & Predictive Timeline│
│ (Recharts / Animated Bar Chart)         │ (Temporal Probability Curve)                 │
│                                         │                                              │
│ • Top 5 feature attributions pushing    │ • Historical risk curve over the last hour   │
│   risk score up or down                 │ • Projected 10–30 minute preictal horizon    │
│ • e.g., "+32% Theta Power in T7-P7"    │ • Seizure threshold marker                   │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
🔄 Real-Time Data Flow LogicBackend Replay Loop (services/simulator_engine.py):Reads 10-second raw buffers from all 22 patient .edf files continuously.Advances timestamps based on the speed multiplier setting ($1\times, 5\times$, etc.).Feature & Model Pipeline (services/inference_engine.py):Computes Welch's PSD for Delta through Gamma bands across 18 channels ($18 \times 5 = 90\text{ features}$).Evaluates feature vector via serialized XGBoost model to obtain probability score $P(\text{Preictal})$.Executes TreeSHAP explainer on the current frame to identify top positive feature attributions.WebSocket Telemetry Payload (api/websocket.py):Emits a synchronized JSON frame to all connected clients every $1000\text{ms}$ per patient:JSON{
  "timestamp": 1240.0,
  "patients": {
    "chb01": {
      "risk_score": 0.14,
      "state": "INTERICTAL",
      "dominant_region": "Occipital",
      "top_shap_features": [
        {"feature": "P7-O1_Alpha", "contribution": -0.12},
        {"feature": "F3-C3_Beta", "contribution": +0.04}
      ],
      "band_power_distribution": {
        "Delta": 0.45,
        "Theta": 0.20,
        "Alpha": 0.22,
        "Beta": 0.10,
        "Gamma": 0.03
      }
    },
    "chb02": {
      "risk_score": 0.82,
      "state": "PREICTAL",
      "dominant_region": "Temporal Left",
      "top_shap_features": [
        {"feature": "T7-P7_Theta", "contribution": +0.34},
        {"feature": "F7-T7_Gamma", "contribution": +0.21}
      ],
      "band_power_distribution": {
        "Delta": 0.15,
        "Theta": 0.48,
        "Alpha": 0.12,
        "Beta": 0.15,
        "Gamma": 0.10
      }
    }
  }
}
