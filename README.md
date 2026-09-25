

## Explainable EEG-Based Seizure Prediction & Brain-State Visualization

> **Understand the brain. Anticipate the event.**

NeuroPulse is an academic machine-learning project that explores **EEG-based seizure prediction** using signal processing, machine learning, explainable AI, and interactive brain visualization.

The system analyzes multi-channel EEG recordings, extracts meaningful temporal/frequency-domain characteristics, estimates **pre-seizure risk**, and presents the results through an interactive dashboard designed for two audiences:

* 🧑 **General users** — simple explanations of what the EEG patterns mean.
* 👨‍⚕️ **Neurologists / researchers** — detailed EEG, signal-processing, prediction, model, and explainability information.

> **Important:** NeuroPulse is an educational/research prototype and is **not a clinically validated medical device, diagnostic system, or emergency warning system.**

---

# 📌 Table of Contents

* [Project Motivation](#-project-motivation)
* [Core Objective](#-core-objective)
* [Detection vs Prediction](#-detection-vs-prediction)
* [Key Features](#-key-features)
* [System Architecture](#-system-architecture)
* [Project Structure](#-project-structure)
* [Machine Learning Pipeline](#-machine-learning-pipeline)
* [EEG Processing Pipeline](#-eeg-processing-pipeline)
* [Brain Visualization](#-brain-visualization)
* [Prediction Architecture](#-prediction-architecture)
* [Explainable AI](#-explainable-ai)
* [Datasets](#-datasets)
* [Model Development](#-model-development)
* [Evaluation](#-evaluation)
* [Frontend Architecture](#-frontend-architecture)
* [Backend Architecture](#-backend-architecture)
* [API Design](#-api-design)
* [Technology Stack](#-technology-stack)
* [Installation](#-installation)
* [Running the Project](#-running-the-project)
* [Configuration](#-configuration)
* [Example Workflow](#-example-workflow)
* [Research Limitations](#-research-limitations)
* [Future Improvements](#-future-improvements)
* [Disclaimer](#-disclaimer)

---

# 🎯 Project Motivation

Epileptic seizures are associated with abnormal electrical activity in the brain.

Electroencephalography (**EEG**) provides a non-invasive way to record electrical activity from multiple scalp electrodes.

Traditional EEG analysis requires expert interpretation and can involve large amounts of time-series data.

The goal of NeuroPulse is to investigate whether machine-learning models can identify **patterns associated with the preictal state** and estimate elevated seizure risk before a seizure occurs.

The project combines:

```text
EEG Signal
    ↓
Signal Processing
    ↓
Feature Extraction
    ↓
Machine Learning
    ↓
Seizure Risk Prediction
    ↓
Explainable AI
    ↓
Brain / EEG Visualization
    ↓
Human-Readable Report
```

---

# 🎯 Core Objective

The primary objective is:

> **Develop an EEG-based machine-learning system that estimates whether an EEG segment exhibits patterns associated with an upcoming seizure.**

The system should not simply classify whether a seizure is currently occurring.

Instead, the research focus is:

### Prediction

```text
Current EEG
     ↓
Preictal-like patterns?
     ↓
Estimated seizure risk
     ↓
Prediction horizon
```

---

# ⚠️ Detection vs Prediction

These concepts must not be confused.

### Seizure Detection

Answers:

> "Is a seizure happening now?"

Example:

```text
EEG → Seizure / Non-Seizure
```

### Seizure Prediction

Answers:

> "Does the current EEG resemble a state that precedes a seizure?"

Example:

```text
EEG
 ↓
Interictal / Preictal characteristics
 ↓
Risk Score
 ↓
Prediction Window
```

NeuroPulse focuses primarily on **prediction**.

---

# ✨ Key Features

## 🧠 Interactive 3D Brain

The dashboard contains an interactive 3D brain visualization.

Users can:

* Rotate the brain
* Zoom
* Inspect hemispheres
* Select anatomical regions
* View regional activity estimates
* Highlight associated EEG electrodes
* Observe changes in estimated abnormal activity

The system should use careful terminology.

Instead of claiming:

> "The seizure originates here."

the application should communicate:

> **"Estimated cortical region associated with abnormal EEG activity."**

Scalp EEG does not directly measure the exact cortical source of electrical activity.

---

# 📡 EEG Visualization

The dashboard provides:

* Multi-channel EEG waveform
* Channel selection
* Time-window selection
* Amplitude scaling
* Sampling-rate information
* Signal-quality indicators
* Filter information
* Electrode labels
* Highlighted abnormal channels

Example channels:

```text
Fp1
Fp2
F3
F4
C3
C4
P3
P4
O1
O2
T3
T4
T5
T6
...
```

---

# 🔥 Seizure Risk Prediction

The primary dashboard prediction card displays:

```text
SEIZURE RISK

73%

ELEVATED

Prediction Horizon
10–30 minutes

Model Probability
0.73
```

The interface should distinguish:

* Model probability
* Risk score
* Prediction horizon
* Model confidence

These should **not** be represented as a clinically validated probability of seizure.

---

# ⏱️ Prediction Timeline

The system visualizes the relationship between:

```text
INTERICTAL
      ↓
PREICTAL
      ↓
PREDICTION ALERT
      ↓
SEIZURE OCCURRENCE
```

Example:

```text
PAST                                      FUTURE

NORMAL ────────────────────────┐
                               │
                         PREICTAL
                               │
                               ▼
                        MODEL ALERT
                               │
                               ├──────────►
                               │
                          SEIZURE WINDOW
```

The prediction horizon and seizure occurrence period should be configurable.

---

# 📊 EEG Frequency Analysis

The system analyzes standard EEG frequency bands:

| Band  | Approx. Frequency |
| ----- | ----------------: |
| Delta |         ~0.5–4 Hz |
| Theta |           ~4–8 Hz |
| Alpha |          ~8–13 Hz |
| Beta  |         ~13–30 Hz |
| Gamma |           ~30+ Hz |

The dashboard may display:

* Power Spectral Density
* FFT-based analysis
* Band power
* Relative band power
* Band ratios
* Spectrogram
* Time-frequency representation

---

# 🧬 Brain Activity Map

The system provides multiple complementary representations.

## 1. 3D Brain

Interactive cortical representation.

## 2. Electrode Map

International 10–20 electrode positions.

## 3. Regional Heatmap

Estimated activity associated with anatomical regions.

## 4. Connectivity Network

Relationships between EEG channels.

This creates a:

```text
Brain Region
     ↕
Electrode
     ↕
EEG Signal
     ↕
Features
     ↕
ML Prediction
```

relationship.

---

# 🔗 EEG Connectivity

NeuroPulse can visualize functional relationships between channels.

Example:

```text
       F3 ●────────● F4
          ╲        ╱
           ╲      ╱
        C3 ●──●──● C4
            ╲│╱
        T3 ●─●─● T4
```

Connectivity metrics may include:

* Correlation
* Coherence
* Phase-based measures
* Other research-appropriate connectivity metrics

The exact method should be documented according to the implemented experiment.

---

# 🤖 Explainable AI

The prediction system should not behave like a black box.

The dashboard should answer:

> **"Why did the model produce this prediction?"**

Possible features:

```text
Theta Power             +24%
Spectral Entropy        +18%
Signal Complexity       +14%
Hjorth Mobility         +11%
Channel Correlation      +9%
```

For tree-based models, SHAP can be used to explain predictions.

For neural networks, appropriate attribution methods may be used.

---

# 🧪 Model Lab

The application includes a model-comparison section.

Potential models:

### Classical ML

* Logistic Regression
* Support Vector Machine
* Random Forest
* XGBoost
* Gradient Boosting
* K-Nearest Neighbors

### Deep Learning

Optional:

* 1D CNN
* CNN-LSTM
* LSTM
* Transformer-based EEG models

The initial project should prioritize strong classical ML baselines before introducing deep learning.

---

# 📈 Model Evaluation

The project should report:

* Accuracy
* Precision
* Recall / Sensitivity
* Specificity
* F1 Score
* ROC-AUC
* PR-AUC
* Confusion Matrix
* False Alarm Rate
* Prediction horizon
* Sensitivity per patient
* Patient-independent performance

For seizure prediction, accuracy alone is insufficient.

False alarms and patient-independent generalization are particularly important.

---

# 📁 Project Structure

```text
neuropulse/
│
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── docker-compose.yml
├── package.json
├── requirements.txt
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   ├── dashboard/
│   │   │   └── page.tsx
│   │   ├── brain/
│   │   │   └── page.tsx
│   │   ├── eeg/
│   │   │   └── page.tsx
│   │   ├── prediction/
│   │   │   └── page.tsx
│   │   ├── models/
│   │   │   └── page.tsx
│   │   ├── history/
│   │   │   └── page.tsx
│   │   └── reports/
│   │       └── page.tsx
│   │
│   ├── components/
│   │   ├── brain/
│   │   │   ├── BrainViewer.tsx
│   │   │   ├── BrainRegion.tsx
│   │   │   ├── ElectrodeMap.tsx
│   │   │   └── ConnectivityGraph.tsx
│   │   │
│   │   ├── eeg/
│   │   │   ├── EEGViewer.tsx
│   │   │   ├── Spectrogram.tsx
│   │   │   ├── FrequencyBands.tsx
│   │   │   └── SignalQuality.tsx
│   │   │
│   │   ├── prediction/
│   │   │   ├── RiskCard.tsx
│   │   │   ├── PredictionTimeline.tsx
│   │   │   └── PredictionExplanation.tsx
│   │   │
│   │   ├── models/
│   │   │   ├── ModelComparison.tsx
│   │   │   ├── MetricsCard.tsx
│   │   │   └── FeatureImportance.tsx
│   │   │
│   │   ├── patient/
│   │   │   ├── PatientOverview.tsx
│   │   │   ├── SeizureHistory.tsx
│   │   │   └── PatientTimeline.tsx
│   │   │
│   │   └── ui/
│   │
│   ├── lib/
│   │   ├── api.ts
│   │   ├── eeg.ts
│   │   ├── brain.ts
│   │   └── utils.ts
│   │
│   ├── hooks/
│   ├── types/
│   ├── public/
│   │   ├── brain/
│   │   └── models/
│   └── styles/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   │
│   │   ├── api/
│   │   │   ├── eeg.py
│   │   │   ├── prediction.py
│   │   │   ├── patients.py
│   │   │   ├── models.py
│   │   │   └── reports.py
│   │   │
│   │   ├── services/
│   │   │   ├── eeg_service.py
│   │   │   ├── prediction_service.py
│   │   │   ├── feature_service.py
│   │   │   └── report_service.py
│   │   │
│   │   ├── schemas/
│   │   ├── utils/
│   │   └── database/
│   │
│   └── tests/
│
├── ml/
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── metadata/
│   │
│   ├── preprocessing/
│   │   ├── filtering.py
│   │   ├── artifact_removal.py
│   │   └── segmentation.py
│   │
│   ├── features/
│   │   ├── temporal.py
│   │   ├── spectral.py
│   │   ├── nonlinear.py
│   │   ├── hjorth.py
│   │   └── connectivity.py
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   ├── random_forest.py
│   │   ├── svm.py
│   │   ├── xgboost_model.py
│   │   └── deep_learning/
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── cross_validation.py
│   │   └── hyperparameter_search.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── confusion_matrix.py
│   │   └── patient_wise.py
│   │
│   ├── explainability/
│   │   ├── shap_analysis.py
│   │   └── feature_attribution.py
│   │
│   └── notebooks/
│
├── models/
│   ├── trained/
│   ├── scalers/
│   └── metadata/
│
├── reports/
│   ├── figures/
│   ├── model_results/
│   └── generated/
│
└── docs/
    ├── architecture.md
    ├── ml-methodology.md
    ├── dataset.md
    ├── api.md
    └── limitations.md
```

---

# 🧠 Machine Learning Pipeline

```text
                    EEG DATA
                       │
                       ▼
              Data Validation
                       │
                       ▼
              Signal Preprocessing
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
          Filtering          Artifacts
             │                   │
             └─────────┬─────────┘
                       ▼
                 Segmentation
                       │
                       ▼
                Feature Extraction
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    Temporal       Spectral       Nonlinear
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                 Feature Matrix
                       │
                       ▼
                  ML Models
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
          SVM          RF        XGBoost
           │           │           │
           └───────────┼───────────┘
                       ▼
                 Risk Prediction
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Explanation          Visualization
```

---

# 🔬 EEG Processing Pipeline

## 1. Load EEG

Input formats:

```text
EDF
CSV
NumPy
```

Primary research format:

```text
EDF
```

---

## 2. Preprocessing

Potential operations:

* Band-pass filtering
* Notch filtering
* Resampling
* Artifact handling
* Channel selection
* Normalization

All preprocessing parameters should be recorded.

---

# 🪟 Window Segmentation

EEG recordings are divided into windows.

Example:

```text
Recording
────────────────────────────────────────────

Window 1
██████████

Window 2
     ██████████

Window 3
          ██████████
```

Window size and overlap must be configurable.

---

# 🧮 Feature Engineering

## Time-domain features

Potential features:

* Mean
* Standard deviation
* Variance
* RMS
* Peak-to-peak amplitude
* Line length
* Zero-crossing rate

## Frequency-domain features

Potential features:

* Delta power
* Theta power
* Alpha power
* Beta power
* Gamma power
* Relative band power
* Spectral entropy
* Dominant frequency

## Nonlinear features

Potential features:

* Approximate entropy
* Sample entropy
* Hjorth parameters
* Signal complexity

## Connectivity features

Potential features:

* Correlation
* Coherence
* Phase-based connectivity

Only features actually implemented should be reported in the final experiment.

---

# 🔮 Prediction Architecture

The prediction system should distinguish between:

```text
Interictal
Preictal
Seizure
```

depending on the experimental labeling strategy.

The output may be represented as:

```json
{
  "risk_score": 0.73,
  "state": "elevated_preictal_risk",
  "prediction_horizon_minutes": 20,
  "model": "xgboost",
  "top_features": [
    "theta_power",
    "spectral_entropy",
    "channel_connectivity"
  ]
}
```

---

# 🧠 Regional Activity Estimation

The dashboard maps EEG channel-level information to broader cortical regions.

Example:

```text
T3 + T5
   ↓
Temporal region
   ↓
Elevated activity estimate
```

This is a **visual/analytical association**, not direct cortical source localization.

The UI should therefore use terms such as:

* Estimated region
* Associated region
* EEG activity distribution
* Regional activity estimate

and avoid unsupported claims such as:

* Exact seizure origin
* Exact brain lesion
* Guaranteed seizure source

---

# 👨‍⚕️ Clinical / Research Mode

The advanced interface should expose:

### Recording

* Patient ID
* Recording ID
* Duration
* Sampling rate
* Number of channels
* Montage

### EEG

* Raw waveform
* Filtered waveform
* Spectrogram
* PSD
* Channel analysis
* Signal quality

### Prediction

* Risk score
* Prediction horizon
* State classification
* Threshold
* Temporal probability curve

### Explainability

* SHAP
* Feature importance
* Feature contribution

### Model

* Model type
* Model version
* Training dataset
* Validation strategy
* Metrics

---

# 🧑 Patient Mode

Patient Mode intentionally hides technical complexity.

Example:

```text
YOUR BRAIN STATE

Stable

────────────────

Seizure Risk

Elevated

73%

────────────────

What does this mean?

The system found EEG patterns
that resemble patterns observed
before seizures in the research
dataset.

────────────────

Where?

Left temporal region

────────────────

Why?

The EEG shows changes in
activity and synchronization.
```

A persistent disclaimer should indicate that this is a research prototype.

---

# 📄 Reporting

The system should support generating a research-oriented report.

Report sections:

```text
Patient / Recording Information
        ↓
Signal Quality
        ↓
EEG Summary
        ↓
Frequency Analysis
        ↓
Prediction Results
        ↓
Regional Activity
        ↓
Feature Importance
        ↓
Model Information
        ↓
Evaluation Context
        ↓
Limitations
```

---

# 📚 Datasets

## Primary Dataset

### CHB-MIT Scalp EEG Database

The project can initially use the **CHB-MIT Scalp EEG Database**.

The dataset contains annotated scalp EEG recordings and seizure events and is widely used in academic seizure-analysis research.

Source:

https://physionet.org/content/chbmit/

---

# ⚠️ Dataset Considerations

The dataset has important limitations.

These include:

* Limited number of subjects
* Pediatric population
* Patient-specific characteristics
* Class imbalance
* Limited seizure events
* Dataset-specific recording characteristics

Therefore, strong performance on CHB-MIT should **not** automatically be interpreted as general clinical performance.

---

# 🧪 Experimental Methodology

The project should prioritize avoiding data leakage.

A naive approach such as:

```text
Randomly split EEG windows
       ↓
Train / Test
```

can produce overly optimistic results because neighboring windows from the same patient may be highly correlated.

A stronger approach is:

```text
Patient A ───── Training
Patient B ───── Training
Patient C ───── Validation
Patient D ───── Testing
```

where appropriate.

The exact validation strategy must be documented.

---

# 📊 Evaluation

At minimum:

```text
Accuracy
Precision
Recall
Specificity
F1
ROC-AUC
PR-AUC
Confusion Matrix
False Alarm Rate
```

For seizure prediction, also report:

```text
Prediction Horizon
Seizure Occurrence Period
Sensitivity
False Prediction Rate
Patient-wise performance
```

---

# 🖥️ Frontend Architecture

Recommended frontend:

```text
Next.js
TypeScript
Tailwind CSS
shadcn/ui
React Three Fiber
Three.js
Recharts
Framer Motion
```

---

# ⚙️ Backend Architecture

Recommended backend:

```text
Python
FastAPI
Pydantic
MNE
NumPy
SciPy
Pandas
Scikit-learn
XGBoost
PyTorch
SHAP
```

---

# 🔌 API Design

## Upload EEG

```http
POST /api/eeg/upload
```

## Analyze EEG

```http
POST /api/eeg/analyze
```

## Predict

```http
POST /api/prediction/predict
```

## Patient

```http
GET /api/patients/{id}
```

## Prediction History

```http
GET /api/patients/{id}/predictions
```

## Model Metrics

```http
GET /api/models/metrics
```

## Generate Report

```http
POST /api/reports/generate
```

---

# 🛠️ Technology Stack

| Layer             | Technology                   |
| ----------------- | ---------------------------- |
| Frontend          | Next.js                      |
| Language          | TypeScript                   |
| UI                | Tailwind CSS                 |
| Components        | shadcn/ui                    |
| 3D                | Three.js / React Three Fiber |
| Charts            | Recharts                     |
| Animation         | Framer Motion                |
| Backend           | FastAPI                      |
| ML                | Scikit-learn                 |
| Gradient Boosting | XGBoost                      |
| Deep Learning     | PyTorch                      |
| EEG               | MNE-Python                   |
| Signal Processing | SciPy                        |
| Explainability    | SHAP                         |
| Data              | NumPy / Pandas               |
| Containerization  | Docker                       |

---

# 🚀 Installation

## Clone

```bash
git clone <repository-url>
cd neuropulse
```

## Backend

```bash
cd backend

python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r ../requirements.txt
```

Run:

```bash
uvicorn app.main:app --reload
```

---

# Frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

Backend:

```text
http://localhost:8000
```

---

# 🐳 Docker

The project can optionally be run using:

```bash
docker compose up --build
```

---

# ⚙️ Configuration

Create:

```text
.env
```

from:

```text
.env.example
```

Example:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000

MODEL_PATH=./models/trained/model.pkl

EEG_SAMPLE_RATE=256

PREDICTION_HORIZON_MINUTES=20
```

---

# 🔄 Example Workflow

```text
1. User uploads EEG
        ↓
2. Backend validates recording
        ↓
3. MNE loads EDF
        ↓
4. Signal preprocessing
        ↓
5. Window segmentation
        ↓
6. Feature extraction
        ↓
7. ML prediction
        ↓
8. Risk score generated
        ↓
9. Feature explanation generated
        ↓
10. EEG visualization updated
        ↓
11. Brain activity visualization updated
        ↓
12. Prediction timeline updated
        ↓
13. User receives human-readable explanation
```

---

# 🧪 Development Phases

## Phase 1 — Dataset

* Obtain dataset
* Understand EDF structure
* Load EEG using MNE
* Identify seizure annotations
* Explore recordings

## Phase 2 — Signal Processing

* Filtering
* Windowing
* Artifact handling
* Signal visualization

## Phase 3 — Feature Engineering

* Time-domain features
* Frequency-domain features
* Nonlinear features

## Phase 4 — Baseline Models

Implement:

```text
Logistic Regression
Random Forest
SVM
XGBoost
```

## Phase 5 — Evaluation

Implement:

```text
Patient-wise validation
Confusion matrix
ROC
PR curve
False alarm analysis
```

## Phase 6 — Explainability

Implement:

```text
SHAP
Feature importance
Prediction explanation
```

## Phase 7 — Backend

Create FastAPI endpoints.

## Phase 8 — Dashboard

Build:

```text
Overview
EEG
Brain
Prediction
Models
History
Reports
```

## Phase 9 — 3D Visualization

Integrate:

```text
3D Brain
Electrodes
Regional activity
Connectivity
```

## Phase 10 — Integration

Connect:

```text
Frontend
     ↕
FastAPI
     ↕
ML Pipeline
     ↕
Trained Model
```

---

# 🚧 Research Limitations

NeuroPulse is a student research project.

Important limitations include:

### Dataset limitations

The training dataset may not represent the general population.

### Patient generalization

Models trained on one group of patients may perform differently on unseen patients.

### False alarms

A practical seizure prediction system must minimize false alarms.

### Prediction uncertainty

A model output is not a guarantee that a seizure will occur.

### Brain localization

Scalp EEG does not directly provide exact cortical source localization.

### Clinical validation

The system has not undergone clinical validation.

---

# 🔮 Future Improvements

Potential future research directions:

* Larger multi-center datasets
* Adult datasets
* Patient-independent learning
* Cross-dataset validation
* Self-supervised EEG representation learning
* Transformer architectures
* CNN-LSTM models
* Graph neural networks
* Real-time EEG streaming
* Personalized seizure prediction
* Domain adaptation
* Federated learning
* Better false-alarm suppression
* Source localization
* MRI/CT integration
* Multimodal neurological analysis

---

# 🏆 Project Goal

NeuroPulse is designed to demonstrate how:

```text
Machine Learning
        +
Signal Processing
        +
Explainable AI
        +
Neuroscience
        +
3D Visualization
```

can be combined into a single end-to-end system.

The ultimate goal is not merely to produce a prediction score.

It is to make the prediction:

> **understandable, explainable, visually interpretable, and scientifically responsible.**

---

# ⚠️ Disclaimer

NeuroPulse is an **academic and research prototype**.

It is not intended to:

* Diagnose epilepsy
* Replace a neurologist
* Provide emergency warnings
* Determine the exact origin of a seizure
* Provide medical advice
* Make clinical treatment decisions

All predictions are experimental machine-learning outputs and should not be interpreted as medical diagnoses.

---


