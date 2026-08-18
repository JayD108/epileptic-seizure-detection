# 🧠 Automated Seizure Detection from EEG Signals

This repository contains a Machine Learning pipeline designed to detect and classify epileptic seizures using Electroencephalography (EEG) data. 

## 📖 Domain Understanding

Before diving into the code, it is essential to understand the biological and clinical concepts driving this project.

### What is EEG?
Electroencephalography (EEG) is a non-invasive medical imaging technique that records the continuous electrical activity of the brain. Electrodes are placed on the scalp to detect tiny electrical charges resulting from the activity of brain cells (neurons). 
* **Signal Output:** A continuous time-series signal representing brainwaves.
* **Standardization:** Electrodes are positioned using standardized anatomical maps (such as the 10-20 system) to ensure consistency across different subjects.

### What is Epilepsy and What are Seizures?
* **Epilepsy:** A central nervous system (neurological) disorder in which brain activity becomes abnormal, causing periods of unusual behavior, sensations, and sometimes loss of awareness.
* **Seizures:** A seizure is a sudden, uncontrolled electrical disturbance in the brain. In an EEG recording, a seizure typically manifests as a sudden onset of rhythmic, high-amplitude, and high-frequency wave patterns (spikes and sharp waves) that stand out distinctly from normal baseline background activity.

---

## 📊 Dataset Parameters and Terminology

EEG datasets contain specific metadata and streams that our data loaders must parse. 

| Parameter | Description | Relevance to ML Pipeline |
| :--- | :--- | :--- |
| **Sampling Frequency (fs)** | Points recorded per second per channel (e.g., 250 Hz or 512 Hz). | Dictates the input length of our tensors. All files must be resampled to a consistent frequency before batching. |
| **Channels/Electrodes** | Scalp locations (e.g., Fz, Cz, Pz). | Acts as the spatial dimension (conceptually similar to color channels in computer vision). |
| **Frequency Bands** | Brainwaves grouped by frequency (Delta: 0.5-4 Hz, Theta: 4-8 Hz, Alpha: 8-13 Hz, Beta: 13-30 Hz). | Used for feature extraction; seizures often cause massive energy shifts in specific bands. |
| **Annotations / Triggers** | Timestamps marking clinical events. | Serves as our ground-truth labels for supervised learning (e.g., `0 = Normal`, `1 = Seizure Onset`). |

---

## ⚙️ How We Use This in Machine Learning

Raw EEG data is incredibly noisy. Building a robust classifier requires a structured pipeline that transforms these raw electrical signals into clean, actionable tensors.

### 1. Data Preprocessing
Biological signals contain artifacts (eye blinks, muscle movement) and environmental noise (power line interference).
* **Bandpass Filtering:** Restricting the signal to a biologically relevant range (e.g., 1 Hz to 40 Hz) to remove baseline drift and high-frequency noise.
* **Notch Filtering:** Removing specific power-line frequencies (50 Hz or 60 Hz).
* **Windowing (Epoching):** Slicing the continuous EEG recording into fixed, overlapping time windows (e.g., 2-second segments) to feed into the model.

### 2. Feature Engineering vs. Deep Learning
Depending on the architecture, we approach the data in one of two ways:
* **Traditional ML:** Extracting Power Spectral Density (PSD) or specific frequency band powers from the signal, flattening them, and passing them to classifiers.
* **Deep Learning (End-to-End):** Passing the raw (but filtered) 2D matrices `(Channels × Timepoints)` directly into neural networks to let the model learn the feature representations.

### 3. Model Architecture & Hardware Focus
For this project, the deep learning approach utilizes spatial-temporal networks built in PyTorch. 
* **Architecture:** Models like EEGNet (a compact Convolutional Neural Network) or CNN-LSTM hybrids are highly effective. The CNN extracts spatial features across electrodes, while the LSTM captures the temporal dynamics of the seizure.
* **Local Compute Acceleration:** Time-series data with high sampling rates creates extremely large tensors. The training loop is optimized to leverage local CUDA-enabled hardware (e.g., RTX 5080) for rapid prototyping and fully local fine-tuning, eliminating the need for cloud computation.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* PyTorch (with CUDA support)
* MNE-Python (for EEG data processing)

*(Add your specific installation instructions and run commands here)*
