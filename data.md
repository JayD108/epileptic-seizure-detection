NeuroPulse Data Pipeline: Acquisition & Ingestion
This module manages the automated downloading, structuring, and preprocessing of the CHB-MIT Scalp EEG Database from PhysioNet. It transforms raw, continuous .edf (European Data Format) brainwave recordings into structured, numerical feature matrices ready for machine learning models.

📌 Prerequisites
Ensure your Python environment is active and the required signal processing and ingestion libraries are installed:

Bash
pip install requests mne numpy scipy pandas
📂 Directory Structure
Running the ingestion pipeline will automatically generate the following structure:

Plaintext
neuropulse/
└── ml/
    └── data/
        ├── raw/                # Unaltered .edf and summary .txt files
        │   ├── chb01/
        │   ├── chb02/
        │   └── ...
        ├── processed/          # Extracted CSV feature matrices for ML
        │   ├── chb01_features.csv
        │   └── ...
        └── metadata/           # Parsed clinical summaries and seizure times
📥 Step 1: Downloading the Dataset
The CHB-MIT dataset is hosted on PhysioNet and contains over 40GB of EEG data across 24 pediatric patients. The script below targets specific patients to avoid overwhelming local storage during development.

Create a file named download_data.py in your project root:

Python
import os
import requests
import re

BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"
RAW_DIR = "./ml/data/raw"

def download_file(url, save_path):
    """Downloads a single file in chunks with a progress indicator."""
    if os.path.exists(save_path):
        print(f"File already exists: {save_path}")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Successfully downloaded: {save_path}")

def get_patient_files(patient_id):
    """Scrapes the PhysioNet directory page to find all files for a patient."""
    patient_url = f"{BASE_URL}{patient_id}/"
    response = requests.get(patient_url)
    
    # Extract all .edf and .txt files from the HTML directory listing
    files = re.findall(r'href="([^"]+\.(?:edf|txt))"', response.text)
    return files

def download_patient_data(patient_id):
    """Downloads all EDF recordings and summaries for a specific patient ID."""
    print(f"--- Fetching index for {patient_id} ---")
    files = get_patient_files(patient_id)
    
    if not files:
        print(f"No files found for {patient_id}. Check the ID format (e.g., 'chb01').")
        return

    for file_name in files:
        file_url = f"{BASE_URL}{patient_id}/{file_name}"
        save_path = os.path.join(RAW_DIR, patient_id, file_name)
        print(f"Fetching {file_name}...")
        download_file(file_url, save_path)

if __name__ == "__main__":
    # Example: Download all data for Patient 01
    target_patients = ["chb01"] 
    
    for patient in target_patients:
        download_patient_data(patient)
To run: python download_data.py

⚙️ Step 2: Data Ingestion & Feature Extraction
Once the .edf files are downloaded, they must be ingested, cleaned, and converted into machine-learning features (Relative Band Power per channel).

Create ingest_eeg.py:

Python
import os
import mne
import numpy as np
import pandas as pd
from scipy.integrate import simps
import warnings

# Suppress MNE verbosity for cleaner console output
mne.set_log_level('WARNING')
warnings.filterwarnings('ignore')

RAW_DIR = "./ml/data/raw"
PROCESSED_DIR = "./ml/data/processed"

class EEGProcessor:
    def __init__(self, window_size_sec=10):
        self.window_size_sec = window_size_sec
        self.bands = {
            'Delta': (0.5, 4), 'Theta': (4, 8), 
            'Alpha': (8, 13), 'Beta':  (13, 30), 'Gamma': (30, 50)
        }

    def process_file(self, edf_path):
        """Loads EDF, filters noise, and extracts PSD features."""
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True)
            
            # Standardize channels (Targeting the 10-20 system)
            valid_channels = [ch for ch in raw.ch_names if '-' in ch and ch != '-']
            raw.pick_channels(valid_channels[:18])
            
            # Preprocessing: Band-pass (0.5-50Hz) and Notch (60Hz Line Noise)
            raw.filter(l_freq=0.5, h_freq=50.0)
            raw.notch_filter(freqs=60.0)
            
            # Epoching (Segmentation)
            events = mne.make_fixed_length_events(raw, duration=self.window_size_sec)
            epochs = mne.Epochs(raw, events, tmin=0, tmax=self.window_size_sec, baseline=None, preload=True)
            
            # Feature Extraction (Welch's PSD)
            psds, freqs = epochs.compute_psd(method='welch', fmin=0.5, fmax=50.0, n_jobs=1).get_data(return_freqs=True)
            
            features = []
            for epoch_idx in range(psds.shape[0]):
                epoch_features = {'epoch_id': epoch_idx, 'file_source': os.path.basename(edf_path)}
                for ch_idx, ch_name in enumerate(epochs.ch_names):
                    channel_psd = psds[epoch_idx, ch_idx, :]
                    total_power = simps(channel_psd, dx=freqs[1] - freqs[0])
                    
                    for band_name, (fmin, fmax) in self.bands.items():
                        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                        band_power = simps(channel_psd[idx_band], dx=freqs[1] - freqs[0])
                        rel_power = band_power / total_power if total_power > 0 else 0
                        epoch_features[f"{ch_name}_{band_name}"] = rel_power
                features.append(epoch_features)
                
            return pd.DataFrame(features)
        except Exception as e:
            print(f"Error processing {edf_path}: {e}")
            return pd.DataFrame()

def ingest_patient(patient_id):
    """Processes all downloaded EDF files for a given patient."""
    patient_dir = os.path.join(RAW_DIR, patient_id)
    if not os.path.exists(patient_dir):
        print(f"No raw data found for {patient_id}. Run download_data.py first.")
        return

    edf_files = [f for f in os.listdir(patient_dir) if f.endswith('.edf')]
    processor = EEGProcessor(window_size_sec=10)
    
    all_features = []
    for edf in edf_files:
        print(f"Processing {edf}...")
        df = processor.process_file(os.path.join(patient_dir, edf))
        if not df.empty:
            all_features.append(df)
            
    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        save_path = os.path.join(PROCESSED_DIR, f"{patient_id}_features.csv")
        final_df.to_csv(save_path, index=False)
        print(f"✅ Ingestion complete. Saved feature matrix to {save_path} (Shape: {final_df.shape})")

if __name__ == "__main__":
    ingest_patient("chb01")
To run: python ingest_eeg.py

⚠️ Important Considerations
Storage Limits: Downloading all 24 patients consumes ~43GB of disk space. Begin with chb01 for development and testing.

Clinical Summaries (.txt files): The PhysioNet directory includes a summary text file for each patient (e.g., chb01-summary.txt). This file contains the exact start and end times in seconds for every seizure. You will need to parse this file in the next phase to map labels (Normal, Preictal, Seizure) to the generated feature matrices.

RAM Usage: The preload=True argument loads entire 1-hour EEG files into memory. If you experience memory crashes on constrained hardware, change this to preload=False, though feature extraction will be significantly slower.
