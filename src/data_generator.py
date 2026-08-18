import numpy as np
import torch

def generate_synthetic_eeg(num_samples_per_class=100, fs=250, duration=2, num_channels=8):
    """
    Generates synthetic EEG data.
    Class 0 (Normal): 10 Hz & 20 Hz sine waves + Gaussian noise.
    Class 1 (Seizure): 4.5 Hz rhythmic spikes + 35 Hz high gamma waves + Gaussian noise.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    num_timepoints = len(t)
    
    X = np.zeros((num_samples_per_class * 2, num_channels, num_timepoints))
    y = np.zeros(num_samples_per_class * 2)
    
    # Generate Class 0 (Normal)
    for i in range(num_samples_per_class):
        for c in range(num_channels):
            wave_10hz = np.sin(2 * np.pi * 10 * t)
            wave_20hz = 0.5 * np.sin(2 * np.pi * 20 * t)
            noise = np.random.normal(0, 0.5, num_timepoints)
            X[i, c, :] = wave_10hz + wave_20hz + noise
        y[i] = 0

    # Generate Class 1 (Seizure)
    for i in range(num_samples_per_class, num_samples_per_class * 2):
        for c in range(num_channels):
            # Simulate high-amplitude spikes
            spikes_4_5hz = 3 * np.sin(2 * np.pi * 4.5 * t)
            spikes_4_5hz[spikes_4_5hz < 2] = 0 # Make them spiky
            gamma_35hz = 1.5 * np.sin(2 * np.pi * 35 * t)
            noise = np.random.normal(0, 0.5, num_timepoints)
            X[i, c, :] = spikes_4_5hz + gamma_35hz + noise
        y[i] = 1

    # Shuffle
    indices = np.random.permutation(len(y))
    X, y = X[indices], y[indices]

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def generate_single_window(is_seizure=False, fs=250, duration=2, num_channels=8):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    num_timepoints = len(t)
    X = np.zeros((1, num_channels, num_timepoints))
    
    for c in range(num_channels):
        if is_seizure:
            spikes_4_5hz = 3 * np.sin(2 * np.pi * 4.5 * t)
            spikes_4_5hz[spikes_4_5hz < 2] = 0
            gamma_35hz = 1.5 * np.sin(2 * np.pi * 35 * t)
            noise = np.random.normal(0, 0.5, num_timepoints)
            X[0, c, :] = spikes_4_5hz + gamma_35hz + noise
        else:
            wave_10hz = np.sin(2 * np.pi * 10 * t)
            wave_20hz = 0.5 * np.sin(2 * np.pi * 20 * t)
            noise = np.random.normal(0, 0.5, num_timepoints)
            X[0, c, :] = wave_10hz + wave_20hz + noise
            
    return torch.tensor(X, dtype=torch.float32)
