# 🧠 Epileptic Seizure Detection using Machine Learning

## 📌 Project Overview
This project applies signal processing and machine learning algorithms to Electroencephalogram (EEG) time-series data to automatically detect epileptic seizure activity. The goal is to build an accurate classification model that assists in rapid clinical analysis.

## 📊 Dataset
- **Primary Source:** CHB-MIT Scalp EEG Database (PhysioNet) / UCI Epileptic Seizure Recognition Dataset
- **Format:** Multi-channel EEG time-series signals (`.edf` or tabulated formats)
- **Target:** Classification of brain states (e.g., Inter-ictal vs. Pre-ictal / Seizure vs. Non-Seizure)

## 🛠️ Tech Stack & Dependencies
- **Language:** Python 3.x
- **Libraries:** NumPy, Pandas, Scikit-Learn, MNE-Python, Matplotlib, Seaborn
- **Frameworks:** PyTorch / TensorFlow (for Deep Learning implementations)

## ⚙️ Pipeline
1. **Data Preprocessing:** Signal filtering (Bandpass filters), artifact removal, and standardization.
2. **Feature Extraction / Embeddings:** Time-domain, frequency-domain features, or 1D spatial-temporal convolutions.
3. **Model Development:** Training baseline models (Random Forest, SVM) alongside Deep Learning classifiers (e.g., EEGNet, 1D CNN).
4. **Evaluation:** Performance evaluation using Precision, Recall, F1-Score, and ROC-AUC metrics.

## 🚀 How to Run
```bash
# Clone the repository
git clone [https://github.com/YOUR-USERNAME/epileptic-seizure-detection.git](https://github.com/YOUR-USERNAME/epileptic-seizure-detection.git)

# Install requirements
pip install -r requirements.txt
