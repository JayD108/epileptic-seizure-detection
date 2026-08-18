import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from src.data_generator import generate_single_window
from src.model import MiniEEGNet, get_device
from src.train import train_model

st.set_page_config(page_title="Clinical EEG Dashboard", layout="wide")

# Custom CSS for minimalist layout
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }
    .status-card {
        padding: 20px;
        border-radius: 5px;
        text-align: center;
        font-family: sans-serif;
        margin-top: 20px;
    }
    .status-alert {
        background-color: #dc3545;
        color: #ffffff;
        border: 1px solid #b02a37;
    }
    .status-normal {
        background-color: #198754;
        color: #ffffff;
        border: 1px solid #146c43;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Automated Epileptic Seizure Detection System")
st.markdown("---")

# --- Top Section: Clinical EEG Viewer ---
st.header("Real-Time Clinical Electroencephalogram")

# Generate standard data to view or use injected data
if 'current_window' not in st.session_state:
    st.session_state['current_window'] = generate_single_window(is_seizure=False)
    st.session_state['is_seizure_injected'] = False

window_tensor = st.session_state['current_window']
signal_data = window_tensor.squeeze().numpy()

fs = 250
duration = 2
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

fig, ax = plt.subplots(figsize=(15, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

channels = ['FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
offset_step = 10

for i in range(8):
    # Offset signals to stack them vertically
    ax.plot(t, signal_data[i, :] + (7 - i) * offset_step, color='#0a192f', linewidth=0.8)

# Grid and formatting
ax.xaxis.grid(True, linestyle='-', linewidth=0.5, color='#a8e6cf')
ax.set_xticks(np.arange(0, duration + 0.1, 1.0))
ax.set_xlim(0, duration)

ax.set_yticks(np.arange(0, 8 * offset_step, offset_step))
ax.set_yticklabels(reversed(channels))
ax.set_ylim(-offset_step, 8 * offset_step)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#ced4da')
ax.tick_params(axis='y', length=0)
ax.set_xlabel("Time (Seconds)", fontweight='bold')

st.pyplot(fig)

st.markdown("---")

# --- Bottom Section: Split Controls ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Engine Settings")
    st.markdown("Configure hyperparameters and initialize training sequence.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10, step=1)
    with c2:
        lr = st.selectbox("Learning Rate", options=[0.01, 0.005, 0.001, 0.0005, 0.0001], index=2)
    with c3:
        batch_size = st.selectbox("Batch Size", options=[8, 16, 32, 64], index=2)
        
    start_training = st.button("Initialize & Train Model", type="primary", use_container_width=True)
    
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    if start_training:
        progress_bar = st.progress(0)
        loss_history = []
        
        for epoch, loss, acc, model in train_model(epochs=epochs, lr=lr, batch_size=batch_size):
            loss_history.append(loss)
            chart_placeholder.line_chart(pd.DataFrame(loss_history, columns=['Loss']), height=200)
            metrics_placeholder.text(f"Epoch: {epoch:02d}/{epochs} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")
            progress_bar.progress(epoch / epochs)
            
        st.session_state['trained_model'] = model
        st.success("Training Sequence Complete.")

with col2:
    st.subheader("Real-Time Inference Monitor")
    st.markdown("Inject simulated clinical data into the detection pipeline.")
    
    signal_type = st.radio("Signal Injection Mode", ("Normal Baseline", "Pre-Ictal Simulation"), horizontal=True)
    inject_signal = st.button("Inject Live Signal", use_container_width=True)
    
    if inject_signal:
        is_seizure = (signal_type == "Pre-Ictal Simulation")
        st.session_state['current_window'] = generate_single_window(is_seizure=is_seizure)
        st.session_state['is_seizure_injected'] = is_seizure
        st.rerun()
        
    st.markdown("### Clinical Diagnostics")
    
    if 'trained_model' not in st.session_state:
        st.info("System awaiting model initialization.")
    else:
        model = st.session_state['trained_model']
        device = get_device()
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            input_tensor = st.session_state['current_window'].to(device)
            prediction = model(input_tensor).item()
            
        threshold = 0.5
        is_predicted_seizure = prediction >= threshold
        
        if is_predicted_seizure:
            st.markdown(
                f"""
                <div class="status-card status-alert">
                    <h2 style="color: white; margin: 0;">CLINICAL ALERT: PRE-ICTAL PATTERN DETECTED</h2>
                    <h4 style="color: white; margin-top: 10px;">Confidence Probability: {prediction:.2%}</h4>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="status-card status-normal">
                    <h2 style="color: white; margin: 0;">BASELINE NORMAL</h2>
                    <h4 style="color: white; margin-top: 10px;">Confidence Probability: {1-prediction:.2%}</h4>
                </div>
                """, 
                unsafe_allow_html=True
            )
