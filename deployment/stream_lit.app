# deployment/streamlit_app.py
"""Streamlit demo application for plasma anomaly detection."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.hybrid_model import HybridNeuromorphicModel
from data.synthetic import SyntheticPlasmaGenerator

st.set_page_config(
    page_title="Neuromorphic Plasma Anomaly Detection",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Hybrid Neuromorphic-CNN-LSTM for Deep-Space Plasma Anomaly Detection")
st.markdown("Real-time anomaly detection in deep-space plasma turbulence using neuromorphic computing")

@st.cache_resource
def load_model():
    """Load the trained model."""
    model = HybridNeuromorphicModel()
    # Load pre-trained weights if available
    try:
        checkpoint = torch.load("models/neuromorphic_plasma_detector.pt", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        st.warning("Pre-trained model not found. Using randomly initialized model for demonstration.")
    model.eval()
    return model

@st.cache_resource
def get_data_generator():
    """Get synthetic data generator."""
    return SyntheticPlasmaGenerator()

def main():
    model = load_model()
    generator = get_data_generator()
    
    st.sidebar.header("Control Panel")
    
    # Data generation controls
    st.sidebar.subheader("Data Generation")
    include_anomalies = st.sidebar.checkbox("Include Anomalies", value=True)
    anomaly_type = st.sidebar.selectbox("Anomaly Type", ["spike", "dropout", "oscillation"])
    
    if st.sidebar.button("Generate New Data"):
        # Generate synthetic data
        sequences, labels = generator.generate_batch(1, include_anomalies=include_anomalies)
        spectrograms = generator.generate_spectrograms(sequences)
        
        # Store in session state
        st.session_state.sequences = sequences
        st.session_state.spectrograms = spectrograms
        st.session_state.labels = labels
    
    # Initialize data if not exists
    if 'sequences' not in st.session_state:
        sequences, labels = generator.generate_batch(1, include_anomalies=True)
        spectrograms = generator.generate_spectrograms(sequences)
        st.session_state.sequences = sequences
        st.session_state.spectrograms = spectrograms
        st.session_state.labels = labels
    
    # Get data from session state
    sequences = st.session_state.sequences
    spectrograms = st.session_state.spectrograms
    labels = st.session_state.labels
    
    # Model inference
    with torch.no_grad():
        results = model.detect_anomalies(sequences, spectrograms)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Anomaly Probability", f"{results['anomaly_probabilities'][0]:.3f}")
    
    with col2:
        st.metric("Inference Time", f"{results['inference_time_ms']:.2f} ms")
    
    with col3:
        meets_target = results['inference_time_ms'] < 1.0
        st.metric("Latency Target", "✅ Met" if meets_target else "❌ Missed")
    
    # Visualization
    st.subheader("Plasma Parameters")
    
    # Plot time series
    fig = go.Figure()
    param_names = ['Density', 'Vel_X', 'Vel_Y', 'Vel_Z', 'Bx', 'By', 'Bz', 'Energy']
    
    for i, name in enumerate(param_names):
        fig.add_trace(go.Scatter(
            y=sequences[0, :, i].numpy(),
            name=name,
            line=dict(width=1)
        ))
    
    fig.update_layout(
        title="Plasma Parameters Over Time",
        xaxis_title="Time Steps",
        yaxis_title="Normalized Value",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly detection visualization
    st.subheader("Anomaly Detection Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly probability over time
        if results['attention_weights'] is not None:
            attention = results['attention_weights'][0].numpy()
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=attention, name="Attention Weights"))
            fig.update_layout(title="Temporal Attention Weights", height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance metrics
        performance = model.get_performance_metrics()
        if performance:
            st.write("**Model Performance:**")
            for key, value in performance.items():
                if isinstance(value, bool):
                    st.write(f"- {key}: {'✅' if value else '❌'}")
                else:
                    st.write(f"- {key}: {value:.3f}")

if __name__ == "__main__":
    main()