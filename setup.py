# setup.py
from setuptools import setup, find_packages

setup(
    name="neuromorphic-plasma-anomaly-detection",
    version="0.1.0",
    description="Hybrid Neuromorphic-CNN-LSTM for Real-Time Anomaly Detection in Deep-Space Plasma Turbulence",
    author="NASA Data Science Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.3.0",
        "snntorch>=0.7.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "streamlit>=1.28.0",
        "gradio>=4.0.0",
        "plasmapy>=2023.10.0",
        "astropy>=5.3.0",
        "scikit-learn>=1.3.0",
        "mlflow>=2.8.0",
        "tqdm>=4.65.0",
        "h5py>=3.9.0",
        "plotly>=5.17.0",
    ],
    python_requires=">=3.12",
)