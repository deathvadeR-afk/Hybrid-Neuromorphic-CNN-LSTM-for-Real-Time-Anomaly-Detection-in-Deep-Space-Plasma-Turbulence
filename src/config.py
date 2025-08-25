# src/config.py
"""Configuration settings for the neuromorphic plasma anomaly detection system."""

import os
from dataclasses import dataclass
from typing import Tuple, List
import torch

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # CNN Configuration
    cnn_channels: List[int] = None
    cnn_kernel_sizes: List[int] = None
    cnn_dropout: float = 0.2
    
    # LSTM Configuration
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    
    # SNN Configuration
    snn_num_neurons: int = 100
    snn_beta: float = 0.95  # Decay rate for LIF neurons
    snn_threshold: float = 1.0
    snn_spike_grad_type: str = "fast_sigmoid"
    
    # Input/Output dimensions
    sequence_length: int = 1000
    num_features: int = 8  # density, velocity_x, velocity_y, velocity_z, Bx, By, Bz, energy
    spectrogram_height: int = 64
    spectrogram_width: int = 128
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 3, 3]

@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 50
    weight_decay: float = 1e-5
    
    # Loss weights
    anomaly_loss_weight: float = 1.0
    reconstruction_loss_weight: float = 0.5
    
    # Optimization
    optimizer: str = "adam"
    scheduler: str = "cosine"
    early_stopping_patience: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpointing
    save_every_n_epochs: int = 5
    checkpoint_dir: str = "checkpoints"

@dataclass
class DataConfig:
    """Data processing configuration."""
    
    # Data paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    synthetic_data_dir: str = "data/synthetic"
    
    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Preprocessing
    normalize_method: str = "zscore"  # "zscore", "minmax", or "robust"
    anomaly_threshold_sigma: float = 3.0
    
    # Synthetic data generation
    num_synthetic_samples: int = 50000
    anomaly_rate: float = 0.15
    noise_level: float = 0.01
    
    # Spectrogram settings
    fft_size: int = 256
    hop_length: int = 64
    window: str = "hann"

@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    
    # Metrics
    target_accuracy: float = 0.95
    target_precision: float = 0.95
    target_recall: float = 0.95
    target_f1: float = 0.95
    
    # Performance targets
    max_inference_time_ms: float = 1.0
    max_power_consumption_mw: float = 100.0
    
    # Visualization
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    plot_pr_curve: bool = True
    save_predictions: bool = True

# Global configuration instance
CONFIG = {
    "model": ModelConfig(),
    "training": TrainingConfig(),
    "data": DataConfig(),
    "evaluation": EvaluationConfig()
}

def get_config(config_name: str = None):
    """Get configuration object."""
    if config_name is None:
        return CONFIG
    return CONFIG.get(config_name, None)

def update_config(config_name: str, **kwargs):
    """Update configuration parameters."""
    if config_name in CONFIG:
        for key, value in kwargs.items():
            if hasattr(CONFIG[config_name], key):
                setattr(CONFIG[config_name], key, value)
            else:
                raise ValueError(f"Invalid config parameter: {key}")
    else:
        raise ValueError(f"Invalid config name: {config_name}")