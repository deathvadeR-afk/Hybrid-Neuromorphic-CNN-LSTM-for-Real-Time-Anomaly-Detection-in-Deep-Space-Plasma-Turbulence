# src/data/preprocessor.py
"""Data preprocessing utilities for plasma data."""

import torch
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, Tuple, Optional

class PlasmaPreprocessor:
    """Preprocessing pipeline for plasma turbulence data."""
    
    def __init__(self, normalize_method: str = "zscore"):
        self.normalize_method = normalize_method
        self.scaler = None
        
    def process_real_data(self, plasma_data: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process real plasma data into model format."""
        # Convert to sequence format
        sequence_length = min(len(v) for v in plasma_data.values())
        
        # Create sequences
        sequences = []
        for key in ['density', 'velocity_x', 'velocity_y', 'velocity_z', 'Bx', 'By', 'Bz', 'energy']:
            if key in plasma_data:
                sequences.append(plasma_data[key][:sequence_length])
            else:
                sequences.append(np.zeros(sequence_length))
        
        sequence_array = np.column_stack(sequences)
        
        # Generate spectrograms
        spectrograms = self._generate_spectrograms(sequence_array)
        
        # Create labels (simple anomaly detection based on statistical outliers)
        labels = self._detect_anomalies(sequence_array)
        
        return torch.FloatTensor(sequence_array), torch.FloatTensor(spectrograms), torch.FloatTensor(labels)
    
    def _generate_spectrograms(self, sequences: np.ndarray) -> np.ndarray:
        """Generate spectrograms from time series."""
        spectrograms = []
        for i in range(sequences.shape[1]):
            f, t, Sxx = signal.spectrogram(sequences[:, i], nperseg=64)
            spectrograms.append(np.log10(Sxx + 1e-10))
        
        return np.array(spectrograms)
    
    def _detect_anomalies(self, sequences: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Simple anomaly detection using statistical thresholds."""
        # Calculate z-scores
        z_scores = np.abs((sequences - np.mean(sequences, axis=0)) / np.std(sequences, axis=0))
        anomalies = (z_scores > threshold).any(axis=1).astype(float)
        return anomalies
    
    def get_augmentation_transform(self):
        """Get data augmentation transform."""
        def transform(sequence, spectrogram):
            # Add small amount of noise for augmentation
            if np.random.random() > 0.5:
                noise = torch.randn_like(sequence) * 0.01
                sequence = sequence + noise
            return sequence, spectrogram
        
        return transform