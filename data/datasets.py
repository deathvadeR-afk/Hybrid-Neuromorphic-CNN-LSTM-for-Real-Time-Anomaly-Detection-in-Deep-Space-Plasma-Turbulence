# src/data/datasets.py
"""PyTorch Dataset classes for plasma turbulence data."""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional, Callable

class PlasmaDataset(Dataset):
    """
    PyTorch Dataset for plasma turbulence sequences and spectrograms.
    
    Handles time-series plasma data with corresponding spectrograms and anomaly labels.
    """
    
    def __init__(self,
                 sequences: torch.Tensor,
                 spectrograms: torch.Tensor,
                 labels: torch.Tensor,
                 transform: Optional[Callable] = None):
        """
        Initialize plasma dataset.
        
        Args:
            sequences: Time-series sequences [N, seq_len, features]
            spectrograms: Spectrogram data [N, channels, freq_bins, time_frames]
            labels: Anomaly labels [N, seq_len] or [N]
            transform: Optional data augmentation transform
        """
        self.sequences = sequences
        self.spectrograms = spectrograms
        self.labels = labels
        self.transform = transform
        
        # Validate shapes
        assert len(sequences) == len(spectrograms) == len(labels), \
            "Sequences, spectrograms, and labels must have same batch dimension"
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (sequence, spectrogram, label)
        """
        sequence = self.sequences[idx]
        spectrogram = self.spectrograms[idx]
        label = self.labels[idx]
        
        # Apply transforms if provided
        if self.transform is not None:
            sequence, spectrogram = self.transform(sequence, spectrogram)
        
        return sequence, spectrogram, label
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Calculate sample weights for imbalanced dataset handling.
        
        Returns:
            Sample weights for balanced training
        """
        # For sequence-level labels, take max over time dimension
        if self.labels.dim() > 1:
            sample_labels = self.labels.max(dim=1)[0]
        else:
            sample_labels = self.labels
        
        # Calculate class frequencies
        positive_samples = (sample_labels > 0.5).sum().float()
        negative_samples = (sample_labels <= 0.5).sum().float()
        total_samples = len(sample_labels)
        
        # Calculate weights (inverse frequency)
        pos_weight = total_samples / (2 * positive_samples) if positive_samples > 0 else 1.0
        neg_weight = total_samples / (2 * negative_samples) if negative_samples > 0 else 1.0
        
        # Assign weights to samples
        weights = torch.zeros_like(sample_labels)
        weights[sample_labels > 0.5] = pos_weight
        weights[sample_labels <= 0.5] = neg_weight
        
        return weights