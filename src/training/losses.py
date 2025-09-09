# src/training/losses.py
"""Loss functions for hybrid neuromorphic model training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class CombinedLoss(nn.Module):
    """Combined loss function for anomaly detection and reconstruction."""
    
    def __init__(self, anomaly_weight: float = 1.0, reconstruction_weight: float = 0.5):
        super().__init__()
        self.anomaly_weight = anomaly_weight
        self.reconstruction_weight = reconstruction_weight
        self.focal_loss = FocalLoss()
        self.reconstruction_loss = ReconstructionLoss()
    
    def forward(self, outputs: Dict, sequences: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Anomaly detection loss
        anomaly_loss = self.focal_loss(outputs['classification_logits'], labels)
        
        # Reconstruction loss
        recon_loss = 0.0
        if outputs['reconstructed_sequences'] is not None:
            recon_loss = self.reconstruction_loss(outputs['reconstructed_sequences'], sequences)
        
        # Combined loss
        total_loss = (self.anomaly_weight * anomaly_loss + 
                     self.reconstruction_weight * recon_loss)
        
        return {
            'total_loss': total_loss,
            'anomaly_loss': anomaly_loss,
            'reconstruction_loss': recon_loss
        }

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class ReconstructionLoss(nn.Module):
    """Reconstruction loss for signal correction."""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return self.mse_loss(reconstructed, original)