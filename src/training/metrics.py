# src/training/metrics.py
"""Evaluation metrics for anomaly detection."""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict

class AnomalyMetrics:
    """Comprehensive metrics for anomaly detection evaluation."""
    
    def calculate_batch_metrics(self, probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
        """Calculate metrics for a single batch."""
        predictions = (probs[:, 1] > threshold).float()
        
        # Convert to numpy for sklearn
        y_true = labels.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
    
    def calculate_comprehensive_metrics(self, probs: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor) -> Dict[str, float]:
        """Calculate comprehensive metrics including AUC-ROC."""
        y_true = labels.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        y_scores = probs[:, 1].cpu().numpy()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
        }
        
        return metrics