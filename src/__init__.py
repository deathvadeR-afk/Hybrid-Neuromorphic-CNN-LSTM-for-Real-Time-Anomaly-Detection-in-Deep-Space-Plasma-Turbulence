# src/__init__.py
"""
Hybrid Neuromorphic-CNN-LSTM for Real-Time Anomaly Detection in Deep-Space Plasma Turbulence

This package provides a complete framework for detecting and correcting anomalies
in plasma turbulence data using a hybrid neural architecture optimized for 
space-based edge computing applications.
"""

__version__ = "0.1.0"
__author__ = "NASA Data Science Team"

from .models import HybridNeuromorphicModel
from .data import PlasmaDataLoader, SyntheticPlasmaGenerator
from .training import Trainer
from .evaluation import AnomalyEvaluator

__all__ = [
    "HybridNeuromorphicModel",
    "PlasmaDataLoader", 
    "SyntheticPlasmaGenerator",
    "Trainer",
    "AnomalyEvaluator"
]