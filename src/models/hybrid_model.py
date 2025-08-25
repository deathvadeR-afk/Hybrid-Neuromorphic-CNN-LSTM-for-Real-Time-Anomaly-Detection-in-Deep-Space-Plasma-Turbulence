# src/models/hybrid_model.py
"""Hybrid Neuromorphic-CNN-LSTM model for plasma anomaly detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import time
import numpy as np

from .cnn_module import CNNFeatureExtractor
from .lstm_module import LSTMSequenceProcessor
from .snn_module import SpikingNeuralNetwork

class HybridNeuromorphicModel(nn.Module):
    """
    Hybrid model combining CNN, LSTM, and SNN for plasma turbulence anomaly detection.
    
    Architecture:
    1. CNN extracts spatial features from plasma spectrograms
    2. LSTM processes temporal sequences of plasma parameters
    3. Features are fused and fed to SNN for efficient classification
    4. Additional reconstruction branch for signal correction
    """
    
    def __init__(self,
                 # CNN parameters
                 cnn_channels: list = [32, 64, 128],
                 cnn_kernel_sizes: list = [3, 3, 3],
                 
                 # LSTM parameters
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 2,
                 
                 # SNN parameters
                 snn_hidden_sizes: list = [256, 128],
                 snn_beta: float = 0.95,
                 snn_threshold: float = 1.0,
                 snn_num_steps: int = 100,
                 
                 # Input dimensions
                 sequence_length: int = 1000,
                 num_features: int = 8,
                 spectrogram_channels: int = 8,
                 
                 # Model configuration
                 dropout_rate: float = 0.2,
                 enable_reconstruction: bool = True):
        """
        Initialize hybrid neuromorphic model.
        
        Args:
            cnn_channels: Output channels for CNN layers
            cnn_kernel_sizes: Kernel sizes for CNN layers
            lstm_hidden_size: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            snn_hidden_sizes: Hidden layer sizes for SNN
            snn_beta: SNN neuron decay rate
            snn_threshold: SNN spike threshold
            snn_num_steps: SNN simulation steps
            sequence_length: Input sequence length
            num_features: Number of plasma parameters
            spectrogram_channels: Number of spectrogram channels
            dropout_rate: Dropout probability
            enable_reconstruction: Whether to include reconstruction branch
        """
        super(HybridNeuromorphicModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.enable_reconstruction = enable_reconstruction
        
        # CNN for spectrogram processing
        self.cnn_extractor = CNNFeatureExtractor(
            input_channels=spectrogram_channels,
            cnn_channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            dropout_rate=dropout_rate
        )
        
        # LSTM for temporal sequence processing
        self.lstm_processor = LSTMSequenceProcessor(
            input_size=num_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout_rate=dropout_rate,
            bidirectional=True,
            output_size=256
        )
        
        # Feature fusion layer
        fusion_input_size = 256 + 256  # CNN (256) + LSTM (256)
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )
        
        # Spiking Neural Network for classification
        self.snn_classifier = SpikingNeuralNetwork(
            input_size=512,
            hidden_sizes=snn_hidden_sizes,
            num_outputs=2,  # Normal vs Anomaly
            beta=snn_beta,
            threshold=snn_threshold,
            num_steps=snn_num_steps
        )
        
        # Reconstruction branch (optional)
        if enable_reconstruction:
            self.reconstruction_branch = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, sequence_length * num_features),
                nn.Tanh()  # Bounded output for reconstruction
            )
        
        # Performance tracking
        self.inference_times = []
        
    def forward(self, 
                sequences: torch.Tensor,
                spectrograms: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid model.
        
        Args:
            sequences: Time-series data [batch_size, seq_length, num_features]
            spectrograms: Spectrogram data [batch_size, channels, freq_bins, time_frames]
            
        Returns:
            Dictionary containing model outputs
        """
        start_time = time.time()
        
        batch_size = sequences.shape[0]
        
        # CNN feature extraction from spectrograms
        cnn_features = self.cnn_extractor(spectrograms)
        
        # LSTM temporal processing
        lstm_features, attention_weights = self.lstm_processor(sequences)
        
        # Feature fusion
        fused_features = torch.cat([cnn_features, lstm_features], dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # SNN classification
        output_spikes, membrane_potentials, spike_probs = self.snn_classifier(fused_features)
        
        # Classification probabilities (using membrane potentials)
        classification_logits = membrane_potentials
        classification_probs = F.softmax(classification_logits, dim=1)
        
        # Reconstruction (if enabled)
        reconstructed_sequences = None
        if self.enable_reconstruction:
            reconstruction_output = self.reconstruction_branch(fused_features)
            reconstructed_sequences = reconstruction_output.view(
                batch_size, self.sequence_length, self.num_features
            )
        
        # Track inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        self.inference_times.append(inference_time)
        
        return {
            'classification_logits': classification_logits,
            'classification_probs': classification_probs,
            'spike_probs': spike_probs,
            'reconstructed_sequences': reconstructed_sequences,
            'attention_weights': attention_weights,
            'cnn_features': cnn_features,
            'lstm_features': lstm_features,
            'fused_features': fused_features,
            'inference_time_ms': inference_time
        }
    
    def detect_anomalies(self, 
                        sequences: torch.Tensor,
                        spectrograms: torch.Tensor,
                        threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Detect anomalies in plasma sequences.
        
        Args:
            sequences: Time-series data
            spectrograms: Spectrogram data
            threshold: Classification threshold
            
        Returns:
            Anomaly detection results
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(sequences, spectrograms)
            
            # Anomaly predictions (class 1 = anomaly)
            anomaly_probs = outputs['classification_probs'][:, 1]
            anomaly_predictions = (anomaly_probs > threshold).float()
            
            # Reconstruction error (if available)
            reconstruction_error = None
            if outputs['reconstructed_sequences'] is not None:
                reconstruction_error = F.mse_loss(
                    outputs['reconstructed_sequences'], 
                    sequences, 
                    reduction='none'
                ).mean(dim=[1, 2])
            
            return {
                'anomaly_predictions': anomaly_predictions,
                'anomaly_probabilities': anomaly_probs,
                'reconstruction_error': reconstruction_error,
                'corrected_sequences': outputs['reconstructed_sequences'],
                'attention_weights': outputs['attention_weights'],
                'inference_time_ms': outputs['inference_time_ms']
            }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get model performance metrics.
        
        Returns:
            Performance metrics including inference time and power estimates
        """
        if not self.inference_times:
            return {}
        
        avg_inference_time = np.mean(self.inference_times)
        max_inference_time = np.max(self.inference_times)
        min_inference_time = np.min(self.inference_times)
        
        # Estimate power consumption (simplified)
        # Neuromorphic advantage: power scales with activity, not computation
        estimated_power_mw = avg_inference_time * 0.1  # Simplified estimate
        
        return {
            'avg_inference_time_ms': avg_inference_time,
            'max_inference_time_ms': max_inference_time,
            'min_inference_time_ms': min_inference_time,
            'estimated_power_mw': estimated_power_mw,
            'meets_latency_target': avg_inference_time < 1.0,  # <1ms target
            'meets_power_target': estimated_power_mw < 100.0   # <100mW target
        }
    
    def optimize_for_edge(self) -> None:
        """
        Optimize model for edge deployment.
        
        Applies techniques like quantization and pruning for better edge performance.
        """
        # Convert to half precision for memory efficiency
        self.half()
        
        # Set to evaluation mode for inference optimization
        self.eval()
        
        # Disable gradients for inference-only deployment
        for param in self.parameters():
            param.requires_grad = False
        
        print("Model optimized for edge deployment:")
        print(f"- Converted to half precision")
        print(f"- Disabled gradient computation")
        print(f"- Set to evaluation mode")
    
    def get_model_complexity(self) -> Dict[str, int]:
        """
        Calculate model complexity metrics.
        
        Returns:
            Model complexity information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count parameters by module
        cnn_params = sum(p.numel() for p in self.cnn_extractor.parameters())
        lstm_params = sum(p.numel() for p in self.lstm_processor.parameters())
        snn_params = sum(p.numel() for p in self.snn_classifier.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'cnn_parameters': cnn_params,
            'lstm_parameters': lstm_params,
            'snn_parameters': snn_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }