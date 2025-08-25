# src/models/cnn_module.py
"""CNN module for spatial feature extraction from plasma spectrograms."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class CNNFeatureExtractor(nn.Module):
    """
    Convolutional Neural Network for extracting spatial features from plasma spectrograms.
    
    Processes frequency-domain representations of plasma turbulence data to identify
    spatial patterns indicative of anomalous behavior.
    """
    
    def __init__(self, 
                 input_channels: int = 8,
                 cnn_channels: list = [32, 64, 128],
                 kernel_sizes: list = [3, 3, 3],
                 dropout_rate: float = 0.2):
        """
        Initialize CNN feature extractor.
        
        Args:
            input_channels: Number of input channels (plasma parameters)
            cnn_channels: List of output channels for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            dropout_rate: Dropout probability for regularization
        """
        super(CNNFeatureExtractor, self).__init__()
        
        self.input_channels = input_channels
        self.cnn_channels = cnn_channels
        self.dropout_rate = dropout_rate
        
        # Build convolutional layers
        layers = []
        in_channels = input_channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_channels, kernel_sizes)):
            # Convolutional layer
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2,
                bias=False
            ))
            
            # Batch normalization
            layers.append(nn.BatchNorm2d(out_channels))
            
            # Activation
            layers.append(nn.ReLU(inplace=True))
            
            # Max pooling (reduce spatial dimensions)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            # Dropout for regularization
            if i < len(cnn_channels) - 1:  # No dropout after last layer
                layers.append(nn.Dropout2d(p=dropout_rate))
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Calculate flattened feature size
        self.feature_size = cnn_channels[-1] * 4 * 4
        
        # Feature compression layer
        self.feature_compress = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.
        
        Args:
            x: Input spectrograms [batch_size, channels, freq_bins, time_frames]
            
        Returns:
            Extracted features [batch_size, 256]
        """
        # Apply convolutional layers
        features = self.conv_layers(x)
        
        # Adaptive pooling to consistent size
        features = self.adaptive_pool(features)
        
        # Flatten spatial dimensions
        batch_size = features.size(0)
        features = features.view(batch_size, -1)
        
        # Compress features
        compressed_features = self.feature_compress(features)
        
        return compressed_features
    
    def get_feature_maps(self, x: torch.Tensor) -> list:
        """
        Extract feature maps from each convolutional layer for visualization.
        
        Args:
            x: Input spectrograms
            
        Returns:
            List of feature maps from each layer
        """
        feature_maps = []
        current_input = x
        
        for i, layer in enumerate(self.conv_layers):
            current_input = layer(current_input)
            if isinstance(layer, nn.Conv2d):
                feature_maps.append(current_input.detach())
        
        return feature_maps