# src/models/lstm_module.py
"""LSTM module for temporal sequence processing of plasma data."""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class LSTMSequenceProcessor(nn.Module):
    """
    LSTM network for processing temporal sequences in plasma turbulence data.
    
    Captures long-term dependencies and temporal patterns that indicate
    the evolution of plasma instabilities over time.
    """
    
    def __init__(self,
                 input_size: int = 8,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout_rate: float = 0.2,
                 bidirectional: bool = True,
                 output_size: int = 256):
        """
        Initialize LSTM sequence processor.
        
        Args:
            input_size: Number of input features per timestep
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout_rate: Dropout probability between layers
            bidirectional: Whether to use bidirectional LSTM
            output_size: Output feature dimension
        """
        super(LSTMSequenceProcessor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Attention mechanism for focusing on important timesteps
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Output projection layer
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_size, output_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM weights using Xavier initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, 
                x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input sequences [batch_size, seq_length, input_size]
            hidden: Optional initial hidden state
            
        Returns:
            Tuple of (processed_features, attention_weights)
        """
        batch_size, seq_length, _ = x.shape
        
        # LSTM forward pass
        lstm_output, hidden_state = self.lstm(x, hidden)
        
        # Apply attention mechanism
        attention_weights = self.attention(lstm_output)
        
        # Weighted sum using attention
        attended_output = torch.sum(lstm_output * attention_weights, dim=1)
        
        # Project to output dimension
        processed_features = self.output_projection(attended_output)
        
        return processed_features, attention_weights.squeeze(-1)
    
    def get_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract hidden states for all timesteps.
        
        Args:
            x: Input sequences
            
        Returns:
            Hidden states for all timesteps
        """
        lstm_output, _ = self.lstm(x)
        return lstm_output