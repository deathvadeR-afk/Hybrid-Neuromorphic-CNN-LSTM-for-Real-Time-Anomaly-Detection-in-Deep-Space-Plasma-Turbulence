# src/models/snn_module.py
"""Spiking Neural Network module for neuromorphic processing."""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen, spikeplot, surrogate
import torch.nn.functional as F
from typing import Tuple, List

class SpikingNeuralNetwork(nn.Module):
    """
    Spiking Neural Network for energy-efficient anomaly classification.
    
    Uses Leaky Integrate-and-Fire (LIF) neurons to process fused features
    from CNN and LSTM modules with minimal power consumption.
    """
    
    def __init__(self,
                 input_size: int = 512,  # CNN (256) + LSTM (256) features
                 hidden_sizes: List[int] = [256, 128],
                 num_outputs: int = 2,  # Normal vs Anomaly
                 beta: float = 0.95,
                 threshold: float = 1.0,
                 spike_grad: str = "fast_sigmoid",
                 num_steps: int = 100):
        """
        Initialize Spiking Neural Network.
        
        Args:
            input_size: Input feature dimension
            hidden_sizes: List of hidden layer sizes
            num_outputs: Number of output classes
            beta: Decay rate for LIF neurons (0-1)
            threshold: Spike threshold
            spike_grad: Gradient surrogate function type
            num_steps: Number of simulation time steps
        """
        super(SpikingNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_outputs = num_outputs
        self.num_steps = num_steps
        self.beta = beta
        self.threshold = threshold
        
        # Set gradient surrogate function
        if spike_grad == "fast_sigmoid":
            spike_grad_fn = surrogate.fast_sigmoid()
        elif spike_grad == "straight_through":
            spike_grad_fn = surrogate.straight_through_estimator()
        else:
            spike_grad_fn = surrogate.atan()
        
        # Build spiking layers
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [num_outputs]
        
        for i in range(len(layer_sizes) - 1):
            # Linear transformation
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # LIF neuron layer (except for output)
            if i < len(layer_sizes) - 2:
                layers.append(snn.Leaky(
                    beta=beta,
                    threshold=threshold,
                    spike_grad=spike_grad_fn,
                    init_hidden=True
                ))
            else:
                # Output layer - use membrane potential directly
                layers.append(snn.Leaky(
                    beta=beta,
                    threshold=threshold,
                    spike_grad=spike_grad_fn,
                    init_hidden=True,
                    output=True  # Return membrane potential
                ))
        
        self.snn_layers = nn.ModuleList(layers)
        
        # Rate encoding for input conversion
        self.rate_encoder = spikegen.rate
        
        # Spike counter for output
        self.spike_counter = torch.zeros(1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through SNN.
        
        Args:
            x: Input features [batch_size, input_size]
            
        Returns:
            Tuple of (output_spikes, membrane_potentials, spike_counts)
        """
        batch_size = x.shape[0]
        
        # Convert input to spike trains using rate encoding
        # Normalize input to [0, 1] range for rate encoding
        x_normalized = torch.sigmoid(x)
        
        # Initialize tracking variables
        membrane_potentials = []
        spike_counts = torch.zeros(batch_size, self.num_outputs, device=x.device)
        output_spikes = []
        
        # Reset hidden states
        for layer in self.snn_layers:
            if hasattr(layer, 'init_leaky'):
                layer.init_leaky()
        
        # Simulate for num_steps time steps
        for step in range(self.num_steps):
            # Generate spike input for this time step
            spike_input = self.rate_encoder(x_normalized, num_steps=1).squeeze(0)
            
            # Forward pass through spiking layers
            current_input = spike_input
            
            for i, layer in enumerate(self.snn_layers):
                if isinstance(layer, nn.Linear):
                    current_input = layer(current_input)
                elif isinstance(layer, snn.Leaky):
                    if i == len(self.snn_layers) - 1:  # Output layer
                        spk, mem = layer(current_input)
                        membrane_potentials.append(mem)
                        spike_counts += spk
                        output_spikes.append(spk)
                        current_input = spk
                    else:  # Hidden layers
                        spk, mem = layer(current_input)
                        current_input = spk
        
        # Average membrane potential over time steps
        avg_membrane_potential = torch.stack(membrane_potentials).mean(dim=0)
        
        # Convert spike counts to probabilities
        spike_probs = spike_counts / self.num_steps
        
        return output_spikes, avg_membrane_potential, spike_probs
    
    def get_spike_activity(self, x: torch.Tensor) -> dict:
        """
        Get detailed spike activity analysis.
        
        Args:
            x: Input features
            
        Returns:
            Dictionary with spike activity metrics
        """
        batch_size = x.shape[0]
        x_normalized = torch.sigmoid(x)
        
        layer_activities = {}
        
        # Reset hidden states
        for layer in self.snn_layers:
            if hasattr(layer, 'init_leaky'):
                layer.init_leaky()
        
        for step in range(self.num_steps):
            spike_input = self.rate_encoder(x_normalized, num_steps=1).squeeze(0)
            current_input = spike_input
            
            for i, layer in enumerate(self.snn_layers):
                layer_name = f"layer_{i}"
                
                if isinstance(layer, nn.Linear):
                    current_input = layer(current_input)
                elif isinstance(layer, snn.Leaky):
                    spk, mem = layer(current_input)
                    
                    if layer_name not in layer_activities:
                        layer_activities[layer_name] = {
                            'spikes': [],
                            'membrane_potentials': [],
                            'spike_rates': []
                        }
                    
                    layer_activities[layer_name]['spikes'].append(spk.detach())
                    layer_activities[layer_name]['membrane_potentials'].append(mem.detach())
                    
                    current_input = spk
        
        # Calculate spike rates for each layer
        for layer_name in layer_activities:
            spikes = torch.stack(layer_activities[layer_name]['spikes'])
            spike_rate = spikes.sum(dim=0) / self.num_steps
            layer_activities[layer_name]['spike_rates'] = spike_rate
        
        return layer_activities
    
    def estimate_power_consumption(self, x: torch.Tensor) -> float:
        """
        Estimate power consumption based on spike activity.
        
        Args:
            x: Input features
            
        Returns:
            Estimated power consumption in arbitrary units
        """
        spike_activities = self.get_spike_activity(x)
        
        total_spikes = 0
        for layer_name in spike_activities:
            layer_spikes = torch.stack(spike_activities[layer_name]['spikes']).sum()
            total_spikes += layer_spikes.item()
        
        # Neuromorphic power is proportional to spike count
        # Assume 1 pJ per spike (typical for neuromorphic chips)
        power_pj = total_spikes * 1.0
        
        return power_pj