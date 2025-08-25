# src/data/synthetic.py
"""Synthetic plasma turbulence data generation using PlasmaPy."""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
import plasmapy as plp
from plasmapy.simulation import ParticleTracker
from astropy import units as u
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class SyntheticPlasmaGenerator:
    """Generate synthetic plasma turbulence data with controlled anomalies."""
    
    def __init__(self, 
                 sequence_length: int = 1000,
                 num_features: int = 8,
                 sampling_rate: float = 1000.0,  # Hz
                 anomaly_rate: float = 0.15):
        """
        Initialize synthetic plasma data generator.
        
        Args:
            sequence_length: Number of time steps per sequence
            num_features: Number of plasma parameters to generate
            sampling_rate: Data sampling frequency in Hz
            anomaly_rate: Fraction of sequences that contain anomalies
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.sampling_rate = sampling_rate
        self.anomaly_rate = anomaly_rate
        
        # Time array
        self.dt = 1.0 / sampling_rate
        self.time_array = np.linspace(0, sequence_length * self.dt, sequence_length)
        
        # Physical constants for plasma simulation
        self.proton_mass = 1.67e-27  # kg
        self.electron_mass = 9.11e-31  # kg
        self.elementary_charge = 1.6e-19  # C
        self.permittivity = 8.85e-12  # F/m
        
    def generate_alfven_waves(self, 
                             amplitude: float = 1.0,
                             frequency: float = 10.0,
                             wave_number: float = 0.1,
                             phase_shift: float = 0.0) -> np.ndarray:
        """
        Generate Alfvén wave patterns.
        
        Args:
            amplitude: Wave amplitude
            frequency: Wave frequency in Hz
            wave_number: Spatial wave number
            phase_shift: Phase shift in radians
            
        Returns:
            Generated wave data [time_steps, 3] for Bx, By, Bz components
        """
        omega = 2 * np.pi * frequency
        k = wave_number
        
        # Alfvén wave equations
        t = self.time_array
        
        # Magnetic field perturbations (transverse to propagation)
        Bx = amplitude * np.sin(omega * t + phase_shift)
        By = amplitude * np.cos(omega * t + phase_shift)
        Bz = 0.1 * amplitude * np.sin(2 * omega * t + phase_shift)  # Small parallel component
        
        return np.column_stack([Bx, By, Bz])
    
    def generate_mhd_turbulence(self, 
                               energy_spectrum_slope: float = -5/3,
                               magnetic_field_strength: float = 1e-5) -> Dict[str, np.ndarray]:
        """
        Generate MHD turbulence using Kolmogorov-like energy spectrum.
        
        Args:
            energy_spectrum_slope: Power law slope for energy spectrum
            magnetic_field_strength: Background magnetic field in Tesla
            
        Returns:
            Dictionary containing plasma parameters
        """
        # Frequency array for spectral generation
        freqs = np.fft.fftfreq(self.sequence_length, self.dt)
        freqs = freqs[freqs > 0]  # Positive frequencies only
        
        # Kolmogorov-like energy spectrum
        energy_spectrum = np.power(freqs, energy_spectrum_slope)
        energy_spectrum[0] = energy_spectrum[1]  # Avoid division by zero
        
        # Generate random phases
        phases = np.random.uniform(0, 2*np.pi, len(freqs))
        
        # Create complex amplitudes
        amplitudes = np.sqrt(energy_spectrum) * np.exp(1j * phases)
        
        # Generate turbulent fields in Fourier space
        def generate_field_component():
            # Create full spectrum (positive and negative frequencies)
            full_spectrum = np.zeros(self.sequence_length, dtype=complex)
            full_spectrum[1:len(freqs)+1] = amplitudes
            full_spectrum[-len(freqs):] = np.conj(amplitudes[::-1])
            
            # Transform to time domain
            field = np.fft.ifft(full_spectrum).real
            return field
        
        # Generate plasma parameters
        plasma_data = {}
        
        # Magnetic field components (Tesla)
        plasma_data['Bx'] = magnetic_field_strength * (1 + 0.1 * generate_field_component())
        plasma_data['By'] = magnetic_field_strength * (1 + 0.1 * generate_field_component()) 
        plasma_data['Bz'] = magnetic_field_strength * (1 + 0.1 * generate_field_component())
        
        # Plasma density (particles/m³)
        base_density = 1e6
        plasma_data['density'] = base_density * (1 + 0.2 * generate_field_component())
        
        # Velocity components (m/s)
        base_velocity = 1e5
        plasma_data['velocity_x'] = base_velocity * 0.1 * generate_field_component()
        plasma_data['velocity_y'] = base_velocity * 0.1 * generate_field_component()
        plasma_data['velocity_z'] = base_velocity * 0.1 * generate_field_component()
        
        # Energy density (J/m³)
        magnetic_energy = (plasma_data['Bx']**2 + plasma_data['By']**2 + plasma_data['Bz']**2) / (2 * 4e-7 * np.pi)
        kinetic_energy = 0.5 * self.proton_mass * plasma_data['density'] * (
            plasma_data['velocity_x']**2 + plasma_data['velocity_y']**2 + plasma_data['velocity_z']**2
        )
        plasma_data['energy'] = magnetic_energy + kinetic_energy
        
        return plasma_data
    
    def inject_anomalies(self, 
                        plasma_data: Dict[str, np.ndarray],
                        anomaly_type: str = "spike",
                        anomaly_intensity: float = 5.0) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Inject controlled anomalies into plasma data.
        
        Args:
            plasma_data: Dictionary of plasma parameters
            anomaly_type: Type of anomaly ("spike", "dropout", "oscillation")
            anomaly_intensity: Intensity multiplier for anomalies
            
        Returns:
            Tuple of (modified_data, anomaly_labels)
        """
        modified_data = {key: val.copy() for key, val in plasma_data.items()}
        anomaly_labels = np.zeros(self.sequence_length, dtype=np.float32)
        
        # Determine anomaly locations
        num_anomalies = int(self.sequence_length * 0.05)  # 5% of points
        anomaly_indices = np.random.choice(
            self.sequence_length, 
            size=num_anomalies, 
            replace=False
        )
        
        for idx in anomaly_indices:
            anomaly_labels[idx] = 1.0
            
            if anomaly_type == "spike":
                # Sudden spike in all parameters
                for key in modified_data:
                    baseline = np.mean(modified_data[key])
                    modified_data[key][idx] = baseline * (1 + anomaly_intensity)
                    
            elif anomaly_type == "dropout":
                # Signal dropout (near zero)
                for key in modified_data:
                    modified_data[key][idx] *= 0.1
                    
            elif anomaly_type == "oscillation":
                # High-frequency oscillation
                osc_length = min(50, self.sequence_length - idx)
                for i in range(osc_length):
                    if idx + i < self.sequence_length:
                        anomaly_labels[idx + i] = 1.0
                        for key in modified_data:
                            baseline = modified_data[key][idx + i]
                            oscillation = baseline * 0.5 * np.sin(20 * np.pi * i / osc_length)
                            modified_data[key][idx + i] += oscillation
        
        return modified_data, anomaly_labels
    
    def generate_batch(self, 
                      batch_size: int,
                      include_anomalies: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of synthetic plasma sequences.
        
        Args:
            batch_size: Number of sequences to generate
            include_anomalies: Whether to include anomalous sequences
            
        Returns:
            Tuple of (sequences, labels) as PyTorch tensors
        """
        sequences = []
        labels = []
        
        for i in range(batch_size):
            # Generate base turbulence
            plasma_data = self.generate_mhd_turbulence(
                energy_spectrum_slope=np.random.uniform(-2.0, -1.5),
                magnetic_field_strength=np.random.uniform(1e-6, 1e-4)
            )
            
            # Convert to array format
            sequence = np.column_stack([
                plasma_data['density'],
                plasma_data['velocity_x'],
                plasma_data['velocity_y'], 
                plasma_data['velocity_z'],
                plasma_data['Bx'],
                plasma_data['By'],
                plasma_data['Bz'],
                plasma_data['energy']
            ])
            
            # Add Alfvén waves
            alfven = self.generate_alfven_waves(
                amplitude=np.random.uniform(0.1, 0.5),
                frequency=np.random.uniform(5.0, 50.0)
            )
            sequence[:, 4:7] += alfven  # Add to magnetic field components
            
            # Inject anomalies if requested
            if include_anomalies and np.random.random() < self.anomaly_rate:
                plasma_dict = {
                    'density': sequence[:, 0],
                    'velocity_x': sequence[:, 1],
                    'velocity_y': sequence[:, 2],
                    'velocity_z': sequence[:, 3],
                    'Bx': sequence[:, 4],
                    'By': sequence[:, 5], 
                    'Bz': sequence[:, 6],
                    'energy': sequence[:, 7]
                }
                
                anomaly_type = np.random.choice(["spike", "dropout", "oscillation"])
                modified_data, anomaly_labels = self.inject_anomalies(
                    plasma_dict, 
                    anomaly_type=anomaly_type,
                    anomaly_intensity=np.random.uniform(3.0, 8.0)
                )
                
                # Reconstruct sequence
                sequence = np.column_stack([
                    modified_data['density'],
                    modified_data['velocity_x'],
                    modified_data['velocity_y'],
                    modified_data['velocity_z'], 
                    modified_data['Bx'],
                    modified_data['By'],
                    modified_data['Bz'],
                    modified_data['energy']
                ])
                
                labels.append(anomaly_labels)
            else:
                labels.append(np.zeros(self.sequence_length, dtype=np.float32))
            
            # Add noise
            noise_level = np.random.uniform(0.01, 0.05)
            sequence += np.random.normal(0, noise_level, sequence.shape)
            
            sequences.append(sequence)
        
        # Convert to tensors
        sequences_tensor = torch.FloatTensor(np.array(sequences))
        labels_tensor = torch.FloatTensor(np.array(labels))
        
        return sequences_tensor, labels_tensor
    
    def generate_spectrograms(self, 
                             sequences: torch.Tensor,
                             fft_size: int = 256,
                             hop_length: int = 64) -> torch.Tensor:
        """
        Generate spectrograms from time-series sequences.
        
        Args:
            sequences: Time-series sequences [batch, time, features]
            fft_size: FFT window size
            hop_length: Hop length for STFT
            
        Returns:
            Spectrograms [batch, features, freq_bins, time_frames]
        """
        batch_size, seq_len, num_features = sequences.shape
        
        # Calculate spectrogram dimensions
        freq_bins = fft_size // 2 + 1
        time_frames = (seq_len - fft_size) // hop_length + 1
        
        spectrograms = torch.zeros(batch_size, num_features, freq_bins, time_frames)
        
        for b in range(batch_size):
            for f in range(num_features):
                # Compute STFT
                frequencies, times, Zxx = signal.stft(
                    sequences[b, :, f].numpy(),
                    fs=self.sampling_rate,
                    nperseg=fft_size,
                    noverlap=fft_size - hop_length
                )
                
                # Convert to magnitude and log scale
                magnitude = np.abs(Zxx)
                log_magnitude = np.log10(magnitude + 1e-10)
                
                # Store in tensor
                spectrograms[b, f, :, :] = torch.FloatTensor(log_magnitude)
        
        return spectrograms

    def save_synthetic_dataset(self, 
                              num_samples: int,
                              save_path: str = "data/synthetic/"):
        """
        Generate and save a large synthetic dataset.
        
        Args:
            num_samples: Total number of samples to generate
            save_path: Directory to save the dataset
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        batch_size = 100
        num_batches = num_samples // batch_size
        
        all_sequences = []
        all_labels = []
        all_spectrograms = []
        
        print(f"Generating {num_samples} synthetic plasma sequences...")
        
        for batch_idx in range(num_batches):
            sequences, labels = self.generate_batch(batch_size)
            spectrograms = self.generate_spectrograms(sequences)
            
            all_sequences.append(sequences)
            all_labels.append(labels)
            all_spectrograms.append(spectrograms)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Generated {(batch_idx + 1) * batch_size} samples...")
        
        # Concatenate all batches
        final_sequences = torch.cat(all_sequences, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        final_spectrograms = torch.cat(all_spectrograms, dim=0)
        
        # Save datasets
        torch.save(final_sequences, os.path.join(save_path, "sequences.pt"))
        torch.save(final_labels, os.path.join(save_path, "labels.pt"))
        torch.save(final_spectrograms, os.path.join(save_path, "spectrograms.pt"))
        
        print(f"Saved synthetic dataset to {save_path}")
        print(f"Sequences shape: {final_sequences.shape}")
        print(f"Labels shape: {final_labels.shape}")
        print(f"Spectrograms shape: {final_spectrograms.shape}")
        
        # Save metadata
        metadata = {
            "num_samples": num_samples,
            "sequence_length": self.sequence_length,
            "num_features": self.num_features,
            "sampling_rate": self.sampling_rate,
            "anomaly_rate": self.anomaly_rate
        }
        
        import json
        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)