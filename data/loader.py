# src/data/loader.py
"""Data loading and management for plasma turbulence datasets."""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
import h5py
import os
from typing import Tuple, Optional, Dict, List
import json
from pathlib import Path

from .datasets import PlasmaDataset
from .preprocessor import PlasmaPreprocessor
from ..config import get_config

class PlasmaDataLoader:
    """
    Data loader for plasma turbulence datasets with support for multiple data sources.
    
    Handles NASA PIC simulation data, ESA Swarm mission data, and synthetic data
    with automatic preprocessing and batching.
    """
    
    def __init__(self, 
                 data_dir: str = "data/processed/",
                 batch_size: int = 64,
                 num_workers: int = 4,
                 pin_memory: bool = True):
        """
        Initialize plasma data loader.
        
        Args:
            data_dir: Directory containing processed data
            batch_size: Batch size for training
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for GPU transfer
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Get configuration
        self.data_config = get_config("data")
        self.model_config = get_config("model")
        
        # Initialize preprocessor
        self.preprocessor = PlasmaPreprocessor()
        
        # Data storage
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def load_nasa_pic_data(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Load NASA Particle-in-Cell simulation data.
        
        Args:
            file_path: Path to NASA PIC data file
            
        Returns:
            Dictionary containing plasma parameters
        """
        print(f"Loading NASA PIC data from {file_path}")
        
        try:
            with h5py.File(file_path, 'r') as f:
                data = {}
                
                # Extract plasma parameters
                if 'density' in f:
                    data['density'] = f['density'][:]
                if 'velocity' in f:
                    velocity = f['velocity'][:]
                    if velocity.ndim > 1:
                        data['velocity_x'] = velocity[:, 0] if velocity.shape[1] > 0 else velocity[:, 0]
                        data['velocity_y'] = velocity[:, 1] if velocity.shape[1] > 1 else np.zeros_like(velocity[:, 0])
                        data['velocity_z'] = velocity[:, 2] if velocity.shape[1] > 2 else np.zeros_like(velocity[:, 0])
                    else:
                        data['velocity_x'] = velocity
                        data['velocity_y'] = np.zeros_like(velocity)
                        data['velocity_z'] = np.zeros_like(velocity)
                
                if 'magnetic_field' in f:
                    b_field = f['magnetic_field'][:]
                    if b_field.ndim > 1:
                        data['Bx'] = b_field[:, 0] if b_field.shape[1] > 0 else b_field[:, 0]
                        data['By'] = b_field[:, 1] if b_field.shape[1] > 1 else np.zeros_like(b_field[:, 0])
                        data['Bz'] = b_field[:, 2] if b_field.shape[1] > 2 else np.zeros_like(b_field[:, 0])
                    else:
                        data['Bx'] = b_field
                        data['By'] = np.zeros_like(b_field)
                        data['Bz'] = np.zeros_like(b_field)
                
                if 'energy' in f:
                    data['energy'] = f['energy'][:]
                
                print(f"Loaded NASA PIC data with {len(data)} parameters")
                return data
                
        except Exception as e:
            print(f"Error loading NASA PIC data: {e}")
            return {}
    
    def load_esa_swarm_data(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Load ESA Swarm mission ionospheric plasma data.
        
        Args:
            file_path: Path to ESA Swarm data file
            
        Returns:
            Dictionary containing plasma parameters
        """
        print(f"Loading ESA Swarm data from {file_path}")
        
        try:
            # ESA Swarm data is typically in CSV or HDF5 format
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                data = {}
                
                # Map Swarm parameters to standard format
                if 'Ne' in df.columns:  # Electron density
                    data['density'] = df['Ne'].values
                if 'Vi_x' in df.columns:  # Ion velocity components
                    data['velocity_x'] = df['Vi_x'].values
                if 'Vi_y' in df.columns:
                    data['velocity_y'] = df['Vi_y'].values
                if 'Vi_z' in df.columns:
                    data['velocity_z'] = df['Vi_z'].values
                
                # Magnetic field components
                if 'B_x' in df.columns:
                    data['Bx'] = df['B_x'].values
                if 'B_y' in df.columns:
                    data['By'] = df['B_y'].values
                if 'B_z' in df.columns:
                    data['Bz'] = df['B_z'].values
                
                # Calculate energy if not present
                if 'energy' not in data and 'density' in data:
                    # Simple energy calculation from available parameters
                    kinetic_energy = 0.5 * data['density'] * (
                        data.get('velocity_x', np.zeros_like(data['density']))**2 +
                        data.get('velocity_y', np.zeros_like(data['density']))**2 +
                        data.get('velocity_z', np.zeros_like(data['density']))**2
                    )
                    data['energy'] = kinetic_energy
                
                print(f"Loaded ESA Swarm data with {len(data)} parameters")
                return data
                
        except Exception as e:
            print(f"Error loading ESA Swarm data: {e}")
            return {}
    
    def load_synthetic_data(self, data_dir: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load synthetic plasma data generated by SyntheticPlasmaGenerator.
        
        Args:
            data_dir: Directory containing synthetic data files
            
        Returns:
            Tuple of (sequences, spectrograms, labels)
        """
        print(f"Loading synthetic data from {data_dir}")
        
        try:
            sequences_path = Path(data_dir) / "sequences.pt"
            spectrograms_path = Path(data_dir) / "spectrograms.pt"
            labels_path = Path(data_dir) / "labels.pt"
            
            sequences = torch.load(sequences_path)
            spectrograms = torch.load(spectrograms_path)
            labels = torch.load(labels_path)
            
            print(f"Loaded synthetic data:")
            print(f"- Sequences: {sequences.shape}")
            print(f"- Spectrograms: {spectrograms.shape}")
            print(f"- Labels: {labels.shape}")
            
            return sequences, spectrograms, labels
            
        except Exception as e:
            print(f"Error loading synthetic data: {e}")
            # Return empty tensors if loading fails
            return torch.empty(0), torch.empty(0), torch.empty(0)
    
    def prepare_datasets(self, use_synthetic: bool = True, 
                        real_data_paths: Optional[List[str]] = None) -> None:
        """
        Prepare training, validation, and test datasets.
        
        Args:
            use_synthetic: Whether to include synthetic data
            real_data_paths: List of paths to real data files
        """
        all_sequences = []
        all_spectrograms = []
        all_labels = []
        
        # Load synthetic data if requested
        if use_synthetic:
            synthetic_dir = self.data_config.synthetic_data_dir
            if os.path.exists(synthetic_dir):
                syn_seq, syn_spec, syn_labels = self.load_synthetic_data(synthetic_dir)
                if syn_seq.numel() > 0:
                    all_sequences.append(syn_seq)
                    all_spectrograms.append(syn_spec)
                    all_labels.append(syn_labels)
        
        # Load real data if paths provided
        if real_data_paths:
            for data_path in real_data_paths:
                if not os.path.exists(data_path):
                    print(f"Warning: Data path {data_path} does not exist")
                    continue
                
                # Determine data source and load accordingly
                if "nasa" in data_path.lower() or "pic" in data_path.lower():
                    plasma_data = self.load_nasa_pic_data(data_path)
                elif "esa" in data_path.lower() or "swarm" in data_path.lower():
                    plasma_data = self.load_esa_swarm_data(data_path)
                else:
                    print(f"Unknown data source format for {data_path}")
                    continue
                
                if plasma_data:
                    # Convert to tensor format and preprocess
                    sequences, spectrograms, labels = self.preprocessor.process_real_data(plasma_data)
                    all_sequences.append(sequences)
                    all_spectrograms.append(spectrograms)
                    all_labels.append(labels)
        
        if not all_sequences:
            raise ValueError("No data loaded. Please check data paths and synthetic data generation.")
        
        # Concatenate all data
        combined_sequences = torch.cat(all_sequences, dim=0)
        combined_spectrograms = torch.cat(all_spectrograms, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)
        
        print(f"Combined dataset shapes:")
        print(f"- Sequences: {combined_sequences.shape}")
        print(f"- Spectrograms: {combined_spectrograms.shape}")
        print(f"- Labels: {combined_labels.shape}")
        
        # Create dataset
        full_dataset = PlasmaDataset(
            sequences=combined_sequences,
            spectrograms=combined_spectrograms,
            labels=combined_labels,
            transform=self.preprocessor.get_augmentation_transform()
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(self.data_config.train_split * total_size)
        val_size = int(self.data_config.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Dataset splits:")
        print(f"- Train: {len(self.train_dataset)} samples")
        print(f"- Validation: {len(self.val_dataset)} samples")
        print(f"- Test: {len(self.test_dataset)} samples")
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get PyTorch data loaders for training, validation, and testing.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.train_dataset is None:
            raise ValueError("Datasets not prepared. Call prepare_datasets() first.")
        
        # Training data loader with shuffling and augmentation
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
        # Validation data loader
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
        
        # Test data loader
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_data_statistics(self) -> Dict[str, float]:
        """
        Calculate dataset statistics for monitoring and analysis.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.train_dataset is None:
            return {}
        
        # Sample some data to calculate statistics
        sample_loader = DataLoader(self.train_dataset, batch_size=100, shuffle=True)
        sample_batch = next(iter(sample_loader))
        
        sequences, spectrograms, labels = sample_batch
        
        stats = {
            'sequence_mean': sequences.mean().item(),
            'sequence_std': sequences.std().item(),
            'sequence_min': sequences.min().item(),
            'sequence_max': sequences.max().item(),
            'spectrogram_mean': spectrograms.mean().item(),
            'spectrogram_std': spectrograms.std().item(),
            'anomaly_rate': labels.mean().item(),
            'total_samples': len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)
        }
        
        return stats