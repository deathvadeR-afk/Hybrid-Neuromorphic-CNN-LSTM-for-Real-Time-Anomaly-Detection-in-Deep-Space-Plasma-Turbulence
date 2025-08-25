"""Main application for neuromorphic plasma anomaly detection."""

import torch
import argparse
from pathlib import Path

from src.models.hybrid_model import HybridNeuromorphicModel
from src.data.loader import PlasmaDataLoader
from src.data.synthetic import SyntheticPlasmaGenerator
from src.training.trainer import Trainer
from src.config import get_config, update_config

def generate_synthetic_data():
    """Generate synthetic training data."""
    print("Generating synthetic plasma data...")
    
    generator = SyntheticPlasmaGenerator()
    generator.save_synthetic_dataset(
        num_samples=50000,
        save_path="data/synthetic/"
    )
    
    print("Synthetic data generation completed!")

def train_model():
    """Train the hybrid neuromorphic model."""
    print("Starting model training...")
    
    # Initialize model
    model = HybridNeuromorphicModel()
    
    # Setup data loader
    data_loader = PlasmaDataLoader("data/processed/")
    data_loader.prepare_datasets(use_synthetic=True)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        experiment_name="neuromorphic_plasma_detection"
    )
    
    # Train model
    history = trainer.train()
    
    print("Training completed!")
    return history

def demo_detection():
    """Run a quick anomaly detection demo."""
    print("Running anomaly detection demo...")
    
    # Load model
    model = HybridNeuromorphicModel()
    
    # Generate sample data
    generator = SyntheticPlasmaGenerator()
    sequences, labels = generator.generate_batch(1, include_anomalies=True)
    spectrograms = generator.generate_spectrograms(sequences)
    
    # Detect anomalies
    results = model.detect_anomalies(sequences, spectrograms)
    
    print(f"Anomaly probability: {results['anomaly_probabilities'][0]:.3f}")
    print(f"Inference time: {results['inference_time_ms']:.2f} ms")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Neuromorphic Plasma Anomaly Detection")
    parser.add_argument("--mode", choices=["generate", "train", "demo", "streamlit"], required=True,
                       help="Mode: generate data, train model, run demo, or start streamlit app")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        generate_synthetic_data()
    elif args.mode == "train":
        train_model()
    elif args.mode == "demo":
        demo_detection()
    elif args.mode == "streamlit":
        print("Starting Streamlit demo...")
        import subprocess
        subprocess.run(["streamlit", "run", "deployment/streamlit_app.py"])

if __name__ == "__main__":
    main()