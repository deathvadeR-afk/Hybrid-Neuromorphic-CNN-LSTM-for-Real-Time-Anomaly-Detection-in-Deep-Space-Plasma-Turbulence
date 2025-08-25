# src/training/trainer.py
"""Main training pipeline with MLflow experiment tracking."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import numpy as np
import time
import os
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import json
from pathlib import Path

from ..models.hybrid_model import HybridNeuromorphicModel
from ..config import get_config
from .losses import CombinedLoss
from .metrics import AnomalyMetrics
from .scheduler import CosineWarmupScheduler

class Trainer:
    """
    Training pipeline for hybrid neuromorphic plasma anomaly detection model.
    
    Integrates MLflow for experiment tracking, supports distributed training,
    and implements advanced training techniques for optimal performance.
    """
    
    def __init__(self,
                 model: HybridNeuromorphicModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 experiment_name: str = "neuromorphic_plasma_detection"):
        """
        Initialize trainer.
        
        Args:
            model: Hybrid neuromorphic model
            train_loader: Training data loader
            val_loader: Validation data loader  
            test_loader: Optional test data loader
            experiment_name: MLflow experiment name
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Get configurations
        self.train_config = get_config("training")
        self.eval_config = get_config("evaluation")
        
        # Setup device
        self.device = torch.device(self.train_config.device)
        self.model = self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = CombinedLoss(
            anomaly_weight=self.train_config.anomaly_loss_weight,
            reconstruction_weight=self.train_config.reconstruction_loss_weight
        )
        
        # Initialize optimizer
        self.optimizer = self._setup_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._setup_scheduler()
        
        # Initialize metrics
        self.metrics = AnomalyMetrics()
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.best_model_path = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': [],
            'inference_times': [],
            'power_estimates': []
        }
        
        # Early stopping
        self.patience_counter = 0
        
        # MLflow setup
        self.experiment_name = experiment_name
        self._setup_mlflow()
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration."""
        if self.train_config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay
            )
        elif self.train_config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay
            )
        elif self.train_config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.train_config.learning_rate,
                momentum=0.9,
                weight_decay=self.train_config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.train_config.optimizer}")
    
    def _setup_scheduler(self) -> Optional[object]:
        """Setup learning rate scheduler."""
        if self.train_config.scheduler.lower() == "cosine":
            return CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_epochs=5,
                max_epochs=self.train_config.num_epochs
            )
        elif self.train_config.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.5
            )
        elif self.train_config.scheduler.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return None
    
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        mlflow.set_experiment(self.experiment_name)
        
        # Start MLflow run
        mlflow.start_run()
        
        # Log model parameters
        model_complexity = self.model.get_model_complexity()
        for key, value in model_complexity.items():
            mlflow.log_param(key, value)
        
        # Log training configuration
        train_params = {
            'batch_size': self.train_config.batch_size,
            'learning_rate': self.train_config.learning_rate,
            'num_epochs': self.train_config.num_epochs,
            'optimizer': self.train_config.optimizer,
            'scheduler': self.train_config.scheduler,
            'device': str(self.device)
        }
        
        for key, value in train_params.items():
            mlflow.log_param(key, value)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (sequences, spectrograms, labels) in enumerate(pbar):
            # Move to device
            sequences = sequences.to(self.device)
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(sequences, spectrograms)
            
            # Calculate loss
            loss_dict = self.criterion(outputs, sequences, labels)
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                batch_metrics = self.metrics.calculate_batch_metrics(
                    outputs['classification_probs'],
                    labels,
                    threshold=0.5
                )
            
            # Update epoch metrics
            epoch_loss += total_loss.item()
            for key in epoch_metrics:
                epoch_metrics[key] += batch_metrics[key]
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Acc': f"{batch_metrics['accuracy']:.3f}",
                'F1': f"{batch_metrics['f1']:.3f}"
            })
        
        # Average metrics
        epoch_loss /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return {
            'loss': epoch_loss,
            **epoch_metrics
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Validation metrics for the epoch
        """
        self.model.eval()
        
        epoch_loss = 0.0
        epoch_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        num_batches = len(self.val_loader)
        
        inference_times = []
        
        with torch.no_grad():
            for sequences, spectrograms, labels in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                sequences = sequences.to(self.device)
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device)
                
                # Time inference
                start_time = time.time()
                outputs = self.model(sequences, spectrograms)
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                inference_times.append(inference_time)
                
                # Calculate loss
                loss_dict = self.criterion(outputs, sequences, labels)
                total_loss = loss_dict['total_loss']
                
                # Calculate metrics
                batch_metrics = self.metrics.calculate_batch_metrics(
                    outputs['classification_probs'],
                    labels,
                    threshold=0.5
                )
                
                # Update epoch metrics
                epoch_loss += total_loss.item()
                for key in epoch_metrics:
                    epoch_metrics[key] += batch_metrics[key]
        
        # Average metrics
        epoch_loss /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Calculate average inference time
        avg_inference_time = np.mean(inference_times)
        
        return {
            'loss': epoch_loss,
            'avg_inference_time_ms': avg_inference_time,
            **epoch_metrics
        }
    
    def train(self, 
              num_epochs: Optional[int] = None,
              save_checkpoints: bool = True) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train (uses config if None)
            save_checkpoints: Whether to save model checkpoints
            
        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.train_config.num_epochs
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_model_complexity()['total_parameters']:,}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1'])
                else:
                    self.scheduler.step()
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_accuracy'].append(train_metrics['accuracy'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['train_f1'].append(train_metrics['f1'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            self.training_history['inference_times'].append(val_metrics['avg_inference_time_ms'])
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_accuracy': val_metrics['accuracy'],
                'train_f1': train_metrics['f1'],
                'val_f1': val_metrics['f1'],
                'inference_time_ms': val_metrics['avg_inference_time_ms'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # Check for best model
            current_score = val_metrics['f1']  # Use F1 as primary metric
            if current_score > self.best_val_score:
                self.best_val_score = current_score
                self.patience_counter = 0
                
                if save_checkpoints:
                    self.save_checkpoint(epoch, is_best=True)
                    
            else:
                self.patience_counter += 1
            
            # Print epoch results
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            print(f"Train Acc: {train_metrics['accuracy']:.3f}, Val Acc: {val_metrics['accuracy']:.3f}")
            print(f"Train F1: {train_metrics['f1']:.3f}, Val F1: {val_metrics['f1']:.3f}")
            print(f"Inference Time: {val_metrics['avg_inference_time_ms']:.2f}ms")
            
            # Check performance targets
            meets_accuracy = val_metrics['accuracy'] >= self.eval_config.target_accuracy
            meets_latency = val_metrics['avg_inference_time_ms'] <= self.eval_config.max_inference_time_ms
            
            if meets_accuracy and meets_latency:
                print(f"✅ Model meets performance targets!")
            
            # Early stopping
            if self.patience_counter >= self.train_config.early_stopping_patience:
                print(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                break
            
            # Save periodic checkpoints
            if save_checkpoints and (epoch + 1) % self.train_config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        # Final evaluation
        if self.test_loader is not None:
            print("\nRunning final test evaluation...")
            test_metrics = self.evaluate_test()
            
            # Log test metrics to MLflow
            mlflow.log_metrics({
                'test_accuracy': test_metrics['accuracy'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_f1': test_metrics['f1'],
                'test_auc_roc': test_metrics['auc_roc']
            })
        
        # Save final model
        if save_checkpoints:
            self.save_final_model()
        
        # End MLflow run
        mlflow.end_run()
        
        print(f"\nTraining completed! Best validation F1: {self.best_val_score:.4f}")
        
        return self.training_history
    
    def evaluate_test(self) -> Dict[str, float]:
        """Evaluate model on test set."""
        if self.test_loader is None:
            raise ValueError("Test loader not provided")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for sequences, spectrograms, labels in tqdm(self.test_loader, desc="Testing"):
                sequences = sequences.to(self.device)
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences, spectrograms)
                
                # Collect predictions and labels
                probs = outputs['classification_probs']
                predictions = (probs[:, 1] > 0.5).float()
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)
        
        # Calculate comprehensive metrics
        test_metrics = self.metrics.calculate_comprehensive_metrics(
            all_probs, all_labels, all_predictions
        )
        
        return test_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.train_config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_score': self.best_val_score,
            'training_history': self.training_history
        }
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_model_path = str(best_path)
            
            # Log best model to MLflow
            mlflow.pytorch.log_model(self.model, "best_model")
    
    def save_final_model(self):
        """Save final trained model."""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        final_path = model_dir / "neuromorphic_plasma_detector.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_model_complexity(),
            'training_config': self.train_config.__dict__,
            'best_val_score': self.best_val_score
        }, final_path)
        
        # Log final model to MLflow
        mlflow.pytorch.log_model(self.model, "final_model")
        
        print(f"Final model saved to {final_path}")