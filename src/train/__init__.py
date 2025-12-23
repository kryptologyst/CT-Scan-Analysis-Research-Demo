"""Training module for CT scan analysis."""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import CTMetrics, CalibrationMetrics
from src.utils import get_device, save_checkpoint, format_time


class CTTrainer:
    """Trainer for CT scan analysis models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        save_dir: str = "checkpoints",
        num_classes: int = 2,
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            criterion: Loss function.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            device: Device to use for training.
            save_dir: Directory to save checkpoints.
            num_classes: Number of classes.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.save_dir = Path(save_dir)
        self.num_classes = num_classes
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize metrics
        self.train_metrics = CTMetrics(num_classes)
        self.val_metrics = CTMetrics(num_classes)
        self.calibration_metrics = CalibrationMetrics()
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dict[str, float]: Training metrics.
        """
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                self.train_metrics.update(predictions, labels, probabilities)
                total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        metrics = self.train_metrics.compute()
        metrics["loss"] = avg_loss
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dict[str, float]: Validation metrics.
        """
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                self.val_metrics.update(predictions, labels, probabilities)
                total_loss += loss.item()
                
                # Store for calibration
                all_probabilities.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.val_metrics.compute()
        metrics["loss"] = avg_loss
        
        # Compute calibration metrics
        if all_probabilities:
            import numpy as np
            all_probabilities = np.concatenate(all_probabilities)
            all_labels = np.concatenate(all_labels)
            
            if self.num_classes == 2:
                pos_probs = all_probabilities[:, 1]
                calib_metrics = self.calibration_metrics.compute_calibration(
                    pos_probs, all_labels
                )
                metrics.update(calib_metrics)
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        save_best: bool = True,
        save_last: bool = True,
        early_stopping_patience: int = 10,
    ) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train.
            save_best: Whether to save best model.
            save_last: Whether to save last model.
            early_stopping_patience: Patience for early stopping.
            
        Returns:
            Dict[str, Any]: Training history.
        """
        start_time = time.time()
        best_epoch = 0
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_metrics["loss"])
            self.val_losses.append(val_metrics["loss"])
            
            if "roc_auc" in val_metrics:
                self.val_scores.append(val_metrics["roc_auc"])
                current_score = val_metrics["roc_auc"]
            else:
                self.val_scores.append(val_metrics["accuracy"])
                current_score = val_metrics["accuracy"]
            
            # Print epoch results
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            if "roc_auc" in val_metrics:
                print(f"Val ROC AUC: {val_metrics['roc_auc']:.4f}")
            if "pr_auc" in val_metrics:
                print(f"Val PR AUC: {val_metrics['pr_auc']:.4f}")
            if "sensitivity" in val_metrics:
                print(f"Val Sensitivity: {val_metrics['sensitivity']:.4f}")
            if "specificity" in val_metrics:
                print(f"Val Specificity: {val_metrics['specificity']:.4f}")
            
            # Save checkpoints
            checkpoint_data = {
                "epoch": epoch,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            
            if self.scheduler:
                checkpoint_data["scheduler_state_dict"] = self.scheduler.state_dict()
            
            # Save best model
            if save_best and current_score > self.best_val_score:
                self.best_val_score = current_score
                best_epoch = epoch
                patience_counter = 0
                
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics["loss"],
                    val_metrics,
                    self.save_dir / "best_model.pth",
                    **checkpoint_data,
                )
                print(f"New best model saved! Score: {current_score:.4f}")
            else:
                patience_counter += 1
            
            # Save last model
            if save_last:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics["loss"],
                    val_metrics,
                    self.save_dir / "last_model.pth",
                    **checkpoint_data,
                )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(total_time)}")
        print(f"Best validation score: {self.best_val_score:.4f} at epoch {best_epoch + 1}")
        
        # Return training history
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_scores": self.val_scores,
            "best_score": self.best_val_score,
            "best_epoch": best_epoch,
            "total_time": total_time,
        }
        
        return history
    
    def evaluate(
        self,
        test_loader: DataLoader,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Args:
            test_loader: Test data loader.
            checkpoint_path: Path to checkpoint to load.
            
        Returns:
            Dict[str, float]: Test metrics.
        """
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        self.model.eval()
        test_metrics = CTMetrics(self.num_classes)
        calibration_metrics = CalibrationMetrics()
        
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                test_metrics.update(predictions, labels, probabilities)
                
                all_probabilities.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Compute metrics
        metrics = test_metrics.compute()
        
        # Compute calibration metrics
        if all_probabilities:
            import numpy as np
            all_probabilities = np.concatenate(all_probabilities)
            all_labels = np.concatenate(all_labels)
            
            if self.num_classes == 2:
                pos_probs = all_probabilities[:, 1]
                calib_metrics = calibration_metrics.compute_calibration(
                    pos_probs, all_labels
                )
                metrics.update(calib_metrics)
        
        return metrics
