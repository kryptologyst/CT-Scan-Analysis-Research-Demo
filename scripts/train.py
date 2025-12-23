#!/usr/bin/env python3
"""Training script for CT scan analysis."""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models import CTScanClassifier, CTVolumeClassifier, EfficientNetCT
from src.data import create_data_loaders, get_ct_transforms
from src.losses import (
    FocalLoss,
    CombinedLoss,
    LabelSmoothingLoss,
)
from src.metrics import CTMetrics
from src.train import CTTrainer
from src.utils import set_seed, get_device, ensure_dir


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Dict containing configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any]) -> nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Model configuration.
        
    Returns:
        PyTorch model.
    """
    model_config = config["model"]
    
    if model_config["name"] == "CTScanClassifier":
        model = CTScanClassifier(
            num_classes=model_config["num_classes"],
            backbone=model_config["backbone"],
            pretrained=model_config["pretrained"],
            dropout=model_config["dropout"],
        )
    elif model_config["name"] == "CTVolumeClassifier":
        model = CTVolumeClassifier(
            num_classes=model_config["num_classes"],
            input_channels=model_config.get("input_channels", 1),
            dropout=model_config["dropout"],
        )
    elif model_config["name"] == "EfficientNetCT":
        model = EfficientNetCT(
            num_classes=model_config["num_classes"],
            model_name=model_config.get("model_name", "efficientnet_b0"),
            pretrained=model_config["pretrained"],
            dropout=model_config["dropout"],
        )
    else:
        raise ValueError(f"Unknown model: {model_config['name']}")
    
    return model


def create_criterion(config: Dict[str, Any]) -> nn.Module:
    """Create loss function based on configuration.
    
    Args:
        config: Loss configuration.
        
    Returns:
        Loss function.
    """
    loss_config = config["loss"]
    
    if loss_config["name"] == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_config["name"] == "focal":
        return FocalLoss(
            alpha=loss_config.get("alpha"),
            gamma=loss_config.get("gamma", 2.0),
        )
    elif loss_config["name"] == "combined":
        return CombinedLoss(
            ce_weight=loss_config.get("ce_weight", 0.5),
            focal_weight=loss_config.get("focal_weight", 0.5),
            focal_gamma=loss_config.get("focal_gamma", 2.0),
            focal_alpha=loss_config.get("focal_alpha"),
        )
    elif loss_config["name"] == "label_smoothing":
        return LabelSmoothingLoss(
            num_classes=loss_config.get("num_classes", 2),
            smoothing=loss_config.get("smoothing", 0.1),
        )
    else:
        raise ValueError(f"Unknown loss: {loss_config['name']}")


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
) -> optim.Optimizer:
    """Create optimizer based on configuration.
    
    Args:
        model: PyTorch model.
        config: Optimizer configuration.
        
    Returns:
        Optimizer.
    """
    training_config = config["training"]
    optimizer_name = training_config["optimizer"]
    lr = training_config["learning_rate"]
    weight_decay = training_config.get("weight_decay", 0.0)
    
    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    elif optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: optim.Optimizer,
    config: Dict[str, Any],
) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer.
        config: Scheduler configuration.
        
    Returns:
        Learning rate scheduler.
    """
    training_config = config["training"]
    scheduler_name = training_config.get("scheduler")
    
    if scheduler_name is None:
        return None
    
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config["num_epochs"],
        )
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.1,
        )
    elif scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CT scan analysis model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    
    # Set random seed
    set_seed(config.get("seed", 42))
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        slice_mode=config["data"]["slice_mode"],
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    criterion = create_criterion(config)
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    
    # Create trainer
    trainer = CTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=output_dir / "checkpoints",
        num_classes=config["model"]["num_classes"],
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.current_epoch = checkpoint["epoch"]
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        num_epochs=config["training"]["num_epochs"],
        save_best=True,
        save_last=True,
        early_stopping_patience=config["training"]["early_stopping_patience"],
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(
        test_loader,
        checkpoint_path=output_dir / "checkpoints" / "best_model.pth",
    )
    
    print("\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save final results
    results = {
        "config": config,
        "history": history,
        "test_metrics": test_metrics,
    }
    
    import json
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
