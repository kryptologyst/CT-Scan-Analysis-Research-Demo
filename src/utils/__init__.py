"""Utility functions for CT scan analysis project."""

import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: str,
    **kwargs: Any,
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        epoch: Current epoch.
        loss: Current loss.
        metrics: Dictionary of metrics.
        filepath: Path to save checkpoint.
        **kwargs: Additional data to save.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics,
        **kwargs,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file.
        model: Optional model to load state into.
        optimizer: Optional optimizer to load state into.
        
    Returns:
        Dict containing checkpoint data.
    """
    checkpoint = torch.load(filepath, map_location=get_device())
    
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        str: Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def ensure_dir(path: str) -> None:
    """Ensure directory exists.
    
    Args:
        path: Directory path.
    """
    import os
    os.makedirs(path, exist_ok=True)
