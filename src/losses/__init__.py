"""Loss functions for CT scan analysis."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focal Loss is designed to down-weight easy examples and focus on hard examples.
    """
    
    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class. If None, uses inverse frequency.
            gamma: Focusing parameter.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Focal Loss.
        
        Args:
            inputs: Predicted logits of shape (N, C).
            targets: Ground truth labels of shape (N,).
            
        Returns:
            torch.Tensor: Focal loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            ce_loss = alpha_t * ce_loss
        
        # Apply focal weighting
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """Tversky Loss for segmentation tasks.
    
    Tversky Loss is a generalization of Dice Loss that allows for different
    weights for false positives and false negatives.
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
    ) -> None:
        """Initialize Tversky Loss.
        
        Args:
            alpha: Weight for false positives.
            beta: Weight for false negatives.
            smooth: Smoothing factor.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Tversky Loss.
        
        Args:
            inputs: Predicted probabilities of shape (N, C, H, W).
            targets: Ground truth labels of shape (N, H, W).
            
        Returns:
            torch.Tensor: Tversky loss.
        """
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Compute intersection and union
        intersection = (inputs * targets_one_hot).sum()
        fp = (inputs * (1 - targets_one_hot)).sum()
        fn = ((1 - inputs) * targets_one_hot).sum()
        
        # Compute Tversky index
        tversky_index = (intersection + self.smooth) / (
            intersection + self.alpha * fp + self.beta * fn + self.smooth
        )
        
        return 1 - tversky_index


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1.0) -> None:
        """Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor.
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice Loss.
        
        Args:
            inputs: Predicted probabilities of shape (N, C, H, W).
            targets: Ground truth labels of shape (N, H, W).
            
        Returns:
            torch.Tensor: Dice loss.
        """
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Compute intersection and union
        intersection = (inputs * targets_one_hot).sum()
        union = inputs.sum() + targets_one_hot.sum()
        
        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined loss function for CT scan analysis.
    
    Combines CrossEntropy with Focal Loss for better performance
    on imbalanced medical datasets.
    """
    
    def __init__(
        self,
        ce_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
    ) -> None:
        """Initialize Combined Loss.
        
        Args:
            ce_weight: Weight for CrossEntropy loss.
            focal_weight: Weight for Focal loss.
            focal_gamma: Gamma parameter for Focal loss.
            focal_alpha: Alpha parameter for Focal loss.
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            inputs: Predicted logits of shape (N, C).
            targets: Ground truth labels of shape (N,).
            
        Returns:
            torch.Tensor: Combined loss.
        """
        ce = self.ce_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        
        return self.ce_weight * ce + self.focal_weight * focal


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Cross Entropy Loss.
    
    Helps prevent overfitting and improves generalization.
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
    ) -> None:
        """Initialize Label Smoothing Loss.
        
        Args:
            num_classes: Number of classes.
            smoothing: Smoothing factor.
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Label Smoothing Loss.
        
        Args:
            inputs: Predicted logits of shape (N, C).
            targets: Ground truth labels of shape (N,).
            
        Returns:
            torch.Tensor: Label smoothing loss.
        """
        log_preds = F.log_softmax(inputs, dim=1)
        
        # Create smoothed labels
        smooth_labels = torch.zeros_like(log_preds)
        smooth_labels.fill_(self.smoothing / (self.num_classes - 1))
        smooth_labels.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # Compute loss
        loss = -smooth_labels * log_preds
        loss = loss.sum(dim=1).mean()
        
        return loss
