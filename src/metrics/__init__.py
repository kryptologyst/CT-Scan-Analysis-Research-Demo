"""Evaluation metrics for CT scan analysis."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


class CTMetrics:
    """Comprehensive metrics for CT scan analysis."""
    
    def __init__(self, num_classes: int = 2) -> None:
        """Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes.
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self.probabilities: List[np.ndarray] = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
    ) -> None:
        """Update metrics with new batch.
        
        Args:
            predictions: Predicted class labels.
            targets: Ground truth labels.
            probabilities: Predicted probabilities (optional).
        """
        self.predictions.append(predictions.cpu().numpy())
        self.targets.append(targets.cpu().numpy())
        
        if probabilities is not None:
            self.probabilities.append(probabilities.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dict[str, float]: Dictionary of computed metrics.
        """
        if not self.predictions:
            return {}
        
        # Concatenate all predictions and targets
        all_predictions = np.concatenate(self.predictions)
        all_targets = np.concatenate(self.targets)
        
        metrics = {}
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(all_targets, all_predictions)
        
        # Confusion matrix metrics
        cm = confusion_matrix(all_targets, all_predictions)
        if self.num_classes == 2:
            tn, fp, fn, tp = cm.ravel()
            
            metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics["recall"] = metrics["sensitivity"]
            metrics["f1_score"] = (
                2 * metrics["precision"] * metrics["recall"] / 
                (metrics["precision"] + metrics["recall"])
                if (metrics["precision"] + metrics["recall"]) > 0 else 0.0
            )
            
            # NPV and PPV
            metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            metrics["ppv"] = metrics["precision"]
        
        # ROC and PR metrics (if probabilities available)
        if self.probabilities:
            all_probabilities = np.concatenate(self.probabilities)
            
            if self.num_classes == 2:
                # Binary classification
                if all_probabilities.ndim > 1:
                    # Use positive class probabilities
                    pos_probs = all_probabilities[:, 1]
                else:
                    pos_probs = all_probabilities
                
                # ROC AUC
                try:
                    metrics["roc_auc"] = roc_auc_score(all_targets, pos_probs)
                except ValueError:
                    metrics["roc_auc"] = 0.0
                
                # PR AUC
                try:
                    metrics["pr_auc"] = average_precision_score(all_targets, pos_probs)
                except ValueError:
                    metrics["pr_auc"] = 0.0
                
                # ROC curve for threshold analysis
                fpr, tpr, roc_thresholds = roc_curve(all_targets, pos_probs)
                precision, recall, pr_thresholds = precision_recall_curve(
                    all_targets, pos_probs
                )
                
                # Find optimal threshold (Youden's J statistic)
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                metrics["optimal_threshold"] = roc_thresholds[optimal_idx]
                metrics["optimal_sensitivity"] = tpr[optimal_idx]
                metrics["optimal_specificity"] = 1 - fpr[optimal_idx]
            
            else:
                # Multi-class classification
                try:
                    metrics["roc_auc_ovr"] = roc_auc_score(
                        all_targets, all_probabilities, multi_class="ovr"
                    )
                    metrics["roc_auc_ovo"] = roc_auc_score(
                        all_targets, all_probabilities, multi_class="ovo"
                    )
                except ValueError:
                    metrics["roc_auc_ovr"] = 0.0
                    metrics["roc_auc_ovo"] = 0.0
        
        return metrics


class CalibrationMetrics:
    """Calibration metrics for model reliability."""
    
    def __init__(self, n_bins: int = 10) -> None:
        """Initialize calibration metrics.
        
        Args:
            n_bins: Number of bins for calibration analysis.
        """
        self.n_bins = n_bins
    
    def compute_calibration(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, float]:
        """Compute calibration metrics.
        
        Args:
            probabilities: Predicted probabilities.
            targets: Ground truth binary labels.
            
        Returns:
            Dict[str, float]: Calibration metrics.
        """
        if probabilities.ndim > 1:
            probabilities = probabilities[:, 1]  # Use positive class
        
        # Brier score
        brier_score = np.mean((probabilities - targets) ** 2)
        
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(probabilities, targets)
        
        # Maximum Calibration Error (MCE)
        mce = self._compute_mce(probabilities, targets)
        
        return {
            "brier_score": brier_score,
            "ece": ece,
            "mce": mce,
        }
    
    def _compute_ece(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = targets[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _compute_mce(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Compute Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = targets[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce


def compute_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    """Generate classification report.
    
    Args:
        predictions: Predicted labels.
        targets: Ground truth labels.
        class_names: Names of classes.
        
    Returns:
        str: Classification report.
    """
    from sklearn.metrics import classification_report
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(targets)))]
    
    return classification_report(
        targets,
        predictions,
        target_names=class_names,
        digits=4,
    )
