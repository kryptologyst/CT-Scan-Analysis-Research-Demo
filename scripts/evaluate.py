#!/usr/bin/env python3
"""Evaluation script for CT scan analysis."""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns

from src.models import CTScanClassifier, CTVolumeClassifier, EfficientNetCT
from src.data import create_data_loaders
from src.metrics import CTMetrics, CalibrationMetrics
from src.eval import ModelExplainer
from src.utils import get_device, load_checkpoint


def load_model_and_checkpoint(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Load model and checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        config: Model configuration.
        device: Device to use.
        
    Returns:
        Loaded model.
    """
    # Create model
    model_config = config["model"]
    
    if model_config["name"] == "CTScanClassifier":
        model = CTScanClassifier(
            num_classes=model_config["num_classes"],
            backbone=model_config["backbone"],
            pretrained=False,  # Don't load pretrained weights
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
            pretrained=False,
            dropout=model_config["dropout"],
        )
    else:
        raise ValueError(f"Unknown model: {model_config['name']}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, model=model)
    
    model.to(device)
    model.eval()
    
    return model


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 2,
) -> Dict[str, Any]:
    """Evaluate model on test set.
    
    Args:
        model: PyTorch model.
        test_loader: Test data loader.
        device: Device to use.
        num_classes: Number of classes.
        
    Returns:
        Dictionary containing evaluation results.
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Compute metrics
    metrics = CTMetrics(num_classes)
    metrics.predictions = [all_predictions]
    metrics.targets = [all_labels]
    metrics.probabilities = [all_probabilities]
    
    results = metrics.compute()
    
    # Compute calibration metrics
    if num_classes == 2:
        calibration_metrics = CalibrationMetrics()
        pos_probs = all_probabilities[:, 1]
        calib_results = calibration_metrics.compute_calibration(pos_probs, all_labels)
        results.update(calib_results)
    
    # Store raw data for plotting
    results["raw_data"] = {
        "predictions": all_predictions.tolist(),
        "labels": all_labels.tolist(),
        "probabilities": all_probabilities.tolist(),
    }
    
    return results


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    save_path: str,
) -> None:
    """Plot confusion matrix.
    
    Args:
        predictions: Predicted labels.
        labels: True labels.
        class_names: Names of classes.
        save_path: Path to save plot.
    """
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    save_path: str,
) -> None:
    """Plot ROC curve.
    
    Args:
        probabilities: Predicted probabilities.
        labels: True labels.
        save_path: Path to save plot.
    """
    if probabilities.shape[1] == 2:
        pos_probs = probabilities[:, 1]
    else:
        pos_probs = probabilities[:, 0]
    
    fpr, tpr, _ = roc_curve(labels, pos_probs)
    auc_score = np.trapz(tpr, fpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    save_path: str,
) -> None:
    """Plot precision-recall curve.
    
    Args:
        probabilities: Predicted probabilities.
        labels: True labels.
        save_path: Path to save plot.
    """
    if probabilities.shape[1] == 2:
        pos_probs = probabilities[:, 1]
    else:
        pos_probs = probabilities[:, 0]
    
    precision, recall, _ = precision_recall_curve(labels, pos_probs)
    auc_score = np.trapz(precision, recall)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR Curve (AUC = {auc_score:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_calibration_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    n_bins: int = 10,
) -> None:
    """Plot calibration curve.
    
    Args:
        probabilities: Predicted probabilities.
        labels: True labels.
        save_path: Path to save plot.
        n_bins: Number of bins for calibration.
    """
    if probabilities.shape[1] == 2:
        pos_probs = probabilities[:, 1]
    else:
        pos_probs = probabilities[:, 0]
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = (bin_lowers + bin_uppers) / 2
    bin_accuracies = []
    bin_confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (pos_probs > bin_lower) & (pos_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = pos_probs[in_bin].mean()
            
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
    
    plt.figure(figsize=(8, 6))
    plt.plot(bin_confidences, bin_accuracies, "o-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_explainability_plots(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 5,
) -> None:
    """Generate explainability plots.
    
    Args:
        model: PyTorch model.
        test_loader: Test data loader.
        device: Device to use.
        output_dir: Output directory.
        num_samples: Number of samples to explain.
    """
    explainer = ModelExplainer(model, device)
    
    explanations = explainer.batch_explain(test_loader, num_samples)
    
    for i, explanation in enumerate(explanations):
        if "gradcam" in explanation and "overlay" in explanation["gradcam"]:
            overlay = explanation["gradcam"]["overlay"]
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(overlay)
            plt.title(f"Grad-CAM Sample {i+1}")
            plt.axis("off")
            
            plt.subplot(1, 2, 2)
            probs = explanation["prediction"]["probabilities"]
            plt.bar(["Normal", "Diseased"], probs)
            plt.title("Prediction Probabilities")
            plt.ylabel("Probability")
            
            plt.tight_layout()
            plt.savefig(output_dir / f"explainability_sample_{i+1}.png", dpi=300)
            plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate CT scan analysis model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_explain_samples",
        type=int,
        default=5,
        help="Number of samples for explainability analysis",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    print("Loading data...")
    _, _, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        slice_mode=config["data"]["slice_mode"],
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Load model
    print("Loading model...")
    model = load_model_and_checkpoint(args.checkpoint, config, device)
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(
        model,
        test_loader,
        device,
        config["model"]["num_classes"],
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in results.items():
        if metric != "raw_data":
            print(f"{metric}: {value:.4f}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Extract raw data
    raw_data = results["raw_data"]
    predictions = np.array(raw_data["predictions"])
    labels = np.array(raw_data["labels"])
    probabilities = np.array(raw_data["probabilities"])
    
    class_names = ["Normal", "Diseased"]
    
    # Plot confusion matrix
    plot_confusion_matrix(
        predictions,
        labels,
        class_names,
        output_dir / "confusion_matrix.png",
    )
    
    # Plot ROC curve
    plot_roc_curve(
        probabilities,
        labels,
        output_dir / "roc_curve.png",
    )
    
    # Plot precision-recall curve
    plot_precision_recall_curve(
        probabilities,
        labels,
        output_dir / "precision_recall_curve.png",
    )
    
    # Plot calibration curve
    plot_calibration_curve(
        probabilities,
        labels,
        output_dir / "calibration_curve.png",
    )
    
    # Generate explainability plots
    generate_explainability_plots(
        model,
        test_loader,
        device,
        output_dir,
        args.num_explain_samples,
    )
    
    # Save results
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
