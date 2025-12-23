"""Explainability and uncertainty quantification for CT scan analysis."""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for CT scans."""
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize Grad-CAM.
        
        Args:
            model: PyTorch model.
            target_layer: Name of target layer for Grad-CAM.
            device: Device to use.
        """
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        
        # Find target layer
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        else:
            self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _find_target_layer(self) -> str:
        """Find the best target layer for Grad-CAM."""
        # Look for common layer names in different architectures
        layer_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                layer_names.append(name)
        
        # Return the last convolutional layer
        if layer_names:
            return layer_names[-1]
        else:
            raise ValueError("No suitable convolutional layer found")
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Get the target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Target layer {self.target_layer} not found")
        
        # Register hooks
        self.forward_handle = target_module.register_forward_hook(forward_hook)
        self.backward_handle = target_module.register_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input tensor.
            class_idx: Class index for which to generate CAM.
            
        Returns:
            np.ndarray: Grad-CAM heatmap.
        """
        self.model.eval()
        
        # Forward pass
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_()
        
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0)
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def visualize(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Visualize Grad-CAM heatmap.
        
        Args:
            input_tensor: Input tensor.
            class_idx: Class index for which to generate CAM.
            alpha: Transparency for overlay.
            colormap: Colormap for heatmap.
            
        Returns:
            Tuple of original image and overlay.
        """
        # Generate CAM
        cam = self.generate_cam(input_tensor, class_idx)
        
        # Get original image
        if input_tensor.dim() == 4:  # Batch dimension
            image = input_tensor[0].cpu().numpy()
        else:
            image = input_tensor.cpu().numpy()
        
        # Handle different input formats
        if image.shape[0] == 3:  # RGB
            image = np.transpose(image, (1, 2, 0))
            image = np.mean(image, axis=2)  # Convert to grayscale
        elif image.shape[0] == 1:  # Grayscale
            image = image[0]
        
        # Normalize image
        image = (image - image.min()) / (image.max() - image.min())
        
        # Resize CAM to match image
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Create overlay
        import matplotlib.cm as cm
        colormap_fn = cm.get_cmap(colormap)
        heatmap = colormap_fn(cam_resized)[:, :, :3]  # Remove alpha channel
        
        overlay = alpha * heatmap + (1 - alpha) * np.stack([image] * 3, axis=2)
        
        return image, overlay
    
    def __del__(self):
        """Clean up hooks."""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()


class UncertaintyQuantification:
    """Uncertainty quantification using Monte Carlo Dropout."""
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 10,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize uncertainty quantification.
        
        Args:
            model: PyTorch model.
            num_samples: Number of Monte Carlo samples.
            device: Device to use.
        """
        self.model = model
        self.num_samples = num_samples
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        
        # Enable dropout during inference
        self._enable_dropout()
    
    def _enable_dropout(self) -> None:
        """Enable dropout layers for Monte Carlo sampling."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep dropout active
    
    def predict_with_uncertainty(
        self,
        input_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty quantification.
        
        Args:
            input_tensor: Input tensor.
            
        Returns:
            Tuple of mean predictions and uncertainty (entropy).
        """
        self.model.eval()
        
        input_tensor = input_tensor.to(self.device)
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                predictions.append(probabilities)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (num_samples, batch_size, num_classes)
        
        # Compute mean and uncertainty
        mean_predictions = torch.mean(predictions, dim=0)
        
        # Compute entropy as uncertainty measure
        entropy = -torch.sum(mean_predictions * torch.log(mean_predictions + 1e-8), dim=1)
        
        return mean_predictions, entropy
    
    def get_prediction_intervals(
        self,
        input_tensor: torch.Tensor,
        confidence_level: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get prediction intervals.
        
        Args:
            input_tensor: Input tensor.
            confidence_level: Confidence level for intervals.
            
        Returns:
            Tuple of mean, lower bound, and upper bound predictions.
        """
        self.model.eval()
        
        input_tensor = input_tensor.to(self.device)
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                predictions.append(probabilities)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)
        
        # Compute statistics
        mean_predictions = torch.mean(predictions, dim=0)
        std_predictions = torch.std(predictions, dim=0)
        
        # Compute confidence intervals
        alpha = 1 - confidence_level
        z_score = 1.96  # For 95% confidence interval
        
        lower_bound = mean_predictions - z_score * std_predictions
        upper_bound = mean_predictions + z_score * std_predictions
        
        # Clip to valid probability range
        lower_bound = torch.clamp(lower_bound, 0.0, 1.0)
        upper_bound = torch.clamp(upper_bound, 0.0, 1.0)
        
        return mean_predictions, lower_bound, upper_bound


class ModelExplainer:
    """Comprehensive model explainer combining multiple techniques."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize model explainer.
        
        Args:
            model: PyTorch model.
            device: Device to use.
        """
        self.model = model
        self.device = device or torch.device("cpu")
        
        # Initialize explainability tools
        self.gradcam = GradCAM(model, device=device)
        self.uncertainty = UncertaintyQuantification(model, device=device)
    
    def explain_prediction(
        self,
        input_tensor: torch.Tensor,
        class_names: Optional[List[str]] = None,
    ) -> dict:
        """Generate comprehensive explanation for a prediction.
        
        Args:
            input_tensor: Input tensor.
            class_names: Names of classes.
            
        Returns:
            Dict containing explanation results.
        """
        results = {}
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        results["prediction"] = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities[0].cpu().numpy().tolist(),
        }
        
        if class_names:
            results["prediction"]["class_name"] = class_names[predicted_class]
        
        # Generate Grad-CAM
        try:
            cam_image, overlay = self.gradcam.visualize(input_tensor, predicted_class)
            results["gradcam"] = {
                "heatmap": cam_image,
                "overlay": overlay,
            }
        except Exception as e:
            results["gradcam"] = {"error": str(e)}
        
        # Uncertainty quantification
        try:
            mean_pred, uncertainty = self.uncertainty.predict_with_uncertainty(input_tensor)
            results["uncertainty"] = {
                "mean_prediction": mean_pred[0].cpu().numpy().tolist(),
                "entropy": uncertainty[0].item(),
            }
        except Exception as e:
            results["uncertainty"] = {"error": str(e)}
        
        return results
    
    def batch_explain(
        self,
        data_loader: DataLoader,
        num_samples: int = 5,
        class_names: Optional[List[str]] = None,
    ) -> List[dict]:
        """Generate explanations for a batch of samples.
        
        Args:
            data_loader: Data loader.
            num_samples: Number of samples to explain.
            class_names: Names of classes.
            
        Returns:
            List of explanation results.
        """
        results = []
        
        for i, (images, labels) in enumerate(data_loader):
            if i >= num_samples:
                break
            
            for j in range(images.shape[0]):
                if len(results) >= num_samples:
                    break
                
                single_image = images[j:j+1]  # Keep batch dimension
                single_label = labels[j].item()
                
                explanation = self.explain_prediction(single_image, class_names)
                explanation["ground_truth"] = single_label
                
                results.append(explanation)
        
        return results
