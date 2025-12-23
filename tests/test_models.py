"""Tests for CT scan analysis project."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models import CTScanClassifier, CTVolumeClassifier, EfficientNetCT
from src.losses import FocalLoss, CombinedLoss, DiceLoss, TverskyLoss
from src.metrics import CTMetrics, CalibrationMetrics
from src.utils import set_seed, get_device, count_parameters


class TestModels:
    """Test model implementations."""
    
    def test_ct_scan_classifier(self):
        """Test CT scan classifier."""
        model = CTScanClassifier(num_classes=2)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (2, 2)
        assert torch.allclose(torch.sum(torch.softmax(output, dim=1), dim=1), torch.ones(2))
    
    def test_ct_volume_classifier(self):
        """Test CT volume classifier."""
        model = CTVolumeClassifier(num_classes=2)
        
        # Test forward pass
        x = torch.randn(2, 1, 64, 64, 64)
        output = model(x)
        
        assert output.shape == (2, 2)
        assert torch.allclose(torch.sum(torch.softmax(output, dim=1), dim=1), torch.ones(2))
    
    def test_efficientnet_ct(self):
        """Test EfficientNet CT classifier."""
        model = EfficientNetCT(num_classes=2)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (2, 2)
        assert torch.allclose(torch.sum(torch.softmax(output, dim=1), dim=1), torch.ones(2))


class TestLosses:
    """Test loss function implementations."""
    
    def test_focal_loss(self):
        """Test focal loss."""
        loss_fn = FocalLoss(gamma=2.0)
        
        # Test forward pass
        inputs = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4,))
        
        loss = loss_fn(inputs, targets)
        assert loss.item() >= 0
        assert loss.shape == torch.Size([])
    
    def test_combined_loss(self):
        """Test combined loss."""
        loss_fn = CombinedLoss()
        
        # Test forward pass
        inputs = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4,))
        
        loss = loss_fn(inputs, targets)
        assert loss.item() >= 0
        assert loss.shape == torch.Size([])
    
    def test_dice_loss(self):
        """Test dice loss."""
        loss_fn = DiceLoss()
        
        # Test forward pass
        inputs = torch.rand(2, 2, 32, 32)
        targets = torch.randint(0, 2, (2, 32, 32))
        
        loss = loss_fn(inputs, targets)
        assert loss.item() >= 0
        assert loss.shape == torch.Size([])
    
    def test_tversky_loss(self):
        """Test Tversky loss."""
        loss_fn = TverskyLoss()
        
        # Test forward pass
        inputs = torch.rand(2, 2, 32, 32)
        targets = torch.randint(0, 2, (2, 32, 32))
        
        loss = loss_fn(inputs, targets)
        assert loss.item() >= 0
        assert loss.shape == torch.Size([])


class TestMetrics:
    """Test metrics implementations."""
    
    def test_ct_metrics(self):
        """Test CT metrics."""
        metrics = CTMetrics(num_classes=2)
        
        # Test update
        predictions = torch.randint(0, 2, (10,))
        targets = torch.randint(0, 2, (10,))
        probabilities = torch.rand(10, 2)
        
        metrics.update(predictions, targets, probabilities)
        
        # Test compute
        results = metrics.compute()
        
        assert "accuracy" in results
        assert "loss" in results
        assert 0 <= results["accuracy"] <= 1
    
    def test_calibration_metrics(self):
        """Test calibration metrics."""
        calib_metrics = CalibrationMetrics()
        
        # Test calibration computation
        probabilities = np.random.rand(100)
        targets = np.random.randint(0, 2, 100)
        
        results = calib_metrics.compute_calibration(probabilities, targets)
        
        assert "brier_score" in results
        assert "ece" in results
        assert "mce" in results
        assert 0 <= results["brier_score"] <= 1
        assert results["ece"] >= 0
        assert results["mce"] >= 0


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate random numbers
        torch_rand = torch.rand(1)
        np_rand = np.random.rand(1)
        
        # Reset seed and generate again
        set_seed(42)
        torch_rand2 = torch.rand(1)
        np_rand2 = np.random.rand(1)
        
        # Should be the same
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = CTScanClassifier(num_classes=2)
        num_params = count_parameters(model)
        
        assert num_params > 0
        assert isinstance(num_params, int)


class TestDataIntegration:
    """Test data integration."""
    
    def test_model_training_step(self):
        """Test a single training step."""
        model = CTScanClassifier(num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Create dummy data
        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 2, (2,))
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0
    
    def test_model_evaluation_step(self):
        """Test a single evaluation step."""
        model = CTScanClassifier(num_classes=2)
        model.eval()
        
        # Create dummy data
        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 2, (2,))
        
        # Evaluation step
        with torch.no_grad():
            output = model(x)
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(output, dim=1)
        
        assert output.shape == (2, 2)
        assert probabilities.shape == (2, 2)
        assert predictions.shape == (2,)
        assert torch.allclose(torch.sum(probabilities, dim=1), torch.ones(2))


if __name__ == "__main__":
    pytest.main([__file__])
