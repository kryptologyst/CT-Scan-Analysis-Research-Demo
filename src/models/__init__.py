"""CT scan analysis models."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CTScanClassifier(nn.Module):
    """2D CT scan classifier using ResNet backbone.
    
    This model processes 2D CT scan slices for binary classification
    (normal vs diseased).
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.5,
    ) -> None:
        """Initialize CT scan classifier.
        
        Args:
            num_classes: Number of output classes.
            backbone: Backbone architecture ('resnet18', 'resnet50', 'efficientnet').
            pretrained: Whether to use pretrained weights.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Load backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class CTVolumeClassifier(nn.Module):
    """3D CT volume classifier using 3D ResNet.
    
    This model processes 3D CT volumes for classification.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 1,
        dropout: float = 0.5,
    ) -> None:
        """Initialize 3D CT volume classifier.
        
        Args:
            num_classes: Number of output classes.
            input_channels: Number of input channels.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # 3D ResNet-like architecture
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Module:
        """Create a ResNet layer.
        
        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            blocks: Number of blocks.
            stride: Stride for first block.
            
        Returns:
            nn.Module: ResNet layer.
        """
        layers = []
        
        # First block
        layers.append(
            BasicBlock3D(in_channels, out_channels, stride)
        )
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(
                BasicBlock3D(out_channels, out_channels)
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W).
            
        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


class BasicBlock3D(nn.Module):
    """3D ResNet basic block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        """Initialize basic block.
        
        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            stride: Stride for convolution.
        """
        super().__init__()
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=stride
                ),
                nn.BatchNorm3d(out_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class EfficientNetCT(nn.Module):
    """EfficientNet-based CT scan classifier."""
    
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.5,
    ) -> None:
        """Initialize EfficientNet CT classifier.
        
        Args:
            num_classes: Number of output classes.
            model_name: EfficientNet model name.
            pretrained: Whether to use pretrained weights.
            dropout: Dropout rate.
        """
        super().__init__()
        
        try:
            import torchvision.models as models
            if model_name == "efficientnet_b0":
                self.backbone = models.efficientnet_b0(pretrained=pretrained)
                in_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Identity()
            elif model_name == "efficientnet_b1":
                self.backbone = models.efficientnet_b1(pretrained=pretrained)
                in_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Identity()
            else:
                raise ValueError(f"Unsupported EfficientNet model: {model_name}")
        except AttributeError:
            # Fallback to ResNet if EfficientNet not available
            print("EfficientNet not available, falling back to ResNet18")
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
