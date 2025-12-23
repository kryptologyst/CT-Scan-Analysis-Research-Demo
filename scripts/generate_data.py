#!/usr/bin/env python3
"""Generate synthetic CT scan data for demonstration purposes."""

import argparse
import numpy as np
from pathlib import Path
import nibabel as nib
from PIL import Image
import os


def generate_synthetic_ct_slice(
    size: tuple = (224, 224),
    noise_level: float = 0.1,
    anomaly_prob: float = 0.3,
) -> np.ndarray:
    """Generate a synthetic CT slice.
    
    Args:
        size: Size of the image (height, width).
        noise_level: Level of noise to add.
        anomaly_prob: Probability of adding anomalies.
        
    Returns:
        np.ndarray: Synthetic CT slice.
    """
    height, width = size
    
    # Create base image with anatomical structure
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width // 2, height // 2
    
    # Create elliptical body outline
    a, b = width // 3, height // 3
    body_mask = ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2 <= 1
    
    # Create lung regions
    lung_left = ((x - center_x + width // 6) / (width // 8)) ** 2 + ((y - center_y) / (height // 4)) ** 2 <= 1
    lung_right = ((x - center_x - width // 6) / (width // 8)) ** 2 + ((y - center_y) / (height // 4)) ** 2 <= 1
    
    # Create spine
    spine = (abs(x - center_x) <= 3) & (y >= center_y - height // 6) & (y <= center_y + height // 6)
    
    # Create synthetic CT image
    ct_image = np.zeros((height, width))
    
    # Set different HU values for different structures
    ct_image[body_mask] = 40  # Soft tissue
    ct_image[lung_left | lung_right] = -800  # Air in lungs
    ct_image[spine] = 400  # Bone
    
    # Add noise
    noise = np.random.normal(0, noise_level * 100, (height, width))
    ct_image += noise
    
    # Add anomalies for diseased cases
    if np.random.random() < anomaly_prob:
        # Add ground glass opacity
        anomaly_mask = ((x - center_x) / (width // 4)) ** 2 + ((y - center_y) / (height // 4)) ** 2 <= 1
        ct_image[anomaly_mask] += np.random.normal(0, 50, np.sum(anomaly_mask))
        
        # Add consolidation
        consolidation_mask = ((x - center_x + width // 8) / (width // 6)) ** 2 + ((y - center_y - height // 8) / (height // 6)) ** 2 <= 1
        ct_image[consolidation_mask] += np.random.normal(100, 30, np.sum(consolidation_mask))
    
    # Normalize to [0, 1]
    ct_image = (ct_image - ct_image.min()) / (ct_image.max() - ct_image.min())
    
    return ct_image


def generate_synthetic_ct_volume(
    size: tuple = (64, 64, 64),
    noise_level: float = 0.1,
    anomaly_prob: float = 0.3,
) -> np.ndarray:
    """Generate a synthetic CT volume.
    
    Args:
        size: Size of the volume (depth, height, width).
        noise_level: Level of noise to add.
        anomaly_prob: Probability of adding anomalies.
        
    Returns:
        np.ndarray: Synthetic CT volume.
    """
    depth, height, width = size
    
    # Generate volume slice by slice
    volume = np.zeros((depth, height, width))
    
    for z in range(depth):
        # Vary anatomy along z-axis
        slice_size = (height, width)
        slice_image = generate_synthetic_ct_slice(
            slice_size,
            noise_level,
            anomaly_prob if z > depth // 4 and z < 3 * depth // 4 else 0.1,
        )
        volume[z] = slice_image
    
    return volume


def save_as_nifti(
    image: np.ndarray,
    filepath: Path,
    is_volume: bool = False,
) -> None:
    """Save image as NIfTI file.
    
    Args:
        image: Image array.
        filepath: Path to save file.
        is_volume: Whether image is a 3D volume.
    """
    if is_volume:
        # 3D volume
        affine = np.eye(4)
        nii_img = nib.Nifti1Image(image, affine)
    else:
        # 2D slice
        affine = np.eye(4)
        nii_img = nib.Nifti1Image(image, affine)
    
    nib.save(nii_img, filepath)


def save_as_png(
    image: np.ndarray,
    filepath: Path,
) -> None:
    """Save image as PNG file.
    
    Args:
        image: Image array.
        filepath: Path to save file.
    """
    # Convert to PIL Image
    image_uint8 = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8, mode='L')
    pil_image.save(filepath)


def create_synthetic_dataset(
    output_dir: str,
    num_normal: int = 100,
    num_diseased: int = 100,
    image_format: str = "png",
    is_volume: bool = False,
) -> None:
    """Create synthetic CT scan dataset.
    
    Args:
        output_dir: Output directory.
        num_normal: Number of normal samples.
        num_diseased: Number of diseased samples.
        image_format: Image format ('png', 'nii', 'nii.gz').
        is_volume: Whether to generate 3D volumes.
    """
    output_path = Path(output_dir)
    
    # Create directory structure
    train_normal_dir = output_path / "train" / "normal"
    train_diseased_dir = output_path / "train" / "diseased"
    test_normal_dir = output_path / "test" / "normal"
    test_diseased_dir = output_path / "test" / "diseased"
    
    for dir_path in [train_normal_dir, train_diseased_dir, test_normal_dir, test_diseased_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate normal samples
    print("Generating normal samples...")
    for i in range(num_normal):
        if is_volume:
            image = generate_synthetic_ct_volume(anomaly_prob=0.0)
        else:
            image = generate_synthetic_ct_slice(anomaly_prob=0.0)
        
        # Split between train and test
        if i < int(0.8 * num_normal):
            if image_format == "png":
                save_as_png(image, train_normal_dir / f"normal_{i:03d}.png")
            else:
                save_as_nifti(image, train_normal_dir / f"normal_{i:03d}.{image_format}", is_volume)
        else:
            if image_format == "png":
                save_as_png(image, test_normal_dir / f"normal_{i:03d}.png")
            else:
                save_as_nifti(image, test_normal_dir / f"normal_{i:03d}.{image_format}", is_volume)
    
    # Generate diseased samples
    print("Generating diseased samples...")
    for i in range(num_diseased):
        if is_volume:
            image = generate_synthetic_ct_volume(anomaly_prob=0.7)
        else:
            image = generate_synthetic_ct_slice(anomaly_prob=0.7)
        
        # Split between train and test
        if i < int(0.8 * num_diseased):
            if image_format == "png":
                save_as_png(image, train_diseased_dir / f"diseased_{i:03d}.png")
            else:
                save_as_nifti(image, train_diseased_dir / f"diseased_{i:03d}.{image_format}", is_volume)
        else:
            if image_format == "png":
                save_as_png(image, test_diseased_dir / f"diseased_{i:03d}.png")
            else:
                save_as_nifti(image, test_diseased_dir / f"diseased_{i:03d}.{image_format}", is_volume)
    
    print(f"Synthetic dataset created in {output_dir}")
    print(f"Format: {image_format}")
    print(f"Volume: {is_volume}")
    print(f"Normal samples: {num_normal}")
    print(f"Diseased samples: {num_diseased}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate synthetic CT scan dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/synthetic",
        help="Output directory for synthetic data",
    )
    parser.add_argument(
        "--num_normal",
        type=int,
        default=100,
        help="Number of normal samples",
    )
    parser.add_argument(
        "--num_diseased",
        type=int,
        default=100,
        help="Number of diseased samples",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "nii", "nii.gz"],
        help="Image format",
    )
    parser.add_argument(
        "--volume",
        action="store_true",
        help="Generate 3D volumes instead of 2D slices",
    )
    
    args = parser.parse_args()
    
    create_synthetic_dataset(
        output_dir=args.output_dir,
        num_normal=args.num_normal,
        num_diseased=args.num_diseased,
        image_format=args.format,
        is_volume=args.volume,
    )


if __name__ == "__main__":
    main()
