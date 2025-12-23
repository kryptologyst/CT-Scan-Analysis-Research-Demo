# CT Scan Analysis Research Demo

## Medical Disclaimer

**IMPORTANT: THIS IS A RESEARCH DEMONSTRATION PROJECT**

This project is designed for educational and research purposes only. It is NOT intended for clinical use, medical diagnosis, or treatment decisions.

### Key Limitations

- This is a demonstration project using synthetic or limited datasets
- Models have not been validated on diverse clinical populations
- No regulatory approval or clinical validation has been performed
- Results should not be used for patient care decisions

### Professional Supervision Required

Any application of these techniques in clinical settings requires:
- Appropriate medical supervision
- Clinical validation studies
- Regulatory compliance
- Ethical review board approval

### No Medical Advice

This software does not provide medical advice, diagnosis, or treatment recommendations. Always consult qualified healthcare professionals for medical decisions.

### Research Use Only

This project is intended for:
- Educational purposes
- Research and development
- Academic study
- Algorithm development

**By using this software, you acknowledge and agree that it is for research purposes only and not for clinical use.**

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/CT-Scan-Analysis-Research-Demo.git
cd CT-Scan-Analysis-Research-Demo

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom configuration
python scripts/train.py --config configs/custom_config.yaml

# Resume training from checkpoint
python scripts/train.py --resume checkpoints/latest.pth
```

### Demo

```bash
# Launch interactive demo
streamlit run demo/app.py
```

## Features

- **CT Scan Classification**: Binary classification (normal vs diseased)
- **3D Volume Processing**: Support for both 2D slice and 3D volume analysis
- **Medical Imaging Pipelines**: DICOM/NIfTI support with CT-specific preprocessing
- **Clinical Evaluation**: AUROC, AUPRC, sensitivity/specificity metrics
- **Explainability**: Grad-CAM visualization and uncertainty quantification
- **Interactive Demo**: Streamlit-based web interface

## Models

- ResNet18/50 for 2D slice classification
- 3D ResNet for volume analysis
- EfficientNet and Vision Transformer variants
- Focal Loss and Tversky Loss for imbalanced data

## Data Processing

- DICOM and NIfTI format support
- CT-specific preprocessing (HU windowing, normalization)
- Data augmentation with MONAI transforms
- Patient-level data splitting

## Evaluation

- Clinical metrics (AUROC, AUPRC, sensitivity, specificity)
- Calibration analysis
- Grad-CAM explainability
- Uncertainty quantification

## Project Structure

```
ct-scan-analysis/
├── src/
│   ├── models/          # Model definitions
│   ├── data/            # Data loading and preprocessing
│   ├── losses/          # Loss functions
│   ├── metrics/         # Evaluation metrics
│   ├── utils/           # Utility functions
│   ├── train/           # Training logic
│   └── eval/            # Evaluation logic
├── configs/             # Configuration files
├── scripts/             # Training and evaluation scripts
├── notebooks/           # Jupyter notebooks
├── tests/               # Unit tests
├── assets/              # Sample outputs and visualizations
├── demo/                # Streamlit demo application
├── data/                # Dataset directory
└── checkpoints/         # Model checkpoints
```

## Configuration

Configuration files are in `configs/`:

- `configs/train.yaml`: Training parameters
- `configs/model.yaml`: Model architecture settings

## Dataset

This project uses synthetic CT scan data for demonstration. In practice, you would use datasets like:

- **COVID-CT**: COVID-19 CT scan classification
- **MosMedData**: COVID-19 CT scans
- **LUNA16**: Lung nodule detection
- **NIH Chest X-ray**: Chest X-ray classification

### Data Format

```
data/
├── train/
│   ├── normal/
│   │   ├── scan_001.nii.gz
│   │   └── scan_002.nii.gz
│   └── diseased/
│       ├── scan_003.nii.gz
│       └── scan_004.nii.gz
└── test/
    ├── normal/
    └── diseased/
```

## Metrics and Performance

The model is evaluated using clinically relevant metrics:

- **AUROC**: Area under ROC curve
- **AUPRC**: Area under Precision-Recall curve
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **Calibration**: Brier score and ECE

## Limitations

- This is a research demonstration project
- Models trained on limited/synthetic data
- No clinical validation performed
- Results should not be used for patient care

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ct_scan_analysis,
  title={CT Scan Analysis - Medical Imaging Classification},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/CT-Scan-Analysis-Research-Demo}
}
```

## Support

For questions and support, please open an issue on GitHub.

---

**Remember: This is a research demonstration project. It is NOT intended for clinical use.**
# CT-Scan-Analysis-Research-Demo
