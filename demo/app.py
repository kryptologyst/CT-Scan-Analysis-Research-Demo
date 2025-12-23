"""Streamlit demo for CT scan analysis."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io
import base64

from src.models import CTScanClassifier
from src.eval import ModelExplainer
from src.utils import get_device, set_seed


# Page configuration
st.set_page_config(
    page_title="CT Scan Analysis - Research Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Medical disclaimer banner
st.error("""
**MEDICAL DISCLAIMER**: This is a research demonstration project for educational purposes only. 
It is NOT intended for clinical use, medical diagnosis, or treatment decisions. 
Always consult qualified healthcare professionals for medical decisions.
""")

# Title and description
st.title("🏥 CT Scan Analysis - Research Demo")
st.markdown("""
This demo showcases CT scan classification using deep learning. Upload a CT scan image to see:
- Disease classification (normal vs diseased)
- Confidence scores
- Explainability visualizations (Grad-CAM)
- Uncertainty quantification
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["ResNet18", "ResNet50", "EfficientNet-B0"],
    help="Select the model architecture"
)

# Load model (simplified - in practice, load from checkpoint)
@st.cache_resource
def load_model(model_type: str):
    """Load the selected model."""
    set_seed(42)
    
    if model_type == "ResNet18":
        model = CTScanClassifier(
            num_classes=2,
            backbone="resnet18",
            pretrained=True,
        )
    elif model_type == "ResNet50":
        model = CTScanClassifier(
            num_classes=2,
            backbone="resnet50",
            pretrained=True,
        )
    elif model_type == "EfficientNet-B0":
        model = CTScanClassifier(
            num_classes=2,
            backbone="efficientnet_b0",
            pretrained=True,
        )
    
    device = get_device()
    model.to(device)
    model.eval()
    
    return model, device

# Load model
model, device = load_model(model_type)

# Class names
class_names = ["Normal", "Diseased"]

# File upload
st.header("📁 Upload CT Scan Image")
uploaded_file = st.file_uploader(
    "Choose a CT scan image",
    type=['png', 'jpg', 'jpeg', 'nii', 'nii.gz'],
    help="Upload a CT scan image for analysis"
)

if uploaded_file is not None:
    # Process uploaded file
    try:
        if uploaded_file.name.endswith(('.nii', '.nii.gz')):
            st.warning("NIfTI files are not supported in this demo. Please use PNG/JPG images.")
        else:
            # Load image
            image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
            
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded CT Scan", use_column_width=True)
            
            # Preprocess image
            import torchvision.transforms as transforms
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
            
            # Convert PIL to tensor
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            with torch.no_grad():
                image_tensor = image_tensor.to(device)
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            # Display results
            with col2:
                st.subheader("Analysis Results")
                
                # Prediction
                st.metric(
                    "Predicted Class",
                    class_names[predicted_class],
                    f"{confidence:.1%} confidence"
                )
                
                # Probability distribution
                st.subheader("Class Probabilities")
                prob_data = probabilities[0].cpu().numpy()
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=class_names,
                        y=prob_data,
                        marker_color=['green' if i == predicted_class else 'lightgray' 
                                    for i in range(len(class_names))]
                    )
                ])
                fig.update_layout(
                    title="Prediction Probabilities",
                    xaxis_title="Class",
                    yaxis_title="Probability",
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Explainability section
            st.header("🔍 Explainability Analysis")
            
            try:
                # Create explainer
                explainer = ModelExplainer(model, device)
                
                # Generate explanation
                explanation = explainer.explain_prediction(image_tensor, class_names)
                
                # Display Grad-CAM
                if "gradcam" in explanation and "overlay" in explanation["gradcam"]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Grad-CAM Heatmap")
                        overlay = explanation["gradcam"]["overlay"]
                        st.image(overlay, caption="Grad-CAM Visualization", use_column_width=True)
                    
                    with col2:
                        st.subheader("Uncertainty Analysis")
                        if "uncertainty" in explanation and "entropy" in explanation["uncertainty"]:
                            entropy = explanation["uncertainty"]["entropy"]
                            st.metric("Prediction Entropy", f"{entropy:.4f}")
                            
                            # Uncertainty interpretation
                            if entropy < 0.5:
                                st.success("Low uncertainty - High confidence prediction")
                            elif entropy < 1.0:
                                st.warning("Medium uncertainty - Moderate confidence")
                            else:
                                st.error("High uncertainty - Low confidence prediction")
                
            except Exception as e:
                st.error(f"Explainability analysis failed: {str(e)}")
            
            # Additional metrics
            st.header("📊 Detailed Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Normal Probability", f"{prob_data[0]:.1%}")
            
            with col2:
                st.metric("Diseased Probability", f"{prob_data[1]:.1%}")
            
            with col3:
                st.metric("Confidence", f"{confidence:.1%}")
            
            # Interpretation
            st.header("💡 Interpretation")
            
            if predicted_class == 0:  # Normal
                st.success("""
                **Normal Classification**: The model predicts this CT scan appears normal.
                However, this is a research demonstration and should not be used for clinical decisions.
                """)
            else:  # Diseased
                st.warning("""
                **Diseased Classification**: The model predicts this CT scan shows signs of disease.
                However, this is a research demonstration and should not be used for clinical decisions.
                """)
            
            # Technical details
            with st.expander("Technical Details"):
                st.write(f"**Model**: {model_type}")
                st.write(f"**Device**: {device}")
                st.write(f"**Input Shape**: {image_tensor.shape}")
                st.write(f"**Predicted Class**: {predicted_class}")
                st.write(f"**Confidence**: {confidence:.4f}")
                
                if "uncertainty" in explanation:
                    st.write(f"**Entropy**: {explanation['uncertainty'].get('entropy', 'N/A')}")
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.write("Please try uploading a different image.")

else:
    # Show sample images
    st.header("📸 Sample Images")
    st.write("Upload a CT scan image above, or try with these sample images:")
    
    # Create sample images (synthetic)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sample Normal CT")
        # Create a synthetic normal CT-like image
        normal_image = np.random.normal(0.5, 0.1, (224, 224))
        normal_image = np.clip(normal_image, 0, 1)
        st.image(normal_image, caption="Synthetic Normal CT", use_column_width=True)
    
    with col2:
        st.subheader("Sample Diseased CT")
        # Create a synthetic diseased CT-like image
        diseased_image = np.random.normal(0.3, 0.15, (224, 224))
        diseased_image = np.clip(diseased_image, 0, 1)
        st.image(diseased_image, caption="Synthetic Diseased CT", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Important Notes**:
- This is a research demonstration project
- Models are trained on limited/synthetic data
- No clinical validation has been performed
- Results should not be used for patient care
- Always consult qualified healthcare professionals
""")

# Add some styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 0.375rem;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)
