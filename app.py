import streamlit as st

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from model_setup import get_model

# Import for the AI Enhancer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# 1. Page Config & Custom Styling
st.set_page_config(page_title="Satellite-AI: Pro Dashboard", layout="wide")
st.title("🛰️ Satellite-AI: Classifier & Enhancer")
st.markdown("---")

# 2. Optimized Loaders (Cached to save RTX 3060 VRAM)
@st.cache_resource
def load_classification_model():
    model, device = get_model(num_classes=10)
    model.load_state_dict(torch.load("land_use_model.pth", map_location=device))
    model.eval()
    return model, device

@st.cache_resource
def load_upscaler():
    # Setup architecture for the 4x upscaler
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=400, # Tiling helps your 6GB VRAM process larger images without crashing
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return upsampler

# Initialize Models
classifier, device = load_classification_model()
classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# 3. Sidebar
st.sidebar.header("Upload Center")
uploaded_file = st.sidebar.file_uploader("Upload a Satellite Patch", type=["jpg", "png", "jpeg"])
enhance_toggle = st.sidebar.checkbox("Auto-Enhance Visuals", value=True)

if uploaded_file:
    # 4. Processing Path A: Original Image for Display
    image = Image.open(uploaded_file).convert("RGB")
    
    # 5. Processing Path B: Inference (Prediction)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = classifier(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)

    # 6. UI Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Original View (64x64)")
        st.image(image, use_container_width=False, width=300)
        st.metric(label="Detected Category", value=classes[pred_idx], delta=f"{conf.item()*100:.1f}% Match")

    with col2:
        if enhance_toggle:
            st.subheader("AI Enhanced Detail (4x Zoom)")
            with st.spinner("Sharpening pixels using GAN..."):
                upsampler = load_upscaler()
                img_np = np.array(image)
                enhanced_img, _ = upsampler.enhance(img_np, outscale=4)
                st.image(enhanced_img, use_container_width=False, width=300)
        else:
            st.info("Enable 'Auto-Enhance' in sidebar to see AI reconstruction.")

    # 7. Analytics Chart
    st.markdown("---")
    st.subheader("Confidence Breakdown")
    prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
    st.bar_chart(prob_dict)

else:
    st.info("Waiting for image upload...")