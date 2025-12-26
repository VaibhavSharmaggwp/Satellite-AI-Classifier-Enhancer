🛰️ Satellite-AI: Land-Use Classifier & Enhancer

## Project Overview

This project presents an advanced deep learning application for satellite image analysis, combining precise land-use classification with AI-driven image enhancement. Developed with PyTorch and deployed via Streamlit, this system demonstrates a robust pipeline for handling real-world satellite data, from raw imagery to insightful, high-resolution visual interpretations.

The core innovation lies in its dual-model approach: leveraging a highly optimized **ResNet-18 classifier** for accurate land-use categorization and integrating a **Real-ESRGAN Generative Adversarial Network (GAN)** for super-resolution, transforming low-fidelity satellite patches into visually stunning, high-definition images. This dual functionality is critical for both automated analysis and human-in-the-loop validation of satellite data.

## 🚀 Key Features

* **Accurate Land-Use Classification**:
    * Utilizes a fine-tuned **ResNet-18** neural network, pre-trained on ImageNet and further trained on the EuroSAT dataset.
    * Categorizes satellite imagery into 10 distinct land-use classes (e.g., Residential, Forest, River, Annual Crop) with high confidence.
    * **Performance**: Achieved significant loss reduction during training (from ~0.50 to ~0.20 over 5 epochs), indicating robust learning.

* **AI-Driven Image Enhancement (Super-Resolution)**:
    * Integrates **Real-ESRGAN**, a state-of-the-art Generative Adversarial Network, for 4x super-resolution.
    * Transforms pixelated 64x64 input images into clear, detailed 256x256 outputs, making complex features discernible.
    * **Bridging the Gap**: This feature enhances human interpretability and validation of AI predictions, crucial for practical applications.

* **Interactive Streamlit Dashboard**:
    * Provides a user-friendly web interface for uploading satellite images (JPG, PNG, JPEG).
    * Displays side-by-side comparisons of the original and AI-enhanced images.
    * Presents classification results, confidence scores, and a detailed probability breakdown across all classes.
    * Designed for intuitive interaction and quick visual feedback.

* **Optimized for GPU Performance**:
    * Engineered for efficient execution on **NVIDIA RTX 3060 (6GB VRAM)**, utilizing CUDA.
    * Employs **Automatic Mixed Precision (AMP)** for faster training and reduced memory footprint.
    * Incorporates **Tiling** strategies in Real-ESRGAN to handle larger images within VRAM constraints, ensuring stability.
    * Models are cached (`@st.cache_resource`) to prevent redundant loading and ensure fluid user experience.

## 📊 Demo & Screenshot

Witness the dual power of classification and enhancement in action:



## 🛠️ Tech Stack

* **Programming Language**: Python 3.x
* **Deep Learning Framework**: PyTorch
* **Computer Vision**: Torchvision, Segmentation Models PyTorch (for initial U-Net exploration)
* **Image Enhancement**: Real-ESRGAN (via `realesrgan` and `basicsr` libraries)
* **Web Framework**: Streamlit
* **Hardware Optimization**: NVIDIA CUDA, AMP, `torch.no_grad()`
* **Data Visualization**: Matplotlib

## 📦 Installation & Setup

To run this project locally, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/VaibhavSharmaggwp/Satellite-AI-Classifier-Enhancer.git](https://github.com/VaibhavSharmaggwp/Satellite-AI-Classifier-Enhancer.git)
    cd Satellite-AI-Classifier-Enhancer
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: This includes `torch`, `torchvision`, `streamlit`, `realesrgan`, `basicsr`, etc.)*

4.  **Fix `basicsr` Dependency (if `ModuleNotFoundError` occurs)**:
    If you encounter `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`, you may need to patch the `basicsr` library due to `torchvision` updates.
    * **Manual Fix**: Navigate to `.\venv\Lib\site-packages\basicsr\data\degradations.py` and change `from torchvision.transforms.functional_tensor import rgb_to_grayscale` to `from torchvision.transforms.functional import rgb_to_grayscale` on Line 8.
    * **Alternative**: You can try `pip install basicsr-fixed` after uninstalling the original `basicsr`.

5.  **Download EuroSAT Dataset**:
    Run the data setup script (this will download the dataset to the `./data` folder):
    ```bash
    python data_loader.py
    ```

6.  **Run the Streamlit Application**:
    ```bash
    streamlit run app.py
    ```
    Your browser will open to the Streamlit dashboard.

