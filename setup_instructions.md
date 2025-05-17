# Sign Language Translation App Setup

## Environment Setup
```bash
# Navigate to the project directory
cd sign_language_app

# Create the conda environment from the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate sign_env

# Verify installation
python -c "import cv2; import tensorflow as tf; import numpy as np; print('OpenCV version:', cv2.__version__); print('TensorFlow version:', tf.__version__); print('NumPy version:', np.__version__)"
```

## Alternative Approaches for Hand Gesture Recognition (without mediapipe)

Since mediapipe has compatibility issues with M-series Macs, consider these alternatives:

1. **Pre-trained TensorFlow/Keras models for hand gesture recognition**
   - Download models from Hugging Face or TensorFlow Hub
   - Use OpenCV for frame capture and preprocessing

2. **Transfer Learning with a CNN model**
   - Use a pre-trained model like MobileNet or ResNet
   - Fine-tune on ASL datasets like ASL Alphabet or WLASL

3. **Use specialized libraries compatible with M-series Macs**
   - HandTracking: https://github.com/EricLindCS/HandTracking
   - Or other libraries that use basic OpenCV contour detection methods

## Development Workflow

1. Start with frame capture using OpenCV
2. Implement preprocessing pipeline (frame normalization, hand region extraction)
3. Load pre-trained model for gesture recognition
4. Create processing loop for real-time translation