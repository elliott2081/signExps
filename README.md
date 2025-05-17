# Sign Language Translation App

An application that uses computer vision to translate sign language into text in real-time.

## Project Structure

- `app.py` - Main application with webcam capture and processing logic
- `custom_hand_detector.py` - Basic hand detection using OpenCV (alternative to mediapipe)
- `model_downloader.py` - Utility to download pre-trained models from Hugging Face
- `environment.yml` - Conda environment configuration
- `setup_instructions.md` - Detailed setup instructions

## Setup

1. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate sign_env
   ```

2. Test your OpenCV installation:
   ```bash
   python -c "import cv2; print(cv2.__version__)"
   ```

3. Try running the custom hand detector:
   ```bash
   python custom_hand_detector.py
   ```

## Development Approach

Since mediapipe is not compatible with M-series Macs, this project uses:

1. OpenCV for basic hand detection and webcam capture
2. TensorFlow for model inference
3. Pre-trained models from Hugging Face for sign language recognition

## Next Steps

1. Download and integrate a pre-trained sign language model
2. Improve the hand detection algorithm
3. Add continuous sign language translation
4. Implement a user interface for better interaction# signExps
