import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from huggingface_hub import snapshot_download

def download_model_from_huggingface():
    """Try to download a sign language model from HuggingFace."""
    print("Attempting to download model from HuggingFace...")
    
    try:
        # Specify a model to download
        repo_id = "sign-language-tf/mobilenetv2-asl"  # This is a placeholder - replace with actual repo
        local_dir = "models/tf_sign_model"
        
        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Try to download
        print(f"Downloading model from {repo_id}...")
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        print(f"Model downloaded to {local_dir}")
        return True
    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        return False

def create_asl_model():
    """Create a simple ASL recognition model based on MobileNetV2."""
    print("Creating a simple ASL recognition model...")
    
    # Create directory
    model_dir = "models/tf_sign_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a base model from MobileNetV2
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create ASL classification model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(26, activation='softmax')  # 26 classes for ASL alphabet
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the model
    model_path = os.path.join(model_dir, "asl_model")
    model.save(model_path)
    print(f"Model created and saved to {model_path}")
    return True

if __name__ == "__main__":
    print("Sign Language TensorFlow Model Setup")
    print("===================================")
    
    # First try to download a pre-trained model
    success = download_model_from_huggingface()
    
    # If downloading fails, create a simple model
    if not success:
        print("\nFalling back to creating a simple model...")
        create_asl_model()
        
    print("\nDone! You can now use this model with the TensorFlow app.")