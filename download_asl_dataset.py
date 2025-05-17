import os
import requests
import zipfile
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# ASL dataset URLs
DATASET_URLS = {
    # ASL alphabet dataset from Kaggle - direct download link
    "asl_alphabet": "https://storage.googleapis.com/kagglesdsdata/datasets/3258/5337/asl_alphabet_train.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240516%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240516T193809Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=79e2a6b0b84ced5c5b8b7af3de0b5c4e0f91c2a0cc48bcb394ae0a7e2ae2d6e4518a59a27eee9bb3e8a54f9ced43b3eee1b87f26a47fe6ad68f7bb79be52bad4a3f7ff3bb2dd0c3d5e60682bb2de03cd3b4a7f2f49dbd7af85b5aa14c1ccb5da6d1ae1aea6e82b27e9d8da55df9f097fd0a38f7aed20f9a0682fe9193db51ba03be456f0fb00cd48e12b4d7d3ed9f3f7f34c6f5d55ad87bb0b83e4b5d33d3dfc3aba28cb8d16c59151c20a68033c8a0e96f59c59beae72c48f6debc1613c8d4fe3d841db2a3d8877d8fca30fc839da92bf00b3f3b7ebc3cc7e2ec3d7c3ceb711c8de5fa6c0f1a5cbe8a8c9cf78dc34a29e2a9ba9f",
    
    # Smaller ASL dataset option from GitHub
    "asl_alphabet_small": "https://github.com/loicmarie/sign-language-alphabet-recognizer/archive/master.zip"
}

# Constants
DATASET_DIR = "datasets"
MODEL_DIR = "models/tf_sign_model"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

def download_asl_dataset():
    """Download ASL dataset"""
    # Create directories
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Choose a dataset to download (smaller one first for testing)
    dataset_name = "asl_alphabet_small"
    dataset_url = DATASET_URLS[dataset_name]
    zip_path = os.path.join(DATASET_DIR, f"{dataset_name}.zip")
    
    # Download if not already downloaded
    if not os.path.exists(zip_path):
        print(f"Downloading {dataset_name} dataset...")
        download_file(dataset_url, zip_path)
    else:
        print(f"Dataset already downloaded: {zip_path}")
    
    # Extract if needed
    extract_dir = os.path.join(DATASET_DIR, dataset_name)
    if not os.path.exists(extract_dir):
        print(f"Extracting dataset to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        print("Extraction complete")
    else:
        print(f"Dataset already extracted: {extract_dir}")
    
    return extract_dir

def prepare_data_generators(dataset_path):
    """Prepare data generators for training"""
    # Find the actual data directory (may be nested in the extracted zip)
    if os.path.exists(os.path.join(dataset_path, "sign-language-alphabet-recognizer-master")):
        dataset_path = os.path.join(dataset_path, "sign-language-alphabet-recognizer-master", "dataset")
    
    print(f"Using dataset at: {dataset_path}")
    
    # Create data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training'
    )
    
    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation'
    )
    
    return train_generator, validation_generator

def create_and_train_model(train_generator, validation_generator):
    """Create and train the ASL recognition model"""
    # Create base model
    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(train_generator.num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    print("Training model...")
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )
    
    # Save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "asl_model")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save class indices mapping
    class_indices = train_generator.class_indices
    class_indices_reversed = {v: k for k, v in class_indices.items()}
    
    # Save class indices as a simple text file
    with open(os.path.join(MODEL_DIR, "class_indices.txt"), 'w') as f:
        for idx, class_name in class_indices_reversed.items():
            f.write(f"{idx}: {class_name}\n")
    
    print("Training complete!")
    return model, history

if __name__ == "__main__":
    print("ASL Dataset Downloader and Model Trainer")
    print("==========================================")
    
    # Download and extract dataset
    dataset_path = download_asl_dataset()
    
    # Prepare data generators
    train_generator, validation_generator = prepare_data_generators(dataset_path)
    
    # Create and train model
    model, history = create_and_train_model(train_generator, validation_generator)
    
    print("\nDone! You can now use this model with the TensorFlow app.")
    print("Run: python tf_app.py")