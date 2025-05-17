import os
from huggingface_hub import snapshot_download

# Function to download a model from Hugging Face
def download_model(model_id, local_dir):
    print(f"Downloading model: {model_id}")
    try:
        local_dir_path = os.path.join(os.getcwd(), local_dir)
        os.makedirs(local_dir_path, exist_ok=True)
        
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir_path,
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded successfully to {local_dir_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

if __name__ == "__main__":
    # ASL/Sign Language models on Hugging Face
    models = [
        ("Niharmahesh/Sign_language_recognition_v1", "models/sign-language-recognition")
    ]
    
    print("Downloading model: Niharmahesh/Sign_language_recognition_v1 - ASL alphabet recognition model")
    print("This may take a few minutes depending on your internet connection...")
    
    for model_id, local_dir in models:
        download_model(model_id, local_dir)