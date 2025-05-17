import cv2
import numpy as np
import tensorflow as tf
import time
import os
from custom_hand_detector import HandDetector

# Define the ASL alphabet labels
ASL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
              'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']

# Initialize hand detector
hand_detector = HandDetector()

# Load the pre-trained model
model_path = 'models/sign-language-recognition'
model = None

print(f"Looking for model in {model_path}")
print(f"Directory exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    # List files in the model directory to debug
    print("Files in model directory:")
    for root, dirs, files in os.walk(model_path):
        for file in files:
            print(f"  {os.path.join(root, file)}")
        for dir in dirs:
            print(f"  {os.path.join(root, dir)} (directory)")
    
    # Try different approaches to load the model
    try:
        # Try loading from top directory
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model from top directory: {e}")
        
        # Try loading with TF saved_model format
        saved_model_path = os.path.join(model_path, 'saved_model')
        if os.path.exists(saved_model_path):
            try:
                model = tf.keras.models.load_model(saved_model_path)
                print(f"Model loaded successfully from {saved_model_path}")
            except Exception as e2:
                print(f"Error loading from saved_model directory: {e2}")
                
                # Try all subdirectories
                for subdir in os.listdir(model_path):
                    subdir_path = os.path.join(model_path, subdir)
                    if os.path.isdir(subdir_path):
                        try:
                            model = tf.keras.models.load_model(subdir_path)
                            print(f"Model loaded successfully from {subdir_path}")
                            break
                        except Exception:
                            pass
        
        # If still no model, try loading serialized model components
        if model is None:
            try:
                # Check for model.json (architecture) and model.h5 (weights)
                model_json_path = None
                model_weights_path = None
                
                for root, _, files in os.walk(model_path):
                    for file in files:
                        if file.endswith('.json') and 'model' in file.lower():
                            model_json_path = os.path.join(root, file)
                        elif file.endswith('.h5') or file.endswith('.weights'):
                            model_weights_path = os.path.join(root, file)
                
                if model_json_path and model_weights_path:
                    # Load model from JSON
                    with open(model_json_path, 'r') as json_file:
                        loaded_model_json = json_file.read()
                    model = tf.keras.models.model_from_json(loaded_model_json)
                    
                    # Load weights
                    model.load_weights(model_weights_path)
                    print(f"Model loaded from JSON and weights files")
            except Exception as e3:
                print(f"Error loading from JSON/weights: {e3}")
    
    if model is None:
        print("Failed to load model using any method. Will run without model.")
else:
    print(f"Model directory {model_path} not found. Please run model_downloader.py first.")

# Function to preprocess frame for the model
def preprocess_frame(frame):
    if frame is None:
        return None
    
    # Process with hand detector to isolate hand region
    _, hand_region, _ = hand_detector.process_frame(frame)
    
    if hand_region is None:
        return None
    
    # Resize to expected input size (224x224 is common for many models)
    resized = cv2.resize(hand_region, (224, 224))
    
    # Convert to RGB if the model expects RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    normalized = rgb / 255.0
    
    return np.expand_dims(normalized, axis=0)

# Function to get prediction from model
def predict_sign(preprocessed_frame):
    if preprocessed_frame is None:
        return "No hand detected"
    
    if model is None:
        return "Model not loaded"
    
    try:
        # Make prediction with debugging
        print("Making prediction...")
        prediction = model.predict(preprocessed_frame, verbose=0)
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction values: {prediction}")
        
        # Get the index of the highest probability
        predicted_index = np.argmax(prediction[0])
        print(f"Predicted index: {predicted_index}")
        
        # Get the confidence score
        confidence = prediction[0][predicted_index]
        print(f"Confidence: {confidence}")
        
        # Only return prediction if confidence is high enough
        if confidence > 0.5:  # Lowered threshold for testing
            if predicted_index < len(ASL_LABELS):
                return f"{ASL_LABELS[predicted_index]} ({confidence:.2f})"
            else:
                return f"Unknown ({confidence:.2f})"
        else:
            return "Low confidence"
            
    except Exception as e:
        import traceback
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return "Prediction error"

# Main application loop
def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set frame dimensions (adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # FPS calculation variables
    prev_frame_time = 0
    new_frame_time = 0
    
    # Prediction smoothing
    prediction_history = []
    prediction_buffer_size = 5
    prediction_message = ""
    
    # Text output display
    translated_text = ""
    last_letter = ""
    letter_stable_count = 0
    space_counter = 0
    
    print("Press 'q' to quit")
    print("Press 'c' to clear the translated text")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        
        # Mirror the frame for more intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Run hand detection and sign recognition
        processed_frame, hand_region, bbox = hand_detector.process_frame(frame)
        
        # If hand detected, try to recognize sign
        if hand_region is not None:
            # Preprocess for model and predict
            preprocessed = preprocess_frame(frame)
            current_prediction = predict_sign(preprocessed)
            
            # Add prediction to history
            prediction_history.append(current_prediction)
            if len(prediction_history) > prediction_buffer_size:
                prediction_history.pop(0)
            
            # Get most common prediction from history
            if prediction_history:
                from collections import Counter
                prediction_counts = Counter(prediction_history)
                most_common = prediction_counts.most_common(1)[0]
                stable_prediction = most_common[0]
                prediction_message = stable_prediction
                
                # Extract letter from prediction (format may be "A (0.95)")
                if "(" in stable_prediction:
                    letter = stable_prediction.split(" ")[0]
                    
                    # Handle letter stability for text output
                    if letter == last_letter:
                        letter_stable_count += 1
                        
                        # If same letter is stable for some time and not already added
                        if letter_stable_count > 10 and not translated_text.endswith(letter):
                            if letter == "space":
                                translated_text += " "
                            elif letter == "del" and translated_text:
                                translated_text = translated_text[:-1]
                            elif letter != "nothing" and letter != "del":
                                translated_text += letter
                            letter_stable_count = 0
                    else:
                        letter_stable_count = 0
                        last_letter = letter
        else:
            # If no hand detected, clear prediction history and increment space counter
            prediction_history = []
            prediction_message = "No hand detected"
            space_counter += 1
            
            # Add space after period of no hand detection
            if space_counter > 30 and not translated_text.endswith(" ") and translated_text:
                translated_text += " "
                space_counter = 0
        
        # Create a transparent overlay for text display
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0]-150), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Display FPS and predicted sign
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Sign: {prediction_message}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display translated text
        cv2.putText(frame, "Translated:", (10, frame.shape[0]-100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Split translated text into lines if too long
        max_width = 60  # characters per line
        text_y = frame.shape[0] - 70
        text_lines = [translated_text[i:i+max_width] for i in range(0, len(translated_text), max_width)]
        
        for line in text_lines:
            cv2.putText(frame, line, (10, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            text_y += 30
            
        # Display instructions
        cv2.putText(frame, "Press 'q' to quit, 'c' to clear text", (10, frame.shape[0]-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('Sign Language Translator', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            translated_text = ""
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()