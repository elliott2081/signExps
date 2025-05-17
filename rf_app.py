import cv2
import numpy as np
import os
import time
import pickle
from custom_hand_detector import HandDetector

# Define the ASL alphabet labels
ASL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
              'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']

# Initialize hand detector
hand_detector = HandDetector()

# Load the data from the pkl file
model_path = os.path.join('models', 'sign-language-recognition', 'best_random_forest_model.pkl')
model_data = None

print(f"Looking for model at: {model_path}")
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print("Model data loaded successfully!")
        print(f"Type of loaded data: {type(model_data)}")
        if isinstance(model_data, np.ndarray):
            print(f"Shape of array: {model_data.shape}")
            # Basic heuristic to determine if it's likely a set of precomputed features
            if len(model_data.shape) == 2:
                print("Loaded data appears to be reference features or embeddings")
    except Exception as e:
        print(f"Error loading model: {e}")
        model_data = None
else:
    print(f"Model file not found at {model_path}")

# Function to extract hand features
def extract_features(hand_region):
    if hand_region is None:
        return None
    
    # Resize to a standard size
    resized = cv2.resize(hand_region, (64, 64))
    
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values
    normalized = gray / 255.0
    
    # Flatten the image to a 1D array of features
    features = normalized.flatten()
    
    return features

# Function to get prediction from model
def predict_sign(features):
    if features is None:
        return "No hand detected"
    
    if model_data is None:
        return "Model not loaded"
    
    try:
        # Since the model_data seems to be a numpy array rather than a model,
        # we'll use a nearest neighbor approach to make predictions
        
        # Reshape if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        # If the model_data is a 2D array where rows correspond to classes
        # and columns to features, we'll find the closest match by using
        # Euclidean distance
        
        # Assume first 26 rows are reference features for ASL letters
        num_classes = min(len(ASL_LABELS), model_data.shape[0])
        
        # Calculate distance to each reference feature
        distances = []
        for i in range(num_classes):
            # Check if the reference feature is a string
            if isinstance(model_data[i], str):
                print(f"Class {i} ({ASL_LABELS[i] if i < len(ASL_LABELS) else 'Unknown'}) is a string: {model_data[i]}")
                # Just use a random distance for strings
                distance = 0.5
            elif isinstance(model_data[i], (np.ndarray, list)):
                # Convert to numpy array if it's a list
                if isinstance(model_data[i], list):
                    ref_feature = np.array(model_data[i])
                else:
                    ref_feature = model_data[i]
                
                # Try to calculate distance
                try:
                    # If shapes match
                    if ref_feature.shape == features.shape:
                        distance = np.linalg.norm(ref_feature - features)
                    else:
                        # If shapes don't match, flatten and trim to common length
                        ref_flat = ref_feature.flatten()
                        feat_flat = features.flatten()
                        min_len = min(ref_flat.shape[0], feat_flat.shape[0])
                        distance = np.linalg.norm(ref_flat[:min_len] - feat_flat[:min_len])
                except Exception as e:
                    print(f"Error calculating distance for class {i}: {e}")
                    distance = 1.0  # Default distance on error
            else:
                print(f"Class {i} has unknown type: {type(model_data[i])}")
                distance = 1.0  # Default distance for unknown types
            
            distances.append(distance)
        
        # Get the index of the minimum distance
        predicted_index = np.argmin(distances)
        
        # Calculate a confidence score based on distance (smaller distance = higher confidence)
        max_distance = max(distances)
        min_distance = min(distances)
        distance_range = max_distance - min_distance if max_distance > min_distance else 1
        
        # Normalize to [0, 1] and invert (1 = highest confidence)
        confidence = 1 - (distances[predicted_index] - min_distance) / distance_range
        
        print(f"Predicted index: {predicted_index}, Confidence: {confidence:.2f}")
        
        # Only return prediction if confidence is high enough
        if confidence > 0.4:  # Lower threshold for testing
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
        
        # Run hand detection
        processed_frame, hand_region, bbox = hand_detector.process_frame(frame)
        
        # If hand detected, try to recognize sign
        if hand_region is not None:
            # Extract features for the model
            features = extract_features(hand_region)
            
            # Get prediction
            current_prediction = predict_sign(features)
            
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