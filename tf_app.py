import cv2
import numpy as np
import tensorflow as tf
import os
import time
from custom_hand_detector import HandDetector

# Define the ASL alphabet labels
ASL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
              'W', 'X', 'Y', 'Z']

# Initialize hand detector
hand_detector = HandDetector()

# Load the TensorFlow model
model_path = os.path.join('models', 'tf_sign_model', 'asl_model')
model = None
label_map = {}

print(f"Looking for TensorFlow model at: {model_path}")
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("TensorFlow model loaded successfully!")
        
        # Try to load class indices if available
        class_indices_path = os.path.join('models', 'tf_sign_model', 'class_indices.txt')
        if os.path.exists(class_indices_path):
            with open(class_indices_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(': ')
                    if len(parts) == 2:
                        idx, class_name = parts
                        label_map[int(idx)] = class_name
            print(f"Loaded {len(label_map)} class labels")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print(f"Model directory {model_path} not found. Please run download_asl_dataset.py first.")
    model = None

# Function to preprocess image for the model
def preprocess_image(hand_region):
    if hand_region is None:
        return None
    
    try:
        # Resize to expected input size (224x224 for MobileNetV2)
        resized = cv2.resize(hand_region, (224, 224))
        
        # Convert to RGB (MobileNetV2 expects RGB input)
        if len(resized.shape) == 2:  # If grayscale
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [-1, 1] range
        normalized = rgb / 127.5 - 1
        
        # Add batch dimension
        input_tensor = np.expand_dims(normalized, axis=0)
        
        return input_tensor
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Function to get prediction from model
def predict_sign(preprocessed_image):
    if preprocessed_image is None:
        return "No hand detected"
    
    if model is None:
        return "Model not loaded"
    
    try:
        # Get predictions
        predictions = model.predict(preprocessed_image, verbose=0)
        
        # Get the predicted class index
        predicted_index = np.argmax(predictions[0])
        
        # Get the confidence score
        confidence = predictions[0][predicted_index]
        
        # Get the class name from label map if available
        if label_map and predicted_index in label_map:
            class_name = label_map[predicted_index]
            print(f"Predicted class: {class_name}, Index: {predicted_index}, Confidence: {confidence:.2f}")
        else:
            class_name = ASL_LABELS[predicted_index] if predicted_index < len(ASL_LABELS) else f"Unknown-{predicted_index}"
            print(f"Predicted index: {predicted_index}, Confidence: {confidence:.2f}")
        
        # Only return prediction if confidence is high enough
        if confidence > 0.6:  # Threshold can be adjusted
            return f"{class_name} ({confidence:.2f})"
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
    
    # Set frame dimensions
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
            # Display hand region in corner
            h, w = hand_region.shape[:2]
            max_dim = 150
            scale = max_dim / max(h, w)
            display_h, display_w = int(h * scale), int(w * scale)
            display_hand = cv2.resize(hand_region, (display_w, display_h))
            
            # Add border
            display_hand_with_border = cv2.copyMakeBorder(
                display_hand, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 
                value=[0, 255, 0]
            )
            
            # Position in top-right corner
            corner_y, corner_x = 20, frame.shape[1] - display_hand_with_border.shape[1] - 20
            
            # Overlay hand region
            h_border, w_border = display_hand_with_border.shape[:2]
            frame[corner_y:corner_y+h_border, corner_x:corner_x+w_border] = display_hand_with_border
            
            # Preprocess for model
            preprocessed = preprocess_image(hand_region)
            
            # Get prediction
            if model is not None:
                current_prediction = predict_sign(preprocessed)
            else:
                current_prediction = "No model loaded"
            
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
        
        # Create space and delete buttons
        btn_height = 40
        btn_width = 100
        btn_margin = 20
        btn_y = frame.shape[0] - btn_height - btn_margin
        
        # Space button
        space_btn_x = frame.shape[1] // 2 - btn_width - btn_margin
        cv2.rectangle(frame, (space_btn_x, btn_y), (space_btn_x + btn_width, btn_y + btn_height), (0, 255, 0), 2)
        cv2.putText(frame, "SPACE", (space_btn_x + 10, btn_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Delete button
        del_btn_x = frame.shape[1] // 2 + btn_margin
        cv2.rectangle(frame, (del_btn_x, btn_y), (del_btn_x + btn_width, btn_y + btn_height), (0, 0, 255), 2)
        cv2.putText(frame, "DELETE", (del_btn_x + 10, btn_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Check if hand is over buttons
        if bbox:
            x, y, w, h = bbox
            hand_center_x = x + w//2
            hand_center_y = y + h//2
            
            # Check if hand is over space button
            if (space_btn_x <= hand_center_x <= space_btn_x + btn_width and 
                btn_y <= hand_center_y <= btn_y + btn_height):
                cv2.rectangle(frame, (space_btn_x, btn_y), (space_btn_x + btn_width, btn_y + btn_height), (0, 255, 0), -1)
                cv2.putText(frame, "SPACE", (space_btn_x + 10, btn_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                # Count frames hand is over button
                if prediction_message == "Space activated":
                    letter_stable_count += 1
                    if letter_stable_count > 10:
                        translated_text += " "
                        letter_stable_count = 0
                else:
                    prediction_message = "Space activated"
                    letter_stable_count = 0
            
            # Check if hand is over delete button
            elif (del_btn_x <= hand_center_x <= del_btn_x + btn_width and 
                  btn_y <= hand_center_y <= btn_y + btn_height):
                cv2.rectangle(frame, (del_btn_x, btn_y), (del_btn_x + btn_width, btn_y + btn_height), (0, 0, 255), -1)
                cv2.putText(frame, "DELETE", (del_btn_x + 10, btn_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # Count frames hand is over button
                if prediction_message == "Delete activated":
                    letter_stable_count += 1
                    if letter_stable_count > 10 and translated_text:
                        translated_text = translated_text[:-1]
                        letter_stable_count = 0
                else:
                    prediction_message = "Delete activated"
                    letter_stable_count = 0
        
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