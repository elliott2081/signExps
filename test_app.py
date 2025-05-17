import cv2
import numpy as np
import os
import time
from custom_hand_detector import HandDetector

# Constants
WINDOW_NAME = "Sign Language Test"

def main():
    print("Starting test application...")
    
    # Initialize hand detector
    print("Initializing hand detector...")
    detector = HandDetector()
    
    # Initialize webcam
    print("Setting up webcam...")
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
    
    print("Press 'q' to quit")
    
    # Main loop
    try:
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
            
            # Process the frame with hand detector
            processed_frame, hand_region, bbox = detector.process_frame(frame)
            
            # Display detection status
            status_text = "Hand detected" if hand_region is not None else "No hand detected"
            cv2.putText(processed_frame, status_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display FPS
            cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # If hand region was detected, show it in a corner
            if hand_region is not None:
                # Resize hand region for display
                h, w = hand_region.shape[:2]
                max_size = 150
                scale = max_size / max(h, w)
                display_h, display_w = int(h * scale), int(w * scale)
                display_hand = cv2.resize(hand_region, (display_w, display_h))
                
                # Create a border around the hand display
                border_size = 2
                display_hand_with_border = cv2.copyMakeBorder(
                    display_hand, 
                    border_size, border_size, border_size, border_size, 
                    cv2.BORDER_CONSTANT, 
                    value=[0, 255, 0]
                )
                
                # Place the hand region in the corner
                corner_y, corner_x = 10, processed_frame.shape[1] - display_hand_with_border.shape[1] - 10
                
                # Create ROI in the corner for the hand region
                h_border, w_border = display_hand_with_border.shape[:2]
                processed_frame[corner_y:corner_y+h_border, corner_x:corner_x+w_border] = display_hand_with_border
            
            # Display the frame
            cv2.imshow(WINDOW_NAME, processed_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Test application closed.")

if __name__ == "__main__":
    main()