import cv2
import numpy as np
import time
from custom_hand_detector import HandDetector

def main():
    # Initialize webcam and hand detector
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    # Set up display settings
    window_name = "Sign Language Detector (Simple Version)"
    frame_history = []
    history_size = 10
    
    # Create buffer for translated text
    translated_text = ""
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 0)
    thickness = 2
    
    print("Press 'q' to quit, 'c' to clear text")
    
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Mirror the frame for more intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Process frame with hand detector
        processed_frame, hand_region, bbox = detector.process_frame(frame)
        
        # If hand detected, show it in a box in corner
        if hand_region is not None:
            # Store frame in history
            if len(frame_history) >= history_size:
                frame_history.pop(0)
            frame_history.append(frame)
            
            # Resize hand region for display
            display_size = 150
            h, w = hand_region.shape[:2]
            aspect_ratio = w/h
            
            if w > h:
                display_w = display_size
                display_h = int(display_size / aspect_ratio)
            else:
                display_h = display_size
                display_w = int(display_size * aspect_ratio)
                
            display_hand = cv2.resize(hand_region, (display_w, display_h))
            
            # Add border to hand display
            display_hand_with_border = cv2.copyMakeBorder(
                display_hand, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0, 255, 0]
            )
            
            # Position in top-right corner
            y_offset = 20
            x_offset = frame.shape[1] - display_hand_with_border.shape[1] - 20
            
            # Place on frame
            h_border, w_border = display_hand_with_border.shape[:2]
            frame[y_offset:y_offset+h_border, x_offset:x_offset+w_border] = display_hand_with_border
            
            # Draw instruction prompt
            cv2.putText(frame, "Make sign with your hand", (20, 40), font, font_scale, color, thickness)
            
            # Draw hint about capturing
            cv2.putText(frame, "Hold still to capture sign", (20, 80), font, font_scale, color, thickness)
            
            # Add "space" and "delete" virtual buttons
            btn_height = 40
            btn_width = 100
            btn_margin = 20
            btn_y = frame.shape[0] - btn_height - btn_margin
            
            # Space button
            space_btn_x = frame.shape[1] // 2 - btn_width - btn_margin
            cv2.rectangle(frame, (space_btn_x, btn_y), (space_btn_x + btn_width, btn_y + btn_height), (0, 255, 0), 2)
            cv2.putText(frame, "SPACE", (space_btn_x + 10, btn_y + 25), font, 0.6, (0, 255, 0), 2)
            
            # Delete button
            del_btn_x = frame.shape[1] // 2 + btn_margin
            cv2.rectangle(frame, (del_btn_x, btn_y), (del_btn_x + btn_width, btn_y + btn_height), (0, 0, 255), 2)
            cv2.putText(frame, "DELETE", (del_btn_x + 10, btn_y + 25), font, 0.6, (0, 0, 255), 2)
            
            # Check if hand is over buttons
            if bbox:
                x, y, w, h = bbox
                hand_center_x = x + w//2
                hand_center_y = y + h//2
                
                # Check if hand is over space button
                if (space_btn_x <= hand_center_x <= space_btn_x + btn_width and 
                    btn_y <= hand_center_y <= btn_y + btn_height):
                    cv2.rectangle(frame, (space_btn_x, btn_y), (space_btn_x + btn_width, btn_y + btn_height), (0, 255, 0), -1)
                    cv2.putText(frame, "SPACE", (space_btn_x + 10, btn_y + 25), font, 0.6, (0, 0, 0), 2)
                    # Add space to text if we hold over button
                    if len(frame_history) >= history_size - 2:  # Need to hold for a bit
                        translated_text += " "
                        frame_history = []  # Reset history to avoid multiple spaces
                
                # Check if hand is over delete button
                elif (del_btn_x <= hand_center_x <= del_btn_x + btn_width and 
                      btn_y <= hand_center_y <= btn_y + btn_height):
                    cv2.rectangle(frame, (del_btn_x, btn_y), (del_btn_x + btn_width, btn_y + btn_height), (0, 0, 255), -1)
                    cv2.putText(frame, "DELETE", (del_btn_x + 10, btn_y + 25), font, 0.6, (255, 255, 255), 2)
                    # Delete last character if we hold over button
                    if len(frame_history) >= history_size - 2 and len(translated_text) > 0:  # Need to hold for a bit
                        translated_text = translated_text[:-1]
                        frame_history = []  # Reset history to avoid multiple deletions
        
        else:
            # Clear frame history when no hand is detected
            frame_history = []
            cv2.putText(frame, "No hand detected", (20, 40), font, font_scale, (0, 0, 255), thickness)
        
        # Create text display area at the bottom
        text_area_height = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0] - text_area_height), 
                     (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Display the translated text
        cv2.putText(frame, "Text:", (20, frame.shape[0] - text_area_height + 30), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # Split text into lines if too long
        max_chars_per_line = 50
        if len(translated_text) > max_chars_per_line:
            lines = [translated_text[i:i+max_chars_per_line] 
                    for i in range(0, len(translated_text), max_chars_per_line)]
        else:
            lines = [translated_text]
        
        for i, line in enumerate(lines):
            y_pos = frame.shape[0] - text_area_height + 60 + i * 30
            cv2.putText(frame, line, (20, y_pos), font, font_scale, (255, 255, 255), thickness)
        
        # Show instructions
        cv2.putText(frame, "Press 'q' to quit, 'c' to clear text", 
                   (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the resulting frame
        cv2.imshow(window_name, frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            translated_text = ""
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()