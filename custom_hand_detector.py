import cv2
import numpy as np

class HandDetector:
    """
    A class for detecting hands in images using OpenCV.
    This is a basic implementation without relying on mediapipe.
    """
    
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        
    def detect_skin(self, frame):
        """
        Detect skin color in HSV color space.
        This is a basic approach - you'll need to adjust for your lighting conditions.
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color in HSV
        # These values are approximate and may need adjustment
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create a mask for skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Blur the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def find_contours(self, mask):
        """Find contours in the mask and identify the largest as the hand."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None, None
        
        # Find the largest contour by area
        max_contour = max(contours, key=cv2.contourArea)
        
        # Only proceed if contour area is large enough
        if cv2.contourArea(max_contour) < 1000:  # Minimum area threshold
            return None, None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(max_contour)
        
        return max_contour, (x, y, w, h)
    
    def extract_hand_region(self, frame, bbox, padding=20):
        """Extract the hand region from the frame using the bounding box."""
        if bbox is None:
            return None
        
        x, y, w, h = bbox
        
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        # Extract the hand region
        hand_region = frame[y:y+h, x:x+w]
        
        return hand_region
    
    def process_frame(self, frame):
        """Process a frame to detect and extract hand regions."""
        # Copy frame to avoid modifying original
        result_frame = frame.copy()
        
        # Detect skin
        skin_mask = self.detect_skin(frame)
        
        # Find hand contour and bounding box
        contour, bbox = self.find_contours(skin_mask)
        
        if contour is not None and bbox is not None:
            # Draw contour on result frame
            cv2.drawContours(result_frame, [contour], 0, (0, 255, 0), 2)
            
            # Draw bounding box
            x, y, w, h = bbox
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Extract hand region
            hand_region = self.extract_hand_region(frame, bbox)
            
            return result_frame, hand_region, bbox
        
        return result_frame, None, None


# Example usage
if __name__ == "__main__":
    # Initialize the hand detector
    detector = HandDetector()
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Process the frame
        result, hand_region, bbox = detector.process_frame(frame)
        
        # Display the hand region if detected
        if hand_region is not None:
            # Resize hand region for display
            display_hand = cv2.resize(hand_region, (200, 200))
            
            # Create ROI in the corner for the hand region
            h, w = display_hand.shape[:2]
            result[10:10+h, 10:10+w] = display_hand
        
        # Display the result
        cv2.imshow('Hand Detection', result)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()