import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize VideoCapture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

is_hand_detected = False  # Flag to check if hand is detected

while True:
    # Read frames from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Check for hand presence
    if results.multi_hand_landmarks:
        # Get the number of hands detected
        num_hands = len(results.multi_hand_landmarks)

        if num_hands < 2:
            print("Hand Detected! Starting Pothole Detection...")
            is_hand_detected = True
        elif num_hands > 1:
            print("Hand Detected! Stopped Pothole Detection...")
            is_hand_detected = False

        for hand_landmarks in results.multi_hand_landmarks:
            # Iterate through each hand's landmarks
            for landmark in hand_landmarks.landmark:
                # Extract the pixel coordinates of the landmarks
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                # Draw a circle at each landmark point
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        if is_hand_detected:
            # Create a background subtractor
            bg_subtractor = cv2.createBackgroundSubtractorMOG2()

            # Apply background subtraction to extract moving objects
            fg_mask = bg_subtractor.apply(frame)

            # Apply morphological operations to remove noise and fill holes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours in the foreground mask
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw bounding boxes around the detected potholes
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Display the resulting frame with pothole detection
            cv2.putText(frame, "Pothole Detected: {}".format(len(contours)), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Hand and Pothole Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()

