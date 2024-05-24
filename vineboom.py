import cv2
import mediapipe as mp
import pygame

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame mixer
pygame.mixer.init()
sound = pygame.mixer.Sound('vine-boom.mp3')

# Function to check if the hand gesture is a middle finger
def is_middle_finger_up(hand_landmarks):
    # Coordinates for the tips of the thumb, index, middle, ring, and pinky fingers
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    
    difference = pinky_tip - middle_tip 
    # Ensure middle finger is the highest
    if ( difference > 0.15):
        return True
    return False

# Initialize Video Capture
cap = cv2.VideoCapture(0)
sound_played = False  # Flag to check if the sound has been played

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(image)
    
    # Convert back to BGR for rendering using OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        gesture_detected = False  # Flag to check if the gesture is detected in the current frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Check if the middle finger is up
            if is_middle_finger_up(hand_landmarks):
                gesture_detected = True
                if not sound_played:
                    sound.play()
                    sound_played = True
    
        # Reset sound_played if the gesture is no longer detected
        if not gesture_detected:
            sound_played = False
    
    # Display the image
    cv2.imshow('Hand Sign Recognition', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
