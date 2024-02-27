import cv2
import mediapipe as mp

def detect_thumbs_up(image=None):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    thumbs_up_detected = False  
    
    if image is not None:
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                try:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y:
                        thumbs_up_detected = True
                        break  
                except Exception as e:
                    pass
    else:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    try:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y:
                            thumbs_up_detected = True
                            break  
                    except Exception as e:
                        pass
            
            cv2.imshow('Thumbs Up Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or thumbs_up_detected:
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    return thumbs_up_detected

# Example usage with image
image_path = '2.jpg'  # Replace with the path to your image file
image = cv2.imread(image_path)
thumbs_up_detected = detect_thumbs_up(image)
if thumbs_up_detected:
    print("Thumbs up detected in the image!")
else:
    print("Thumbs up not detected in the image.")

# Example usage with live camera feed
thumbs_up_detected = detect_thumbs_up()
if thumbs_up_detected:
    print("Thumbs up detected in the live feed!")
else:
    print("Thumbs up not detected in the live feed.")
