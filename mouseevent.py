import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Screen size
screen_w, screen_h = pyautogui.size()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    frame_h, frame_w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip = Landmark 8
            index_tip = hand_landmarks.landmark[12]

            # Convert normalized coordinates to webcam pixels
            x = int(index_tip.x * frame_w)
            y = int(index_tip.y * frame_h)

            # Draw a circle on index finger tip
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # Map webcam coordinates to screen coordinates
            screen_x = int(index_tip.x * screen_w)
            screen_y = int(index_tip.y * screen_h)

            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()