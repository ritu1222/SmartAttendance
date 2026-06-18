import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    finger_count = 0

    if results.multi_hand_landmarks:

        total_hands = len(results.multi_hand_landmarks)

        # More than one hand detected
        if total_hands > 1:

            cv2.putText(
                frame,
                "PUT ONLY ONE HAND",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        else:
            # One hand detected
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            hand_label = results.multi_handedness[0].classification[0].label

            lm = hand_landmarks.landmark

            fingers = []

            # Thumb
            if hand_label == "Right":
                if lm[4].x < lm[3].x:
                    fingers.append(1)
                else:
                    fingers.append(0)

            else:  # Left Hand
                if lm[4].x > lm[3].x:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Index, Middle, Ring, Pinky
            tip_ids = [8, 12, 16, 20]

            for tip in tip_ids:
                if lm[tip].y < lm[tip - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            finger_count = sum(fingers)

            cv2.putText(
                frame,
                f"Hand: {hand_label}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"Fingers: {finger_count}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("One Hand Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()