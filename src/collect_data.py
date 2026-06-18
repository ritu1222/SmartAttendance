import cv2
import mediapipe as mp
import os

# ==========================
# ENTER STUDENT NAME
# ==========================

student_name = input(
    "Enter Student Name (Example: st1, st2, st3): "
).strip()

if not student_name:
    print("Student name cannot be empty")
    exit()

dataset_path = "dataset"

os.makedirs(dataset_path, exist_ok=True)

save_path = os.path.join(dataset_path, student_name)

os.makedirs(save_path, exist_ok=True)

# Count existing images
existing_images = [
    file for file in os.listdir(save_path)
    if file.endswith(".jpg")
]

count = len(existing_images)

print(f"\nCurrent Student: {student_name}")
print(f"Existing Images: {count}")

# ==========================
# MEDIAPIPE FACE DETECTION
# ==========================

mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

# ==========================
# WEBCAM
# ==========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("\nInstructions")
print("Press T = Capture Face")
print("Press Q = Quit\n")
print("Tip: capture 20-30+ images per student, varying angle,")
print("distance, and expression slightly for a more robust model.\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb)

    face_crop = None
    multiple_faces = False

    if results.detections:

        h, w, _ = frame.shape

        if len(results.detections) > 1:
            multiple_faces = True

        # Pick the largest detected face (closest to camera) rather than
        # just the first one MediaPipe happens to return.
        def box_area(det):
            bb = det.location_data.relative_bounding_box
            return bb.width * bb.height

        detection = max(results.detections, key=box_area)

        bbox = detection.location_data.relative_bounding_box

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        x = max(0, x)
        y = max(0, y)

        x2 = min(w, x + bw)
        y2 = min(h, y + bh)

        face_crop = frame[y:y2, x:x2]

        box_color = (0, 0, 255) if multiple_faces else (0, 255, 0)

        cv2.rectangle(
            frame,
            (x, y),
            (x2, y2),
            box_color,
            2
        )

    cv2.putText(
        frame,
        f"Student: {student_name}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Saved: {count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    if multiple_faces:
        cv2.putText(
            frame,
            "Multiple faces detected - only one person at a time!",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow("Dataset Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # ==========================
    # SAVE IMAGE
    # ==========================

    if key == ord('t'):

        if face_crop is None or face_crop.size == 0:

            print("No face detected")

        elif multiple_faces:

            print("Multiple faces in frame - capture skipped. "
                  "Make sure only the target student is visible.")

        else:

            count += 1

            face_crop = cv2.resize(
                face_crop,
                (200, 200)
            )

            file_name = os.path.join(
                save_path,
                f"{count}.jpg"
            )

            cv2.imwrite(
                file_name,
                face_crop
            )

            print(f"Saved: {file_name}")

    # ==========================
    # QUIT
    # ==========================

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nDataset Collection Complete")
print(f"Student: {student_name}")
print(f"Total Images: {count}")

if count < 15:
    print(
        "\nWarning: fewer than 15 images saved. "
        "Consider capturing more for reliable recognition."
    )