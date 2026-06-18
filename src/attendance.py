import cv2
import mediapipe as mp
import mysql.connector
import joblib
import os
from datetime import datetime, timedelta
import time

# --------------------
# Load face recognition model
# --------------------
# train_model.py now saves a bundle: {"model": ..., "threshold": ...}
# The threshold is calibrated from the dataset itself instead of being a
# guessed constant, so it correctly rejects faces that aren't in the
# training set.
model_bundle = joblib.load("models/model.pkl")
model = model_bundle["model"]
THRESHOLD = model_bundle["threshold"]

print(f"Loaded model. Match threshold = {THRESHOLD:.4f}")

# --------------------
# Output helpers (MySQL)
# --------------------
# One row per login/logout session, stored in a MySQL database that you
# can browse with MySQL Workbench (Workbench connects to a MySQL SERVER,
# not to a file - unlike the old SQLite version, this now needs a running
# MySQL server with credentials).
#
# Credentials are read from environment variables so you don't have to
# hardcode your password into this script. Set them before running, e.g.:
#   export DB_HOST=localhost
#   export DB_USER=root
#   export DB_PASSWORD=yourpassword
#   export DB_NAME=attendance_system
#
# If a variable isn't set, the fallback values below are used - change
# DB_HOST/DB_USER/DB_PASSWORD/DB_NAME to match your own MySQL Workbench
# connection if you'd rather not use environment variables.
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "Priya@123")
DB_NAME = os.environ.get("DB_NAME", "attendance_system")


def init_attendance_db():
    """Connect to MySQL, create the database/table if they don't already
    exist, and return an open connection."""
    # First connect with no database selected, so we can CREATE DATABASE
    # if it doesn't exist yet.
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    cursor.close()
    conn.close()

    # Now reconnect directly into that database for normal use.
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            login_date DATE NOT NULL,
            login_time TIME NOT NULL,
            status VARCHAR(50) NOT NULL,
            logout_time TIME
        )
        """
    )
    conn.commit()
    cursor.close()
    return conn


def save_attendance_record(conn, user_name: str, status: str, login_time: datetime, logout_time: datetime | None):
    """Insert ONE row per user per session.

    Store login time + logout time in the SAME row, matching the
    previous one-CSV-per-session behavior.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO attendance (name, login_date, login_time, status, logout_time)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            user_name,
            login_time.strftime("%Y-%m-%d"),
            login_time.strftime("%H:%M:%S"),
            status,
            logout_time.strftime("%H:%M:%S") if logout_time else None,
        ),
    )
    conn.commit()
    cursor.close()

    print("\nAttendance Saved")
    print(
        f"Name={user_name}  LoginDate={login_time.strftime('%Y-%m-%d')}  "
        f"LoginTime={login_time.strftime('%H:%M:%S')}  Status={status}  "
        f"LogoutTime={logout_time.strftime('%H:%M:%S') if logout_time else ''}"
    )
    print(f"\nSaved to MySQL database '{DB_NAME}' on {DB_HOST}")


# --------------------
# Face: MediaPipe FaceDetection + FaceMesh
# --------------------
# IMPORTANT: collect_data.py saves CROPPED face images (just the face
# region, resized to 200x200), and train_model.py runs FaceMesh on those
# crops. FaceMesh landmark x/y values are normalized to the image it is
# given, so a face mesh run on a tight crop produces different normalized
# coordinates than the same face run on a full, uncropped webcam frame
# (the face occupies a different position/scale in the image). If we feed
# FaceMesh the full frame here, every comparison against the training
# data is comparing two different coordinate spaces -> distances are
# inflated -> nobody matches.
#
# Fix: detect + crop the face first here too, exactly like collect_data.py
# does, then run FaceMesh on that crop. This keeps train/inference
# conditions consistent.
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5,
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
)

# --------------------
# Hands: MediaPipe Hands
# --------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


def get_finger_states(hand_landmarks, hand_label):
    """Returns [thumb, index, middle, ring, pinky] as 0/1.
    Landmark rules match basic.py.
    """
    lm = hand_landmarks.landmark
    fingers = []

    # Thumb
    if hand_label == "Right":
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    else:  # Left
        fingers.append(1 if lm[4].x > lm[3].x else 0)

    # Index/Middle/Ring/Pinky
    tip_ids = [8, 12, 16, 20]
    for tip in tip_ids:
        fingers.append(1 if lm[tip].y < lm[tip - 2].y else 0)

    return fingers


def identify_face(frame, rgb):
    """Run the SAME detect -> crop -> FaceMesh -> kNN pipeline used at
    login, and return (name, distance) if a confident match is found,
    or (None, distance) / (None, None) otherwise.

    This is factored out so it can be reused both for the initial login
    match (WAIT_FACE / FACE_MATCHED) and for re-verifying identity during
    LOGGED_IN, right before allowing a logout gesture to take effect.
    """
    detection_results = face_detection.process(rgb)
    if not detection_results.detections:
        return None, None

    h, w, _ = frame.shape

    def box_area(det):
        bb = det.location_data.relative_bounding_box
        return bb.width * bb.height

    detection = max(detection_results.detections, key=box_area)
    bbox = detection.location_data.relative_bounding_box

    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    x = max(0, x)
    y = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)

    if x2 <= x or y2 <= y:
        return None, None

    crop = frame[y:y2, x:x2]
    if crop.size == 0:
        return None, None

    crop = cv2.resize(crop, (200, 200))
    face_crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # draw the box for visual feedback wherever this is called from
    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

    face_results = face_mesh.process(face_crop_rgb)
    if not face_results.multi_face_landmarks:
        return None, None

    landmarks = face_results.multi_face_landmarks[0]
    features = []
    for lm in landmarks.landmark:
        features.append(lm.x)
        features.append(lm.y)

    distances, _ = model.kneighbors([features])
    distance = distances[0][0]
    prediction = model.predict([features])[0]

    if distance < THRESHOLD:
        return prediction, distance
    return None, distance


# --------------------
# State machine
# --------------------
WAIT_FACE = "WAIT_FACE"
FACE_MATCHED = "FACE_MATCHED"
LOGGED_IN = "LOGGED_IN"  # waiting period before logout is allowed

state = WAIT_FACE

recognized_name = None
login_time = None
logout_unlock_time = None  # the moment AFTER which logout becomes allowed
logout_time = None

# Prevent accidental logout due to noisy thumb detection
thumb_only_streak = 0
REQUIRED_THUMB_ONLY_FRAMES = 8  # require thumb-only for N consecutive frames

# How long the user must wait after login before they're allowed to logout
LOGOUT_WAIT_SECONDS = 60  # 1 minute

# --------------------
# Face re-verification during LOGGED_IN / logout
# --------------------
# Running the full face pipeline (detection + mesh + kNN) on every single
# frame in addition to hand tracking is wasteful, since identity doesn't
# change frame-to-frame. Instead we only re-check identity every N frames,
# and cache the result in between checks.
FACE_RECHECK_INTERVAL_FRAMES = 10
logout_frame_counter = 0
same_person_present = False  # cached result of the last identity check

cap = cv2.VideoCapture(0)
db_conn = init_attendance_db()
print("Attendance / Login System Started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    now = datetime.now()

    # --------------------
    # FACE stage (login matching)
    # --------------------
    if state in (WAIT_FACE, FACE_MATCHED):
        match_name, distance = identify_face(frame, rgb)

        if distance is not None:
            print(f"distance={distance:.4f}  threshold={THRESHOLD:.4f}  closest={match_name}")

        if match_name is not None:
            recognized_name = match_name
            state = FACE_MATCHED
            login_time = None
            logout_time = None

            cv2.putText(
                frame,
                "Face match found",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
        else:
            recognized_name = None
            state = WAIT_FACE
            cv2.putText(
                frame,
                "Face not found",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        if state == FACE_MATCHED and recognized_name is not None:
            cv2.putText(
                frame,
                "Raise all fingers to Login",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (200, 200, 0),
                2,
            )
            cv2.putText(
                frame,
                f"User: {recognized_name}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

    # --------------------
    # HAND stage for LOGIN (all fingers)
    # --------------------
    if state == FACE_MATCHED and recognized_name is not None:
        hand_results = hands.process(rgb)
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            hand_label = hand_results.multi_handedness[0].classification[0].label

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb, index, middle, ring, pinky = get_finger_states(hand_landmarks, hand_label)
            all_fingers = (thumb and index and middle and ring and pinky)

            cv2.putText(
                frame,
                f"Thumb:{thumb} Index:{index} Middle:{middle} Ring:{ring} Pinky:{pinky}",
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            if all_fingers:
                state = LOGGED_IN
                login_time = now
                logout_unlock_time = login_time + timedelta(seconds=LOGOUT_WAIT_SECONDS)
                logout_time = None
                thumb_only_streak = 0  # reset streak counter for the new session

                # Reset re-verification cache for the new session so the
                # first logout-stage frame forces a fresh identity check
                # rather than reusing a stale cached result.
                logout_frame_counter = 0
                same_person_present = False

                cv2.putText(
                    frame,
                    "Login successfully",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    3,
                )
                print(f"{recognized_name} Login successfully")

                time.sleep(1)

    # --------------------
    # LOGOUT stage
    # Rule 1: thumb-only is REJECTED until 1 minute has passed since login.
    # Rule 2 (NEW): thumb-only is REJECTED unless the face currently in
    #   front of the camera re-matches recognized_name. This stops anyone
    #   else from logging the original user out just by showing thumbs-up
    #   after the wait period, and stops a logout firing if the user has
    #   simply walked away and someone else's hand is in frame.
    # --------------------
    if state == LOGGED_IN and logout_unlock_time is not None and login_time is not None:
        can_logout_now = now >= logout_unlock_time

        # --- Throttled face re-verification ---
        # Only run the full face pipeline every FACE_RECHECK_INTERVAL_FRAMES
        # frames; reuse the cached result on the frames in between. This
        # keeps the loop responsive (hand tracking still runs every frame)
        # while still catching an identity swap within ~a few hundred ms.
        if logout_frame_counter % FACE_RECHECK_INTERVAL_FRAMES == 0:
            current_name, _ = identify_face(frame, rgb)
            same_person_present = (current_name == recognized_name)
        logout_frame_counter += 1

        if not can_logout_now:
            remaining_seconds = int((logout_unlock_time - now).total_seconds())
            if remaining_seconds < 0:
                remaining_seconds = 0
            mm = remaining_seconds // 60
            ss = remaining_seconds % 60

            cv2.putText(
                frame,
                f"Wait to logout: 00:{mm:02d}:{ss:02d}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
        elif not same_person_present:
            cv2.putText(
                frame,
                "Verify your face to logout",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Now you can logout",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        # If the present person isn't verified as recognized_name, don't
        # let a streak build up at all, regardless of hand gesture.
        if not same_person_present:
            thumb_only_streak = 0

        hand_results = hands.process(rgb)
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            hand_label = hand_results.multi_handedness[0].classification[0].label

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb, index, middle, ring, pinky = get_finger_states(hand_landmarks, hand_label)
            thumb_only = (thumb == 1 and index == 0 and middle == 0 and ring == 0 and pinky == 0)

            cv2.putText(
                frame,
                f"Thumb-only: {thumb_only}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            if thumb_only and not can_logout_now:
                # User tried to logout too early
                cv2.putText(
                    frame,
                    "Can't logout now",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )
                thumb_only_streak = 0  # don't let early attempts build up a streak

            elif thumb_only and can_logout_now and not same_person_present:
                # Right gesture, right time, WRONG / no confirmed face.
                cv2.putText(
                    frame,
                    "Face mismatch - can't logout",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3,
                )
                thumb_only_streak = 0

            elif thumb_only and can_logout_now and same_person_present:
                thumb_only_streak += 1
                if thumb_only_streak < REQUIRED_THUMB_ONLY_FRAMES:
                    # keep waiting until thumb-only is stable
                    cv2.putText(
                        frame,
                        "Hold thumb steady to logout...",
                        (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                else:
                    logout_time = now

                    cv2.putText(
                        frame,
                        "Logout successfully",
                        (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        3,
                    )
                    print(f"{recognized_name} Logout successfully")

                    # Save login + logout BOTH in one row in the database
                    save_attendance_record(
                        db_conn,
                        recognized_name,
                        "PresentLogout",
                        login_time,
                        logout_time,
                    )

                    time.sleep(1)
                    break
            else:
                # Hand present but not thumb-only -> reset streak
                thumb_only_streak = 0

    cv2.imshow("Attendance System", frame)

    # manual exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
db_conn.close()