import cv2
import mediapipe as mp
import os
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

dataset_path = "dataset"

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

X = []
y = []

print("Loading Dataset...")

per_student_count = {}

for student in sorted(os.listdir(dataset_path)):

    student_folder = os.path.join(
        dataset_path,
        student
    )

    if not os.path.isdir(student_folder):
        continue

    per_student_count[student] = 0

    for image_name in sorted(os.listdir(student_folder)):

        image_path = os.path.join(
            student_folder,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            landmarks = results.multi_face_landmarks[0]

            features = []

            for lm in landmarks.landmark:

                features.append(lm.x)
                features.append(lm.y)

            X.append(features)
            y.append(student)
            per_student_count[student] += 1

            print(
                f"Loaded -> {student}/{image_name}"
            )

print("\nTotal Samples =", len(X))

if len(X) == 0:

    print("No training data found")
    exit()

print("\nSamples per student:")
for student, n in per_student_count.items():
    print(f"  {student}: {n}")
    if n < 10:
        print(f"    Warning: only {n} samples - recognition for "
              f"{student} may be unreliable. Aim for 15+.")

X = np.array(X)
y = np.array(y)

model = KNeighborsClassifier(
    n_neighbors=1
)

model.fit(X, y)

# ==========================
# CALIBRATE MATCH THRESHOLD
# ==========================
# A fixed/guessed threshold (e.g. 7.0) does not generalize: it depends on
# how many landmarks you use and how varied your dataset is. Instead we
# measure, on the training data itself:
#   - intra-class distances: how far apart photos of the SAME person are
#   - inter-class distances: how close the NEAREST different person is
# and pick a threshold that separates them. This makes "unknown face"
# rejection actually data-driven instead of a guess.

print("\nCalibrating match threshold...")

# Ask each sample for its 2 nearest neighbors (itself + next closest),
# since with n_neighbors=1 a query against the training set would
# otherwise just match itself with distance 0.
calib_model = KNeighborsClassifier(n_neighbors=2)
calib_model.fit(X, y)

distances, indices = calib_model.kneighbors(X)

intra_distances = []  # same person, different photo
inter_distances = []  # nearest different person

for i in range(len(X)):
    # indices[i][0] is always i itself (distance 0); the real neighbor is [1]
    neighbor_idx = indices[i][1]
    neighbor_dist = distances[i][1]

    if y[neighbor_idx] == y[i]:
        intra_distances.append(neighbor_dist)
    else:
        inter_distances.append(neighbor_dist)

if len(intra_distances) > 0:
    max_intra = float(np.max(intra_distances))
    mean_intra = float(np.mean(intra_distances))
else:
    # Only happens if every student has exactly 1 image - no same-person
    # pair exists to measure against. Fall back to a conservative value.
    max_intra = 0.05
    mean_intra = 0.05
    print("  Warning: could not measure intra-class distance (each "
          "student needs 2+ images for proper calibration).")

if len(inter_distances) > 0:
    min_inter = float(np.min(inter_distances))
else:
    # Only one student in the dataset - no other class to compare against.
    min_inter = max_intra * 3
    print("  Warning: only one student in dataset - inter-class "
          "distance could not be measured.")

print(f"  Max distance between same person's photos:  {max_intra:.4f}")
print(f"  Min distance between different people:       {min_inter:.4f}")

if min_inter > max_intra:
    # Healthy case: there's a clear gap between same-person and
    # different-person distances. Sit the threshold in the middle of it.
    threshold = (max_intra + min_inter) / 2
else:
    # Same-person and different-person distances overlap, meaning the
    # dataset itself is ambiguous (e.g. very similar-looking people, or
    # poor quality images). Lean toward the stricter (same-person) side
    # with a small margin rather than the midpoint, to reduce false
    # accepts, and warn the user.
    threshold = max_intra * 1.2
    print(
        "  Warning: same-person and different-person distances overlap. "
        "Recognition may be less reliable - consider adding more/better "
        "quality images per student."
    )

print(f"\nCalibrated threshold = {threshold:.4f}")

os.makedirs("models", exist_ok=True)

# Save the model AND the calibrated threshold together so the attendance
# script never has to guess or hardcode a separate constant.
model_bundle = {
    "model": model,
    "threshold": threshold,
}

joblib.dump(
    model_bundle,
    "models/model.pkl"
)

print("\nModel Trained Successfully")
print(f"Saved to models/model.pkl (includes calibrated threshold)")