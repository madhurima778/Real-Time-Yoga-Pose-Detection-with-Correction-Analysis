import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import math

# -------------------------------
# ✔ FIXED PATH SETUP FOR VS CODE
# -------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))  # Project folder

MODEL_PATH = os.path.join(BASE, "yoga_small_model.h5")
TRAIN_NPZ = os.path.join(BASE, "train_sequences_small.npz")

STATIC_DIR = os.path.join(BASE, "static")
EXAMPLES_DIR = os.path.join(STATIC_DIR, "examples")

os.makedirs(EXAMPLES_DIR, exist_ok=True)

print("BASE DIRECTORY:", BASE)
print("MODEL EXISTS:", os.path.exists(MODEL_PATH))
print("TRAIN_NPZ EXISTS:", os.path.exists(TRAIN_NPZ))

# ----------------------------
# ✔ FLASK APP INITIALIZATION
# ----------------------------
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)
CORS(app)

# ----------------------------
# ✔ LOAD MODEL
# ----------------------------
print("\nLoading model, please wait...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ----------------------------
# ✔ LOAD TRAIN SEQUENCES
# ----------------------------
data = np.load(TRAIN_NPZ, allow_pickle=True)
X_train = data["X"]            # (N, 45, 99)
y_train = data["y"]            # class indices
classes = [
    c.decode() if isinstance(c, bytes) else c
    for c in data["classes"]
]
num_classes = len(classes)

print("Classes:", classes)

# ----------------------------
# ✔ CREATE CANONICAL SEQUENCES
# ----------------------------
canon = {}
for idx, cls in enumerate(classes):
    samples = X_train[y_train == idx]
    if len(samples) == 0:
        canon[cls] = np.zeros((45, X_train.shape[2]))
    else:
        canon[cls] = samples.mean(axis=0)  # shape (45, 99)

# ----------------------------
# ✔ SKELETON CONNECTIONS
# ----------------------------
CONNS = [
    (11,13),(13,15),(12,14),(14,16),
    (11,12),(23,24),
    (23,25),(25,27),(24,26),(26,28),
    (27,29),(29,31),(28,30),(30,32)
]

# ----------------------------
# ✔ SAVE EXAMPLE IMAGES FOR UI
# ----------------------------
def save_example_images():
    for cls in classes:
        seq = canon[cls]
        midframe = seq[22].reshape(33, 3)

        x = midframe[:, 0]
        y = -midframe[:, 1]

        plt.figure(figsize=(4, 6))
        plt.scatter(x, y, s=25)

        for a, b in CONNS:
            plt.plot([x[a], x[b]], [y[a], y[b]], linewidth=2)

        plt.title(cls)
        plt.axis("off")

        save_path = os.path.join(EXAMPLES_DIR, f"{cls}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

save_example_images()
print("Example images saved to static/examples/")

# ----------------------------
# ✔ ANGLE CALCULATION
# ----------------------------
def angle_between(A, B, C):
    A = np.array(A[:2])
    B = np.array(B[:2])
    C = np.array(C[:2])

    BA = A - B
    BC = C - B

    cosang = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-8)
    cosang = np.clip(cosang, -1.0, 1.0)

    return math.degrees(math.acos(cosang))

# JOINTS FOR CORRECTION
JOINTS = {
    "left_elbow":   (11, 13, 15),
    "right_elbow":  (12, 14, 16),
    "left_knee":    (25, 27, 29),
    "right_knee":   (26, 28, 30),
    "left_hip":     (23, 25, 27),
    "right_hip":    (24, 26, 28),
    "left_shoulder": (23, 11, 13),
    "right_shoulder": (24, 12, 14)
}

# CANONICAL ANGLES
canon_angles = {}
for cls in classes:
    seq = canon[cls].reshape(45, 33, 3)
    angs = {}

    for name, (a, b, c) in JOINTS.items():
        vals = []
        for frame in seq:
            vals.append(angle_between(frame[a], frame[b], frame[c]))
        angs[name] = float(np.mean(vals))

    canon_angles[cls] = angs


# ----------------------------
# ✔ ROUTES
# ----------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/templates")
def templates_list():
    out = []
    for cls in classes:
        out.append({
            "class": cls,
            "example": f"/static/examples/{cls}.png"
        })
    return jsonify(out)


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json()
    seq = np.array(body.get("sequence"))
    target = body.get("target", None)

    if seq.shape != (45, 99):
        return jsonify({"error": "Invalid sequence shape"}), 400

    pred = model.predict(seq.reshape(1, 45, 99))
    idx = int(np.argmax(pred[0]))
    pred_class = classes[idx]
    score = float(pred[0][idx])

    # compute user angles
    frames = seq.reshape(45, 33, 3)
    user_frame = frames.mean(axis=0)

    feedback = []

    if target in classes:
        tgt_angles = canon_angles[target]

        for name, (a, b, c) in JOINTS.items():
            user_ang = angle_between(user_frame[a], user_frame[b], user_frame[c])
            diff = user_ang - tgt_angles[name]

            if abs(diff) > 15:
                if diff > 0:
                    msg = f"{name.replace('_',' ')}: lower angle by {abs(int(diff))}°"
                else:
                    msg = f"{name.replace('_',' ')}: raise angle by {abs(int(diff))}°"

                feedback.append({
                    "joint": name,
                    "user_angle": float(user_ang),
                    "target_angle": tgt_angles[name],
                    "message": msg
                })

    return jsonify({
        "predicted_class": pred_class,
        "predicted_score": score,
        "is_target_match": (pred_class == target) if target else None,
        "feedback": feedback
    })


# ----------------------------
# ✔ RUN SERVER
# ----------------------------
if __name__ == "__main__":
    print("Starting server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)


'''
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt


python app.py
'''