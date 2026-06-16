import os
import numpy as np
import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================
# CONFIG
# ==========================
IMG_SIZE = 160
BATCH_SIZE = 16

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "clasificacion_em.h5")
IMG_DIR = os.path.join(BASE_DIR, "data", "val_set", "val_set", "images")
ANN_DIR = os.path.join(BASE_DIR, "data", "val_set", "val_set", "annotations")

CLASS_NAMES = [
    "Neutral", "Happy", "Sad", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt"
]

# ==========================
# KERAS COMPATIBILITY
# ==========================
class CompatibleDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

# ==========================
# PATHS
# ==========================
def get_image_label_paths(img_dir, ann_dir):
    img_paths = []
    labels = []

    for img_path in glob(os.path.join(img_dir, "*.jpg")):
        img_name = os.path.basename(img_path).replace(".jpg", "")
        exp_file = os.path.join(ann_dir, img_name + "_exp.npy")

        if not os.path.exists(exp_file):
            continue

        label = int(np.load(exp_file))
        if label > 7:
            continue

        img_paths.append(img_path)
        labels.append(label)

    return np.array(img_paths, dtype=str), np.array(labels, dtype=np.int32)

# ==========================
# PREPROCESSING
# ==========================
def preprocess(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = preprocess_input(img)
    return img, label

# ==========================
# PATH INFO
# ==========================
print("MODEL_PATH:", MODEL_PATH)
print("IMG_DIR:", IMG_DIR)
print("ANN_DIR:", ANN_DIR)
print("IMG_DIR exists:", os.path.exists(IMG_DIR))
print("ANN_DIR exists:", os.path.exists(ANN_DIR))

# ==========================
# DATASET
# ==========================
print("Preparing evaluation set...")

img_paths, labels = get_image_label_paths(IMG_DIR, ANN_DIR)

if len(img_paths) == 0:
    raise ValueError("No images were found. Check IMG_DIR and ANN_DIR.")

print(f"Image count: {len(img_paths)}")
print(f"img_paths type: {type(img_paths)}")
print(f"First path type: {type(img_paths[0])}")
print(f"First path: {img_paths[0]}")
print(f"First label type: {type(labels[0])}")
print(f"First label: {labels[0]}")

dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==========================
# LOAD MODEL
# ==========================
print("Loading model...")
model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={"Dense": CompatibleDense},
    compile=False
)

# ==========================
# PREDICTION
# ==========================
print("Evaluating model...")
y_true = np.concatenate([y.numpy() for _, y in dataset], axis=0)
y_pred_probs = model.predict(dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

# ==========================
# GLOBAL METRICS
# ==========================
acc = accuracy_score(y_true, y_pred)

precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

cm = confusion_matrix(y_true, y_pred)

print("\n===== GLOBAL RESULTS =====")
print(f"Accuracy: {acc:.4f}")
print(f"Precision (Macro): {precision_macro:.4f}")
print(f"Recall (Macro): {recall_macro:.4f}")
print(f"F1-score (Macro): {f1_macro:.4f}")
print(f"Precision (Weighted): {precision_weighted:.4f}")
print(f"Recall (Weighted): {recall_weighted:.4f}")
print(f"F1-score (Weighted): {f1_weighted:.4f}")

print("\n===== CONFUSION MATRIX =====")
print(cm)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

# ==========================
# CONFUSION MATRIX AS IMAGE
# ==========================
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True,
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


