import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ==========================
# CONFIG
# ==========================
IMG_SIZE = 160
BATCH_SIZE = 32

MODEL_PATH = os.path.join(BASE_DIR, "app_streamlit", "models", "va_best.h5")
IMG_DIR = os.path.join(BASE_DIR, "data", "val_set", "val_set", "images")
ANN_DIR = os.path.join(BASE_DIR, "data", "val_set", "val_set", "annotations")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "metrics_output")


# ==========================
# COMPATIBILITY PATCH
# ==========================
_original_dense_from_config = Dense.from_config.__func__

@classmethod
def _patched_dense_from_config(cls, config):
    config = dict(config)
    config.pop("quantization_config", None)
    return _original_dense_from_config(cls, config)

Dense.from_config = _patched_dense_from_config


# ==========================
# DATA LOADING
# ==========================
def load_sample(img_path, ann_dir):
    name = os.path.splitext(os.path.basename(img_path))[0]

    val_path = os.path.join(ann_dir, f"{name}_val.npy")
    aro_path = os.path.join(ann_dir, f"{name}_aro.npy")

    if not os.path.exists(val_path) or not os.path.exists(aro_path):
        return None

    val = float(np.load(val_path))
    aro = float(np.load(aro_path))

    if val == -2 or aro == -2:
        return None

    val = np.clip(val, -1, 1)
    aro = np.clip(aro, -1, 1)

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = preprocess_input(img)

    return img.astype(np.float32), np.array([val, aro], dtype=np.float32)


# ==========================
# METRICS
# ==========================
def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))


def corr(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def sagr(y_true, y_pred):
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def ccc(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denominator == 0:
        return float("nan")

    return float((2 * covariance) / denominator)


# ==========================
# PLOTS
# ==========================
def scatter_plot(y_true, y_pred, title, output_path):
    min_axis = float(min(np.min(y_true), np.min(y_pred)))
    max_axis = float(max(np.max(y_true), np.max(y_pred)))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=18)
    plt.plot([min_axis, max_axis], [min_axis, max_axis], "r--", linewidth=1.5)
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


# ==========================
# EVALUATION
# ==========================
def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("MODEL_PATH:", MODEL_PATH)
    print("IMG_DIR:", IMG_DIR)
    print("ANN_DIR:", ANN_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("MODEL exists:", os.path.exists(MODEL_PATH))
    print("IMG_DIR exists:", os.path.exists(IMG_DIR))
    print("ANN_DIR exists:", os.path.exists(ANN_DIR))

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully")

    img_paths = sorted(glob(os.path.join(IMG_DIR, "*.jpg")))
    print(f"Total images found: {len(img_paths)}")

    X = []
    y_true = []
    skipped = 0

    for img_path in img_paths:
        sample = load_sample(img_path, ANN_DIR)
        if sample is None:
            skipped += 1
            continue

        img, target = sample
        X.append(img)
        y_true.append(target)

    if len(X) == 0:
        raise RuntimeError("No valid samples were found for evaluation")

    X = np.array(X, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.float32)

    print(f"Valid samples: {len(X)}")
    print(f"Skipped samples: {skipped}")

    print("Generating predictions...")
    y_pred = model.predict(X, batch_size=BATCH_SIZE, verbose=1)

    val_true = y_true[:, 0]
    aro_true = y_true[:, 1]
    val_pred = y_pred[:, 0]
    aro_pred = y_pred[:, 1]

    val_metrics = {
        "MAE": mae(val_true, val_pred),
        "RMSE": rmse(val_true, val_pred),
        "R2": r2(val_true, val_pred),
        "CORR": corr(val_true, val_pred),
        "SAGR": sagr(val_true, val_pred),
        "CCC": ccc(val_true, val_pred),
    }

    aro_metrics = {
        "MAE": mae(aro_true, aro_pred),
        "RMSE": rmse(aro_true, aro_pred),
        "R2": r2(aro_true, aro_pred),
        "CORR": corr(aro_true, aro_pred),
        "SAGR": sagr(aro_true, aro_pred),
        "CCC": ccc(aro_true, aro_pred),
    }

    print("\nREGRESSION METRICS")
    print("------------------------------------------------------")
    print(f"{'':12s}{'VALENCE':>12s}{'AROUSAL':>12s}")
    print(f"{'MAE':12s}{val_metrics['MAE']:12.6f}{aro_metrics['MAE']:12.6f}")
    print(f"{'RMSE':12s}{val_metrics['RMSE']:12.6f}{aro_metrics['RMSE']:12.6f}")
    print(f"{'R2':12s}{val_metrics['R2']:12.6f}{aro_metrics['R2']:12.6f}")
    print(f"{'CORR':12s}{val_metrics['CORR']:12.6f}{aro_metrics['CORR']:12.6f}")
    print(f"{'SAGR':12s}{val_metrics['SAGR']:12.6f}{aro_metrics['SAGR']:12.6f}")
    print(f"{'CCC':12s}{val_metrics['CCC']:12.6f}{aro_metrics['CCC']:12.6f}")

    scatter_plot(
        val_true,
        val_pred,
        "Valence: Prediction vs Ground Truth",
        os.path.join(OUTPUT_DIR, "scatter_valence.png"),
    )

    scatter_plot(
        aro_true,
        aro_pred,
        "Arousal: Prediction vs Ground Truth",
        os.path.join(OUTPUT_DIR, "scatter_arousal.png"),
    )

    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("REGRESSION METRICS\n")
        f.write("------------------------------------------------------\n")
        f.write(f"{'':12s}{'VALENCE':>12s}{'AROUSAL':>12s}\n")
        f.write(f"{'MAE':12s}{val_metrics['MAE']:12.6f}{aro_metrics['MAE']:12.6f}\n")
        f.write(f"{'RMSE':12s}{val_metrics['RMSE']:12.6f}{aro_metrics['RMSE']:12.6f}\n")
        f.write(f"{'R2':12s}{val_metrics['R2']:12.6f}{aro_metrics['R2']:12.6f}\n")
        f.write(f"{'CORR':12s}{val_metrics['CORR']:12.6f}{aro_metrics['CORR']:12.6f}\n")
        f.write(f"{'SAGR':12s}{val_metrics['SAGR']:12.6f}{aro_metrics['SAGR']:12.6f}\n")
        f.write(f"{'CCC':12s}{val_metrics['CCC']:12.6f}{aro_metrics['CCC']:12.6f}\n")
        f.write("\n")
        f.write(f"Total images: {len(img_paths)}\n")
        f.write(f"Evaluated samples: {len(X)}\n")
        f.write(f"Skipped samples: {skipped}\n")

    np.save(os.path.join(OUTPUT_DIR, "y_true.npy"), y_true)
    np.save(os.path.join(OUTPUT_DIR, "y_pred.npy"), y_pred)

    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    evaluate()
