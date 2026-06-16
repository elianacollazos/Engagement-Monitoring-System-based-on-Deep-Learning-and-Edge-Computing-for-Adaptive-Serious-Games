# ==========================================
# VALENCE-AROUSAL CONNECTED TO EMOTION (PRO)
# ==========================================

import os
import numpy as np
import tensorflow as tf
from glob import glob
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ==========================
# CONFIG
# ==========================
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-5

train_img_dir = "data/train_set/images"
train_ann_dir = "data/train_set/annotations"
val_img_dir = "data/val_set/val_set/images"
val_ann_dir = "data/val_set/val_set/annotations"

print("Configuration ready")

# ==========================
# GENERATOR
# ==========================
class AffectNetGenerator(Sequence):

    def __init__(self, img_paths, ann_dir, batch_size):
        super().__init__()  # Avoids a warning
        self.img_paths = img_paths
        self.ann_dir = ann_dir
        self.batch_size = batch_size

    def __len__(self):
        return len(self.img_paths) // self.batch_size

    def __getitem__(self, idx):

        batch_paths = self.img_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        X, y = [], []

        for path in batch_paths:

            name = os.path.basename(path).replace(".jpg", "")

            val_path = os.path.join(self.ann_dir, name + "_val.npy")
            aro_path = os.path.join(self.ann_dir, name + "_aro.npy")

            if not os.path.exists(val_path) or not os.path.exists(aro_path):
                continue

            val = float(np.load(val_path))
            aro = float(np.load(aro_path))

            if val == -2 or aro == -2:
                continue

            val = np.clip(val, -1, 1)
            aro = np.clip(aro, -1, 1)

            img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
            img = image.img_to_array(img)
            img = preprocess_input(img)

            X.append(img)
            y.append([val, aro])

        if len(X) == 0:
            return self.__getitem__((idx + 1) % self.__len__())

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ==========================
# METRIC
# ==========================
def rmse_tf(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# ==========================
# LOAD EMOTION MODEL
# ==========================
print("Loading emotion model...")

emotion_model = tf.keras.models.load_model("clasificacion_em.h5")

feature_extractor = Model(
    inputs=emotion_model.input,
    outputs=emotion_model.layers[-2].output
)

print("Feature extractor ready")

# ==========================
# BUILD VA MODEL
# ==========================
print("Building connected VA model...")

x = feature_extractor.output

x = Dense(256, activation='relu', name="va_dense_1")(x)
x = Dropout(0.3, name="va_dropout_1")(x)
x = Dense(128, activation='relu', name="va_dense_2")(x)

output = Dense(2, activation='tanh', name="va_output")(x)

va_model = Model(
    inputs=feature_extractor.input,
    outputs=output
)

# ==========================
# FREEZE
# ==========================
for layer in feature_extractor.layers:
    layer.trainable = False

print("Emotion model frozen")

# ==========================
# COMPILE
# ==========================
va_model.compile(
    optimizer=Adam(LR),
    loss="mean_squared_error",
    metrics=[rmse_tf]
)

# ==========================
# CALLBACKS
# ==========================
checkpoint = ModelCheckpoint(
    "va_best.h5",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# ==========================
# DATA
# ==========================
print("Loading data...")

train_paths = sorted(glob(os.path.join(train_img_dir, "*.jpg")))
val_paths = sorted(glob(os.path.join(val_img_dir, "*.jpg")))

print("Train:", len(train_paths))
print("Val:", len(val_paths))

train_gen = AffectNetGenerator(train_paths, train_ann_dir, BATCH_SIZE)
val_gen = AffectNetGenerator(val_paths, val_ann_dir, BATCH_SIZE)

# ==========================
# TRAINING PHASE 1
# ==========================
print("Phase 1: initial training...")

va_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# ==========================
# FINE-TUNING
# ==========================
print("Phase 2: fine-tuning...")

for layer in feature_extractor.layers[-50:]:
    layer.trainable = True

va_model.compile(
    optimizer=Adam(1e-5),
    loss="mean_squared_error",
    metrics=[rmse_tf]
)

va_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=[checkpoint, early_stop]
)

# ==========================
# SAVE FINAL MODEL
# ==========================
va_model.save("va_final.h5")

print("Final model saved as va_final.h5")

