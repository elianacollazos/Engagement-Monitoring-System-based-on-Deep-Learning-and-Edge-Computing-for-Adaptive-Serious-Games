import os
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==========================
# CONFIG
# ==========================
IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 25
NUM_CLASSES = 8

train_img_dir = "data/train_set/images"
train_ann_dir = "data/train_set/annotations"
val_img_dir = "data/val_set/val_set/images"
val_ann_dir = "data/val_set/val_set/annotations"

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

    return img_paths, labels

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
# DATA AUGMENTATION
# ==========================
def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return img, label

print("Preparing dataset...")

train_paths, train_labels = get_image_label_paths(train_img_dir, train_ann_dir)
val_paths, val_labels = get_image_label_paths(val_img_dir, val_ann_dir)

# ==========================
# DATASET
# ==========================
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

train_dataset = train_dataset.shuffle(10000)
train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

val_dataset = val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==========================
# CLASS WEIGHTS
# ==========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)

# ==========================
# MODEL
# ==========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Direct fine-tuning
base_model.trainable = True

for layer in base_model.layers[:-100]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)

output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=2e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================
# CALLBACKS
# ==========================
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(patience=3, factor=0.3)

# ==========================
# TRAINING
# ==========================
print("Training (fine-tuning)...")

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

# ==========================
# SAVE
# ==========================
model.save("clasificacion_em.h5")
print("Model saved as clasificacion_em.h5")

# ==========================
# EVALUATION
# ==========================
print("Evaluating model...")

y_true = np.concatenate([y for x, y in val_dataset], axis=0)
y_pred_probs = model.predict(val_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {acc:.4f}")

# Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion matrix:")
print(cm)

# Report
class_names = [
    "Neutral", "Happy", "Sad", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt"
]

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
