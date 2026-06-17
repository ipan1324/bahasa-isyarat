"""
train_model.py
==============
Script untuk melatih model CNN pada dataset Sign Language MNIST.
Jalankan script ini sekali untuk menghasilkan:
  - sign_language_model.h5   (model terlatih)
  - label_mapping.json       (mapping index -> huruf)
  - training_history.json    (riwayat akurasi & loss per epoch)

Dataset: Sign Language MNIST (Kaggle)
         https://www.kaggle.com/datasets/datamunge/sign-language-mnist

Struktur CSV:
  Baris pertama: header "label,pixel1,pixel2,...,pixel784"
  label: 0=A, 1=B, ..., 24=Y  (J=9 dan Z=25 tidak ada)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Kompatibilitas Keras 3.x (TF >= 2.16)
try:
    import keras
    from keras import layers
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ── KONFIGURASI ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, 'Data', 'archive (2)')
TRAIN_CSV    = os.path.join(DATA_DIR, 'sign_mnist_train.csv')
TEST_CSV     = os.path.join(DATA_DIR, 'sign_mnist_test.csv')

MODEL_OUT    = os.path.join(BASE_DIR, 'sign_language_model.h5')
LABEL_OUT    = os.path.join(BASE_DIR, 'label_mapping.json')
HISTORY_OUT  = os.path.join(BASE_DIR, 'training_history.json')
PLOT_OUT     = os.path.join(BASE_DIR, 'static', 'training_plot.png')

IMG_SIZE     = 28
BATCH_SIZE   = 64
EPOCHS       = 15          # EarlyStopping akan berhenti lebih awal jika sudah konvergen
VALIDATION_SPLIT = 0.15   # 15% dari data train untuk validasi
RANDOM_SEED  = 42

# ── 1. LOAD DATASET ─────────────────────────────────────────────────────────
print("=" * 55)
print(" SIGN LANGUAGE CNN – TRAINING SCRIPT")
print("=" * 55)
print(f"\n[1/6] Memuat dataset...")
print(f"      Train CSV : {TRAIN_CSV}")
print(f"      Test  CSV : {TEST_CSV}")

df_train_full = pd.read_csv(TRAIN_CSV)
df_test       = pd.read_csv(TEST_CSV)

print(f"      Baris train : {len(df_train_full):,}")
print(f"      Baris test  : {len(df_test):,}")

# ── 2. PREPROCESSING ─────────────────────────────────────────────────────────
print("\n[2/6] Preprocessing data...")

def load_images_labels(df):
    """Pisahkan kolom label dan pixel, reshape ke (N, 28, 28, 1)."""
    labels  = df['label'].values
    pixels  = df.drop('label', axis=1).values.astype(np.float32) / 255.0
    images  = pixels.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return images, labels

X_train_full, y_train_full = load_images_labels(df_train_full)
X_test,       y_test       = load_images_labels(df_test)

# Split train → train + val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_SEED,
    stratify=y_train_full
)

print(f"      X_train : {X_train.shape}")
print(f"      X_val   : {X_val.shape}")
print(f"      X_test  : {X_test.shape}")

# Label unik dan pemetaan index ↔ label asli
unique_labels = sorted(np.unique(y_train_full).tolist())   # e.g. [0,1,2,...,24]
num_classes   = len(unique_labels)
label_to_idx  = {lbl: idx for idx, lbl in enumerate(unique_labels)}
idx_to_label  = {idx: lbl for idx, lbl in enumerate(unique_labels)}

print(f"      Kelas unik    : {num_classes}  →  {unique_labels}")

# One-hot encode
y_train_cat = keras.utils.to_categorical(
    [label_to_idx[l] for l in y_train], num_classes
)
y_val_cat = keras.utils.to_categorical(
    [label_to_idx[l] for l in y_val], num_classes
)
y_test_cat = keras.utils.to_categorical(
    [label_to_idx[l] for l in y_test], num_classes
)

# ── 3. DATA AUGMENTATION ─────────────────────────────────────────────────────
print("\n[3/6] Menyiapkan augmentasi data...")

data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
], name='augmentation')

# ── 4. BANGUN MODEL CNN ───────────────────────────────────────────────────────
print("\n[4/6] Membangun arsitektur CNN...")

def build_model(num_classes):
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='input')

    # Augmentasi (hanya aktif saat training)
    x = data_augmentation(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                      name='conv1a')(x)
    x = layers.BatchNormalization(name='bn1a')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                      name='conv1b')(x)
    x = layers.BatchNormalization(name='bn1b')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(0.25, name='drop1')(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                      name='conv2a')(x)
    x = layers.BatchNormalization(name='bn2a')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                      name='conv2b')(x)
    x = layers.BatchNormalization(name='bn2b')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(0.30, name='drop2')(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                      name='conv3a')(x)
    x = layers.BatchNormalization(name='bn3a')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(0.35, name='drop3')(x)

    # Classifier
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization(name='bn_fc')(x)
    x = layers.Dropout(0.40, name='drop_fc')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs, outputs, name='SignLanguageCNN')
    return model

model = build_model(num_classes)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── 5. TRAINING ───────────────────────────────────────────────────────────────
print("\n[5/6] Melatih model...")

os.makedirs(os.path.join(BASE_DIR, 'static'), exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=MODEL_OUT.replace('.h5', '_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ── 6. EVALUASI & SIMPAN ──────────────────────────────────────────────────────
print("\n[6/6] Evaluasi dan menyimpan artefak...")

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n  Test Accuracy : {test_acc * 100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")

# Simpan model
model.save(MODEL_OUT)
print(f"  Model disimpan → {MODEL_OUT}")

# Simpan label mapping
label_data = {
    'unique_labels': unique_labels,
    'idx_to_label' : {str(k): v for k, v in idx_to_label.items()},
    'label_to_idx' : {str(k): v for k, v in label_to_idx.items()},
    'num_classes'  : num_classes
}
with open(LABEL_OUT, 'w') as f:
    json.dump(label_data, f, indent=2)
print(f"  Label mapping  → {LABEL_OUT}")

# Simpan history
hist_dict = {
    'accuracy'    : [float(v) for v in history.history['accuracy']],
    'val_accuracy': [float(v) for v in history.history['val_accuracy']],
    'loss'        : [float(v) for v in history.history['loss']],
    'val_loss'    : [float(v) for v in history.history['val_loss']],
    'test_accuracy': float(test_acc),
    'test_loss'   : float(test_loss),
    'num_epochs_trained': len(history.history['accuracy'])
}
with open(HISTORY_OUT, 'w') as f:
    json.dump(hist_dict, f, indent=2)
print(f"  History        → {HISTORY_OUT}")

# Plot training
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#0a0e17')
for ax in axes:
    ax.set_facecolor('#111827')
    ax.tick_params(colors='#94a3b8')
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('#e2e8f0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')

epochs_range = range(1, len(history.history['accuracy']) + 1)

axes[0].plot(epochs_range, [v*100 for v in history.history['accuracy']],
             color='#6366f1', linewidth=2, label='Train')
axes[0].plot(epochs_range, [v*100 for v in history.history['val_accuracy']],
             color='#06b6d4', linewidth=2, label='Validasi')
axes[0].set_title('Akurasi per Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Akurasi (%)')
axes[0].legend(facecolor='#1a2234', labelcolor='#94a3b8')
axes[0].grid(color='#1e293b')

axes[1].plot(epochs_range, history.history['loss'],
             color='#8b5cf6', linewidth=2, label='Train')
axes[1].plot(epochs_range, history.history['val_loss'],
             color='#ef4444', linewidth=2, label='Validasi')
axes[1].set_title('Loss per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(facecolor='#1a2234', labelcolor='#94a3b8')
axes[1].grid(color='#1e293b')

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Plot           → {PLOT_OUT}")

print("\n" + "=" * 55)
print(f"  SELESAI! Test Accuracy: {test_acc * 100:.2f}%")
print("=" * 55)