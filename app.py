"""
app.py
======
Flask web application untuk klasifikasi bahasa isyarat menggunakan CNN.

Jalankan setelah train_model.py selesai:
    python app.py

Lalu buka browser: http://localhost:5000
"""

import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow import keras
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max upload

# ── PATH ARTEFAK ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, 'sign_language_model.h5')
LABEL_PATH    = os.path.join(BASE_DIR, 'label_mapping.json')
HISTORY_PATH  = os.path.join(BASE_DIR, 'training_history.json')

# ── LOAD MODEL & METADATA ─────────────────────────────────────────────────────
print("[app] Memuat model CNN...")
model = keras.models.load_model(MODEL_PATH)
print("[app] Model berhasil dimuat.")

with open(LABEL_PATH) as f:
    label_data = json.load(f)

with open(HISTORY_PATH) as f:
    training_history = json.load(f)

unique_labels = label_data['unique_labels']   # [0, 1, 2, ..., 24]
num_classes   = label_data['num_classes']

# ── HELPER ───────────────────────────────────────────────────────────────────
def idx_to_letter(idx: int) -> str:
    """Konversi index prediksi → huruf (A-Y, J tidak ada)."""
    orig_label = unique_labels[idx]        # 0→A … 24→Y
    return chr(65 + orig_label)            # ASCII

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize, grayscale, normalize gambar untuk input CNN."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('L')                 # grayscale
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 28, 28, 1)

# ── ROUTES ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    """Halaman utama aplikasi."""
    history_data = {
        'epochs'      : list(range(1, len(training_history['accuracy']) + 1)),
        'accuracy'    : [round(v * 100, 2) for v in training_history['accuracy']],
        'val_accuracy': [round(v * 100, 2) for v in training_history['val_accuracy']],
        'loss'        : [round(v, 4) for v in training_history['loss']],
        'val_loss'    : [round(v, 4) for v in training_history['val_loss']],
    }

    final_acc   = round(training_history.get('test_accuracy',
                        training_history['val_accuracy'][-1]) * 100, 2)
    final_loss  = round(training_history['val_loss'][-1], 4)
    epochs_ran  = training_history.get('num_epochs_trained',
                                       len(training_history['accuracy']))

    return render_template(
        'index.html',
        history      = history_data,
        final_acc    = final_acc,
        final_loss   = final_loss,
        num_classes  = num_classes,
        epochs_ran   = epochs_ran,
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint POST untuk prediksi huruf isyarat dari gambar."""
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    allowed = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': f'Format .{ext} tidak didukung. Gunakan PNG/JPG/BMP/WEBP.'}), 400

    try:
        image_bytes = file.read()

        # Preview 28×28 grayscale → base64 untuk ditampilkan di web
        img_prev = Image.open(io.BytesIO(image_bytes)).convert('L').resize((28, 28))
        buf = io.BytesIO()
        img_prev.save(buf, format='PNG')
        preview_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Prediksi
        img_arr     = preprocess_image(image_bytes)
        predictions = model.predict(img_arr, verbose=0)[0]

        # Top-5 hasil
        top5_idx = np.argsort(predictions)[::-1][:5]
        top5 = [
            {
                'letter'    : idx_to_letter(int(i)),
                'confidence': round(float(predictions[i]) * 100, 2)
            }
            for i in top5_idx
        ]

        return jsonify({
            'predicted_letter': top5[0]['letter'],
            'confidence'      : top5[0]['confidence'],
            'top5'            : top5,
            'preview'         : preview_b64,
        })

    except Exception as exc:
        return jsonify({'error': f'Gagal memproses gambar: {exc}'}), 500

@app.route('/health')
def health():
    """Health-check endpoint untuk deployment."""
    return jsonify({'status': 'ok', 'model': 'sign_language_cnn', 'classes': num_classes})

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)