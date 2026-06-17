# SignVision – Klasifikasi Bahasa Isyarat ASL dengan CNN 🤟

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3-black?logo=flask)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)](https://tensorflow.org)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple?logo=bootstrap)](https://getbootstrap.com)

> **Tugas 10** — Penerapan CNN pada Aplikasi Berbasis Web Menggunakan Flask  
> Praktikum Kecerdasan Buatan | Teknik Informatika – Universitas Bale Bandung

---

## 📌 Deskripsi

**SignVision** adalah aplikasi web berbasis Flask yang menggunakan **Convolutional Neural Network (CNN)** untuk mengklasifikasikan huruf dalam **American Sign Language (ASL)**. Pengguna cukup mengunggah foto tangan, dan sistem akan memprediksi huruf isyarat yang ditunjukkan secara real-time.

### Fitur Utama
- 🖼️ Upload gambar dengan drag-and-drop atau klik
- 🤖 Prediksi huruf isyarat menggunakan CNN (TensorFlow/Keras)
- 📊 Top-5 probabilitas huruf kandidat dengan progress bar
- 📈 Grafik akurasi dan loss selama pelatihan (Chart.js)
- 🔤 Grid referensi alfabet ASL interaktif
- 🌙 Dark mode premium UI menggunakan Bootstrap 5

---

## 🗂️ Struktur Proyek

```
bhs isyarat/
├── app.py                    # Flask backend (main app)
├── train_model.py            # Script pelatihan model CNN
├── requirements.txt          # Dependensi Python
├── Procfile                  # Konfigurasi Render/Heroku
├── runtime.txt               # Versi Python untuk deployment
├── sign_language_model.h5    # ← Dihasilkan oleh train_model.py
├── label_mapping.json        # ← Dihasilkan oleh train_model.py
├── training_history.json     # ← Dihasilkan oleh train_model.py
├── templates/
│   └── index.html            # Halaman web utama (Jinja2)
├── static/
│   └── training_plot.png     # ← Dihasilkan oleh train_model.py
└── Data/
    └── archive (2)/
        ├── sign_mnist_train.csv
        └── sign_mnist_test.csv
```

---

## 🧠 Arsitektur Model CNN

```
Input (28×28×1)
    │
    ├─ Data Augmentation (Flip, Rotate, Zoom, Translate)
    │
    ├─ Conv2D Block 1: 2× Conv 32 filter (3×3) + BN + MaxPool + Dropout 0.25
    ├─ Conv2D Block 2: 2× Conv 64 filter (3×3) + BN + MaxPool + Dropout 0.30
    ├─ Conv2D Block 3:    Conv 128 filter (3×3) + BN + MaxPool + Dropout 0.35
    │
    ├─ Flatten → Dense 256 + BN + Dropout 0.40
    │
    └─ Dense 24 (Softmax) → Output
```

**Optimizer:** Adam (lr=1e-3, ReduceLROnPlateau)  
**Loss:** Categorical Cross-Entropy  
**Regularisasi:** L2 (1e-4) + Batch Normalization + Dropout  
**Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  

---

## 📦 Dataset

- **Nama:** Sign Language MNIST
- **Sumber:** [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- **Train:** 27.455 sampel
- **Test:** 7.172 sampel
- **Format:** CSV (label + 784 nilai piksel grayscale 28×28)
- **Kelas:** 24 huruf (A-Y, kecuali J dan Z yang memerlukan gerakan dinamis)

---

## 🚀 Cara Menjalankan Lokal

### 1. Clone / Download Proyek
```bash
git clone https://github.com/username/signvision.git
cd signvision
```

### 2. Buat Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. Latih Model CNN (jalankan sekali)
```bash
python train_model.py
```
> ⏱️ Proses ini membutuhkan waktu beberapa menit. Akan menghasilkan:
> - `sign_language_model.h5`
> - `label_mapping.json`
> - `training_history.json`

### 5. Jalankan Aplikasi Flask
```bash
python app.py
```

### 6. Buka Browser
```
http://localhost:5000
```

---

## ☁️ Deployment ke Render (Gratis)

1. Push kode ke GitHub (pastikan `sign_language_model.h5`, `label_mapping.json`, dan `training_history.json` sudah ter-commit)
2. Buat akun di [render.com](https://render.com)
3. New → Web Service → Connect GitHub Repository
4. Konfigurasi:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - **Environment:** Python 3
5. Deploy!

---

## 🛠️ Teknologi yang Digunakan

| Kategori | Teknologi |
|---|---|
| Backend | Flask 2.3, Python 3.10 |
| Deep Learning | TensorFlow 2.13, Keras |
| Image Processing | Pillow (PIL), NumPy |
| Data Processing | Pandas, Scikit-learn |
| Frontend | Bootstrap 5.3, Chart.js 4.4 |
| Deployment | Render / Heroku (Gunicorn) |
| Dataset | Sign Language MNIST (Kaggle) |

---

## 📝 Informasi Tugas

- **Mata Kuliah:** Praktikum Kecerdasan Buatan
- **Tugas:** Tugas 10 – Penerapan CNN pada Aplikasi Berbasis Web
- **Studi Kasus:** Klasifikasi Huruf Bahasa Isyarat (Sign Language Recognition)
- **Institusi:** Teknik Informatika – Universitas Bale Bandung

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik. Dataset Sign Language MNIST tersedia secara publik di Kaggle.
