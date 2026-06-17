"""
test_app.py
===========
Script pengujian sederhana untuk memastikan semua komponen bekerja dengan benar
sebelum menjalankan aplikasi Flask.

Jalankan: python test_app.py
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def check_file(path, label):
    exists = os.path.isfile(path)
    status = "[OK]" if exists else "[X]"
    size   = f"({os.path.getsize(path) / 1024:.1f} KB)" if exists else "(tidak ada)"
    print(f"  {status}  {label:<30}  {size}")
    return exists

def main():
    print("=" * 55)
    print("  SIGNVISION – SYSTEM CHECK")
    print("=" * 55)

    # 1. Cek file model dan metadata
    print("\n[1] File Artefak Model:")
    ok_model   = check_file(os.path.join(BASE_DIR, 'sign_language_model.h5'),  'sign_language_model.h5')
    ok_label   = check_file(os.path.join(BASE_DIR, 'label_mapping.json'),       'label_mapping.json')
    ok_history = check_file(os.path.join(BASE_DIR, 'training_history.json'),    'training_history.json')

    # 2. Cek template
    print("\n[2] File Template:")
    ok_tmpl = check_file(os.path.join(BASE_DIR, 'templates', 'index.html'),     'templates/index.html')

    # 3. Cek dataset
    print("\n[3] Dataset:")
    check_file(os.path.join(BASE_DIR, 'Data', 'archive (2)', 'sign_mnist_train.csv'), 'sign_mnist_train.csv')
    check_file(os.path.join(BASE_DIR, 'Data', 'archive (2)', 'sign_mnist_test.csv'),  'sign_mnist_test.csv')

    # 4. Validasi JSON
    if ok_label:
        print("\n[4] Validasi label_mapping.json:")
        with open(os.path.join(BASE_DIR, 'label_mapping.json')) as f:
            data = json.load(f)
        print(f"  ✅ num_classes    = {data['num_classes']}")
        print(f"  ✅ unique_labels  = {data['unique_labels']}")

    if ok_history:
        print("\n[5] Validasi training_history.json:")
        with open(os.path.join(BASE_DIR, 'training_history.json')) as f:
            hist = json.load(f)
        epochs = len(hist['accuracy'])
        best_val_acc = max(hist['val_accuracy']) * 100
        test_acc = hist.get('test_accuracy', 0) * 100
        print(f"  ✅ Epoch dilatih  = {epochs}")
        print(f"  ✅ Best val acc   = {best_val_acc:.2f}%")
        print(f"  ✅ Test accuracy  = {test_acc:.2f}%")

    # 5. Uji import library
    print("\n[6] Import Library:")
    libs = [
        ('flask',       'Flask'),
        ('tensorflow',  'TensorFlow'),
        ('PIL',         'Pillow'),
        ('numpy',       'NumPy'),
        ('pandas',      'Pandas'),
    ]
    all_ok = True
    for module, name in libs:
        try:
            m = __import__(module)
            ver = getattr(m, '__version__', '?')
            print(f"  ✅  {name:<15} v{ver}")
        except ImportError:
            print(f"  ❌  {name:<15} TIDAK TERPASANG – jalankan: pip install {module}")
            all_ok = False

    # 6. Uji prediksi
    if ok_model and ok_label and all_ok:
        print("\n[7] Uji Prediksi Model:")
        try:
            import numpy as np
            from tensorflow import keras

            model = keras.models.load_model(os.path.join(BASE_DIR, 'sign_language_model.h5'))
            dummy = np.random.rand(1, 28, 28, 1).astype('float32')
            pred  = model.predict(dummy, verbose=0)[0]
            print(f"  ✅ Model loaded  — output shape: {pred.shape}, sum={pred.sum():.4f}")
        except Exception as e:
            print(f"  ❌ Gagal load model: {e}")

    # Summary
    print("\n" + "=" * 55)
    if ok_model and ok_label and ok_history and ok_tmpl and all_ok:
        print("  🎉 SEMUA KOMPONEN SIAP! Jalankan: python app.py")
    else:
        print("  ⚠️  Ada komponen yang belum siap. Periksa output di atas.")
        if not (ok_model and ok_label and ok_history):
            print("     → Jalankan dahulu: python train_model.py")
        if not all_ok:
            print("     → Install dependensi: pip install -r requirements.txt")
    print("=" * 55)

if __name__ == '__main__':
    main()
