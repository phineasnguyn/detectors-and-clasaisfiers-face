"""
╔══════════════════════════════════════════════════════════════╗
║  SAVE MODELS - Chạy script này trong Jupyter Notebook       ║
║  để lưu các model đã train vào thư mục models/             ║
╚══════════════════════════════════════════════════════════════╝

Thêm cell này vào cuối Jupyter Notebook của bạn và chạy.
Sau đó app.py sẽ tự động nạp các model khi khởi động.
"""

import os
import joblib

# Tạo thư mục models nếu chưa có
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)

print("📁 Thư mục models:", models_dir)
print()

# ============================================================
# COPY-PASTE ĐOẠN NÀY VÀO CUỐI JUPYTER NOTEBOOK CỦA BẠN:
# ============================================================

SAVE_CODE = '''
import os, joblib

models_dir = "models"  # Thay bằng đường dẫn tuyệt đối nếu cần
os.makedirs(models_dir, exist_ok=True)

# 1. Lưu SVM model (nếu đã train)
if 'svm_model' in dir():
    joblib.dump(svm_model, os.path.join(models_dir, "svm_model.pkl"))
    print("✅ Đã lưu SVM model →", os.path.join(models_dir, "svm_model.pkl"))

# 2. Lưu Random Forest model (nếu đã train)  
if 'rf_model' in dir():
    joblib.dump(rf_model, os.path.join(models_dir, "rf_model.pkl"))
    print("✅ Đã lưu RF model →", os.path.join(models_dir, "rf_model.pkl"))

# 3. Lưu CNN model (nếu đã train)
if 'cnn_model' in dir():
    cnn_model.save(os.path.join(models_dir, "cnn_model.h5"))
    print("✅ Đã lưu CNN model →", os.path.join(models_dir, "cnn_model.h5"))

print()
print("🎉 Hoàn tất! Bây giờ hãy chạy: python app.py")
'''

print("=" * 60)
print("📋 Copy đoạn code sau vào Jupyter Notebook của bạn:")
print("=" * 60)
print(SAVE_CODE)
print("=" * 60)
