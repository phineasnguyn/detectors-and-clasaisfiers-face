"""
Face Detection and Mask Classification App
==========================================
Module: core/classifier.py
Trích xuất logic Classification từ notebook gốc.
Hỗ trợ: Linear SVM, Random Forest, CNN (Keras).
"""

import cv2
import numpy as np
import os
import joblib

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_BASE_DIR, "models")

# Singleton models
_models = {}

LABELS = {0: "Without Mask", 1: "With Mask"}
LABEL_COLORS = {
    0: (0, 0, 220),    # Đỏ cho không đeo khẩu trang (BGR)
    1: (0, 200, 60),   # Xanh lá cho đeo khẩu trang (BGR)
}


# ============================================================
# PREPROCESSING (từ notebook gốc)
# ============================================================

def custom_preprocess(resized_bgr):
    """Chuyển sang grayscale + BilateralFilter - từ notebook gốc."""
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 5, 75, 75)
    return filtered


def crop_and_resize(img, bbox, size=(64, 64)):
    """Cắt và resize vùng khuôn mặt - từ notebook gốc."""
    x, y, w, h = bbox
    # Đảm bảo tọa độ hợp lệ
    h_img, w_img = img.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    if w <= 0 or h <= 0:
        return cv2.resize(img, size)
    face_roi = img[y:y+h, x:x+w]
    return cv2.resize(face_roi, size)


def extract_hog_features(gray_img):
    """Trích xuất HOG features - từ notebook gốc."""
    from skimage.feature import hog
    feat = hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        block_norm='L2-Hys'
    )
    return feat


# ============================================================
# LOAD MODELS
# ============================================================

def load_svm_model():
    """Nạp SVM model từ file .pkl."""
    global _models
    if "svm" not in _models:
        path = os.path.join(MODELS_DIR, "svm_model.pkl")
        if not os.path.exists(path):
            return None
        _models["svm"] = joblib.load(path)
    return _models["svm"]


def load_rf_model():
    """Nạp Random Forest model từ file .pkl."""
    global _models
    if "rf" not in _models:
        path = os.path.join(MODELS_DIR, "rf_model.pkl")
        if not os.path.exists(path):
            return None
        _models["rf"] = joblib.load(path)
    return _models["rf"]


def load_cnn_model():
    """Nạp CNN model từ file .h5."""
    global _models
    if "cnn" not in _models:
        path = os.path.join(MODELS_DIR, "cnn_model.h5")
        if not os.path.exists(path):
            # Thử .keras extension
            path_keras = os.path.join(MODELS_DIR, "cnn_model.keras")
            if os.path.exists(path_keras):
                path = path_keras
            else:
                return None
        try:
            from tensorflow.keras.models import load_model
            _models["cnn"] = load_model(path)
        except Exception:
            return None
    return _models["cnn"]


def get_available_classifiers():
    """Trả về danh sách các classifier có sẵn (đã có file model)."""
    available = []
    if os.path.exists(os.path.join(MODELS_DIR, "svm_model.pkl")):
        available.append("SVM")
    if os.path.exists(os.path.join(MODELS_DIR, "rf_model.pkl")):
        available.append("Random Forest")
    if (os.path.exists(os.path.join(MODELS_DIR, "cnn_model.h5")) or
            os.path.exists(os.path.join(MODELS_DIR, "cnn_model.keras"))):
        available.append("CNN")
    return available


# ============================================================
# PREDICT
# ============================================================

def predict_mask(roi_bgr, classifier_type="CNN"):
    """
    Dự đoán có khẩu trang hay không từ vùng khuôn mặt đã cắt.
    
    Args:
        roi_bgr: Numpy array BGR, shape bất kỳ (sẽ tự resize về 64x64)
        classifier_type: "CNN", "SVM", "Random Forest"
    
    Returns:
        dict: {
            "label": "With Mask" / "Without Mask",
            "label_id": 1 / 0,
            "confidence": float 0.0-1.0,
            "color": (B, G, R)
        }
    """
    # Resize về 64x64
    roi_64 = cv2.resize(roi_bgr, (64, 64))

    if classifier_type == "CNN":
        model = load_cnn_model()
        if model is None:
            return _not_available_result("CNN")
        # CNN dùng ảnh màu RGB
        roi_rgb = cv2.cvtColor(roi_64, cv2.COLOR_BGR2RGB)
        cnn_input = np.array([roi_rgb]).astype('float32') / 255.0
        prob = float(model.predict(cnn_input, verbose=0)[0][0])
        label_id = 1 if prob >= 0.5 else 0
        confidence = prob if label_id == 1 else 1.0 - prob
        return _make_result(label_id, confidence)

    elif classifier_type == "SVM":
        model = load_svm_model()
        if model is None:
            return _not_available_result("SVM")
        # SVM dùng HOG features
        preprocessed = custom_preprocess(roi_64)
        feat = extract_hog_features(preprocessed)
        label_id = int(model.predict([feat])[0])
        # Lấy score để ước tính confidence
        try:
            score = model.decision_function([feat])[0]
            confidence = float(1.0 / (1.0 + np.exp(-abs(score))))
        except Exception:
            confidence = 1.0
        return _make_result(label_id, confidence)

    elif classifier_type == "Random Forest":
        model = load_rf_model()
        if model is None:
            return _not_available_result("Random Forest")
        preprocessed = custom_preprocess(roi_64)
        feat = extract_hog_features(preprocessed)
        label_id = int(model.predict([feat])[0])
        proba = model.predict_proba([feat])[0]
        confidence = float(max(proba))
        return _make_result(label_id, confidence)

    return _not_available_result(classifier_type)


def _make_result(label_id, confidence):
    return {
        "label": LABELS[label_id],
        "label_id": label_id,
        "confidence": confidence,
        "color": LABEL_COLORS[label_id]
    }


def _not_available_result(name):
    return {
        "label": f"Model {name} chưa có",
        "label_id": -1,
        "confidence": 0.0,
        "color": (128, 128, 128)
    }
