"""
Face Detection and Mask Classification App
==========================================
Module: core/detector.py
Hỗ trợ phát hiện NHIỀU khuôn mặt cùng lúc.
"""

import cv2
import numpy as np
import os
import urllib.request

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_BASE_DIR, "models")

PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

PROTOTXT_PATH = os.path.join(MODELS_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

_dnn_net = None
_hog_detector = None


def _ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


def _download_dnn_models(progress_callback=None):
    _ensure_models_dir()
    if not os.path.exists(PROTOTXT_PATH):
        if progress_callback:
            progress_callback("Dang tai deploy.prototxt tu GitHub...")
        urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)
    if not os.path.exists(CAFFEMODEL_PATH):
        if progress_callback:
            progress_callback("Dang tai caffemodel (~5MB)...")
        urllib.request.urlretrieve(CAFFEMODEL_URL, CAFFEMODEL_PATH)


def get_dnn_net(progress_callback=None):
    global _dnn_net
    if _dnn_net is None:
        _download_dnn_models(progress_callback)
        _dnn_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    return _dnn_net


def get_hog_detector():
    global _hog_detector
    if _hog_detector is None:
        try:
            import dlib
            _hog_detector = dlib.get_frontal_face_detector()
        except ImportError:
            _hog_detector = "unavailable"
    return _hog_detector


# ============================================================
# PHÁT HIỆN NHIỀU KHUÔN MẶT
# ============================================================

def detect_faces_dnn(img_bgr, conf_threshold=0.5, padding_ratio=0.15, progress_callback=None):
    """
    Phát hiện TẤT CẢ khuôn mặt bằng OpenCV DNN.

    Returns:
        list of (bbox, confidence): [(x,y,w,h), conf], ...
        Trả về list rỗng nếu không phát hiện được.
    """
    net = get_dnn_net(progress_callback)
    h_img, w_img = img_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img_bgr, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_threshold:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w_img, h_img, w_img, h_img])
        (startX, startY, endX, endY) = box.astype("int")

        w = endX - startX
        h = endY - startY

        # Bỏ qua box quá nhỏ hoặc không hợp lệ
        if w <= 0 or h <= 0:
            continue

        # Áp dụng padding
        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)
        new_x = max(0, startX - pad_w)
        new_y = max(0, startY - pad_h)
        new_w = min(w_img - new_x, w + 2 * pad_w)
        new_h = min(h_img - new_y, h + 2 * pad_h)

        results.append(((new_x, new_y, new_w, new_h), confidence))

    # Sắp xếp theo diện tích bounding box (lớn trước)
    results.sort(key=lambda r: r[0][2] * r[0][3], reverse=True)
    return results


def detect_faces_hog(img_bgr, padding_ratio=0.15):
    """
    Phát hiện TẤT CẢ khuôn mặt bằng Dlib HOG.

    Returns:
        list of (bbox, confidence)
    """
    detector = get_hog_detector()
    if detector == "unavailable":
        return detect_faces_dnn(img_bgr)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    h_img, w_img = img_bgr.shape[:2]

    results = []
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.width()
        h = face.height()

        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)
        new_x = max(0, x - pad_w)
        new_y = max(0, y - pad_h)
        new_w = min(w_img - new_x, w + 2 * pad_w)
        new_h = min(h_img - new_y, h + 2 * pad_h)

        results.append(((new_x, new_y, new_w, new_h), 1.0))

    results.sort(key=lambda r: r[0][2] * r[0][3], reverse=True)
    return results


def detect_all_faces(img_bgr, method="DNN", conf_threshold=0.5):
    """
    Wrapper thống nhất: phát hiện TẤT CẢ khuôn mặt.

    Returns:
        list of ((x,y,w,h), confidence)  — rỗng nếu không tìm thấy
    """
    if method == "HOG":
        return detect_faces_hog(img_bgr)
    else:
        return detect_faces_dnn(img_bgr, conf_threshold=conf_threshold)


# Giữ lại API cũ (single-face) để tương thích
def detect_faces(img_bgr, method="DNN", conf_threshold=0.5):
    faces = detect_all_faces(img_bgr, method=method, conf_threshold=conf_threshold)
    if not faces:
        h_img, w_img = img_bgr.shape[:2]
        return (0, 0, w_img, h_img), 0.0
    return faces[0]


def is_fallback(bbox, img_shape):
    x, y, w, h = bbox
    h_img, w_img = img_shape[:2]
    return w == w_img and h == h_img
