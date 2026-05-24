"""
Face Detection and Mask Classification App
==========================================
Module: core/pipeline.py
Hỗ trợ xử lý NHIỀU khuôn mặt cùng lúc trong 1 frame.
"""

import cv2
import numpy as np
import time
from .detector import detect_all_faces, is_fallback
from .classifier import predict_mask, crop_and_resize, LABEL_COLORS


# ============================================================
# VẼ KẾT QUẢ LÊN ẢNH
# ============================================================

def draw_face_result(output, bbox, result, face_conf=None):
    """
    Vẽ bounding box + nhãn cho MỘT khuôn mặt lên frame (in-place).
    """
    x, y, w, h = bbox
    color = result["color"]
    label = result["label"]
    cls_conf = result["confidence"]

    # Bounding box góc bo
    _draw_rounded_rect(output, (x, y), (x + w, y + h), color, thickness=3, radius=10)

    # Nhãn nền
    label_text = f"{label}  {cls_conf*100:.1f}%"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.60
    font_thickness = 1
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

    tag_x1 = x
    tag_y1 = max(0, y - th - 14)
    tag_x2 = min(output.shape[1], x + tw + 14)
    tag_y2 = max(tag_y1 + th + 6, y - 2)

    cv2.rectangle(output, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
    cv2.putText(
        output, label_text,
        (tag_x1 + 7, tag_y2 - baseline - 2),
        font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
    )

    # Confidence phát hiện mặt (nhỏ, bên dưới box)
    if face_conf is not None and face_conf > 0:
        det_text = f"Det:{face_conf*100:.0f}%"
        cv2.putText(
            output, det_text,
            (x + 4, y + h + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
        )


def draw_result_on_frame(frame_bgr, bbox, result, face_conf=None, fps=None):
    """Wrapper đơn giản cho single-face (tương thích ngược)."""
    output = frame_bgr.copy()
    draw_face_result(output, bbox, result, face_conf)
    if fps is not None:
        _draw_fps(output, fps)
    return output


def draw_multi_face_results(frame_bgr, face_results, fps=None):
    """
    Vẽ kết quả cho NHIỀU khuôn mặt lên frame.

    Args:
        frame_bgr: Frame gốc
        face_results: list of (bbox, result_dict, face_conf)
        fps: FPS counter (optional)

    Returns:
        numpy array BGR đã annotate
    """
    output = frame_bgr.copy()

    for bbox, result, face_conf in face_results:
        draw_face_result(output, bbox, result, face_conf)

    # Đếm số mặt
    n = len(face_results)
    count_text = f"Faces: {n}"
    cv2.putText(output, count_text, (12, 62),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, count_text, (12, 62),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 200, 255), 1, cv2.LINE_AA)

    if fps is not None:
        _draw_fps(output, fps)

    return output


def _draw_fps(img, fps):
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(img, fps_text, (12, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, fps_text, (12, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (30, 220, 130), 1, cv2.LINE_AA)


def draw_no_face(frame_bgr):
    output = frame_bgr.copy()
    h, w = output.shape[:2]
    text = "Khong phat hien khuon mat"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.7
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    tx = (w - tw) // 2
    ty = h - 20
    cv2.putText(output, text, (tx, ty), font, scale, (80, 80, 220), 1, cv2.LINE_AA)
    return output


def _draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r <= 0:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90,  0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0,   0, 90, color, thickness)


# ============================================================
# PIPELINE CHO ẢNH TĨNH (multi-face)
# ============================================================

def process_image(img_rgb, detector_type="DNN", classifier_type="CNN", conf_threshold=0.5):
    """
    End-to-end pipeline cho ảnh tĩnh — hỗ trợ nhiều khuôn mặt.

    Returns:
        tuple: (result_img_rgb, info_dict)
    """
    t0 = time.time()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Phát hiện TẤT CẢ khuôn mặt
    faces = detect_all_faces(img_bgr, method=detector_type, conf_threshold=conf_threshold)

    elapsed = (time.time() - t0) * 1000

    if not faces:
        output_bgr = draw_no_face(img_bgr)
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        info = {
            "label": "Khong phat hien khuon mat",
            "confidence": 0.0,
            "face_detected": False,
            "face_conf": 0.0,
            "bbox": None,
            "num_faces": 0,
            "elapsed_ms": elapsed,
            "detector": detector_type,
            "classifier": classifier_type,
        }
        return output_rgb, info

    # Phân loại từng khuôn mặt
    face_results = []
    for bbox, face_conf in faces:
        roi_bgr = crop_and_resize(img_bgr, bbox, size=(64, 64))
        result = predict_mask(roi_bgr, classifier_type=classifier_type)
        face_results.append((bbox, result, face_conf))

    output_bgr = draw_multi_face_results(img_bgr, face_results)
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

    # Thông tin khuôn mặt đầu tiên (lớn nhất) làm thông tin chính
    main_bbox, main_result, main_face_conf = face_results[0]

    info = {
        "label": main_result["label"],
        "confidence": main_result["confidence"],
        "face_detected": True,
        "face_conf": main_face_conf,
        "bbox": main_bbox,
        "num_faces": len(face_results),
        "elapsed_ms": elapsed,
        "detector": detector_type,
        "classifier": classifier_type,
    }
    return output_rgb, info


# ============================================================
# PIPELINE CHO VIDEO / WEBCAM STREAM (multi-face)
# ============================================================

def process_frame(frame_bgr, detector_type="DNN", classifier_type="CNN",
                  conf_threshold=0.5, prev_time=None):
    """
    Xử lý một frame video với multi-face support.

    Returns:
        tuple: (result_frame_bgr, curr_time, result_dict)
    """
    t0 = time.time()

    fps = None
    if prev_time is not None:
        dt = t0 - prev_time
        fps = 1.0 / (dt + 1e-9)

    faces = detect_all_faces(frame_bgr, method=detector_type, conf_threshold=conf_threshold)

    if not faces:
        output = draw_no_face(frame_bgr)
        if fps is not None:
            _draw_fps(output, fps)
        dummy_result = {"label_id": -1, "label": "no_face", "confidence": 0.0}
        return output, t0, dummy_result

    # Phân loại từng khuôn mặt
    face_results = []
    for bbox, face_conf in faces:
        roi_bgr = crop_and_resize(frame_bgr, bbox, size=(64, 64))
        result = predict_mask(roi_bgr, classifier_type=classifier_type)
        face_results.append((bbox, result, face_conf))

    output = draw_multi_face_results(frame_bgr, face_results, fps=fps)

    # Thống kê dùng khuôn mặt chính (lớn nhất)
    main_result = face_results[0][1]
    return output, t0, main_result

def _make_video_writer(output_path, fps_video, width, height):
    """Tao VideoWriter, thu cac codec theo thu tu uu tien."""
    # Thu H.264 truoc (browser-compatible), fallback ve mp4v
    for fourcc_str in ['avc1', 'H264', 'X264', 'mp4v']:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    return None


def process_video_file(input_path, output_path, detector_type="DNN",
                       classifier_type="CNN", conf_threshold=0.5,
                       progress_callback=None):
    """
    Xu ly toan bo file video voi multi-face support.

    Returns:
        dict: Thong ke xu ly
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return {"error": f"Khong the mo file video: {input_path}"}

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = _make_video_writer(output_path, fps_video, width, height)
    if out is None:
        cap.release()
        return {"error": "Khong the tao file video dau ra (khong co codec nao hop le)"}

    frame_count = 0
    mask_count = 0
    no_mask_count = 0
    no_face_count = 0
    prev_time = None
    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame, curr_time, result = process_frame(
            frame, detector_type, classifier_type, conf_threshold, prev_time
        )

        if prev_time is not None:
            fps_val = 1.0 / (curr_time - prev_time + 1e-9)
            fps_list.append(fps_val)

        prev_time = curr_time
        out.write(result_frame)
        frame_count += 1

        if result["label_id"] == 1:
            mask_count += 1
        elif result["label_id"] == 0:
            no_mask_count += 1
        else:
            no_face_count += 1

        if progress_callback and total_frames > 0:
            progress = frame_count / total_frames
            progress_callback(progress, f"Frame {frame_count}/{total_frames}...")

    cap.release()
    out.release()

    avg_fps = float(np.mean(fps_list)) if fps_list else 0.0

    return {
        "total_frames": frame_count,
        "mask_frames": mask_count,
        "no_mask_frames": no_mask_count,
        "no_face_frames": no_face_count,
        "avg_fps": avg_fps,
        "output_path": output_path,
    }
