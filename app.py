"""
╔══════════════════════════════════════════════════════════════╗
║     FACE MASK DETECTION - WEB GUI (Gradio)                   ║
║     Nhóm 21 - Computer Vision                                ║
║     Đề tài: Phát hiện khuôn mặt & Phân loại khẩu trang       ║
╚══════════════════════════════════════════════════════════════╝

Ứng dụng chạy cục bộ (local) với giao diện Gradio.
Chạy: python app.py
Mở trình duyệt tại: http://127.0.0.1:7860
"""

import gradio as gr
import cv2
import numpy as np
import os
import tempfile
import time
import json
from pathlib import Path

from core.pipeline import process_image, process_video_file, process_frame
from core.classifier import get_available_classifiers
from core.detector import get_dnn_net

# ============================================================
# CẤU HÌNH
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

APP_TITLE = " Face Mask Detection"
APP_DESCRIPTION = """
<div style="text-align:center; padding: 8px 0;">
    <span style="font-size:1.1em; color:#a0aec0;">
        Phát hiện khuôn mặt và phân loại khẩu trang theo thời gian thực
    </span><br/>
    <span style="font-size:0.9em; color:#718096;">
        Nhóm 21 · Computer Vision
    </span>
</div>
"""

# CSS tuỳ chỉnh giao diện tối - premium dark theme
CUSTOM_CSS = """
/* ===== GLOBAL ===== */
:root {
    --primary:    #6c63ff;
    --primary-dk: #5a52e0;
    --accent:     #00d4aa;
    --danger:     #ff5e7a;
    --bg-dark:    #0d1117;
    --bg-card:    #161b22;
    --bg-panel:   #1c2230;
    --border:     #30363d;
    --text:       #e6edf3;
    --text-muted: #8b949e;
    --success:    #3fb950;
    --warn:       #d29922;
    --radius:     14px;
    --shadow:     0 8px 32px rgba(0,0,0,0.4);
}

body, .gradio-container {
    background: var(--bg-dark) !important;
    color: var(--text) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ===== HEADER ===== */
.main-header {
    background: linear-gradient(135deg, #1a1f35 0%, #0d1117 50%, #1a1f35 100%);
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    margin-bottom: 0;
}
.header-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(108,99,255,0.15); border: 1px solid rgba(108,99,255,0.4);
    border-radius: 999px; padding: 6px 16px; font-size: 0.82em;
    color: var(--primary); font-weight: 600; letter-spacing: 0.03em;
}

/* ===== TABS ===== */
.tab-nav button {
    background: transparent !important;
    color: var(--text-muted) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-size: 0.95em !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
}
.tab-nav button:hover:not(.selected) {
    color: var(--text) !important;
    background: rgba(108,99,255,0.08) !important;
}

/* ===== PANELS & CARDS ===== */
.panel-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: var(--shadow);
}

/* ===== GRADIO COMPONENTS ===== */
.gr-box, .gr-panel, .gr-form, .gr-block {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
.gradio-image, .gr-image { border-radius: 12px !important; }

label.svelte-1b6s6im, .gr-input-label {
    color: var(--text-muted) !important;
    font-size: 0.82em !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* ===== BUTTONS ===== */
button.primary-btn, .gr-button.primary {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dk) 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1em !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 15px rgba(108,99,255,0.35) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}
button.primary-btn:hover, .gr-button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(108,99,255,0.5) !important;
}
.gr-button.secondary {
    background: var(--bg-panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
.gr-button.secondary:hover {
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}

/* ===== SLIDERS ===== */
input[type=range] { accent-color: var(--primary); }

/* ===== RADIO / DROPDOWN ===== */
.gr-radio input[type=radio]:checked + span,
.gr-dropdown { color: var(--primary) !important; }

/* ===== RESULT INFO BOX ===== */
.result-box {
    background: linear-gradient(135deg, var(--bg-panel) 0%, var(--bg-card) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.88em;
    line-height: 1.8;
    color: var(--text);
}
.result-with-mask  { border-left: 4px solid var(--success); }
.result-no-mask    { border-left: 4px solid var(--danger); }
.result-no-face    { border-left: 4px solid var(--warn); }

/* ===== STAT CHIPS ===== */
.stat-chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.82em;
    font-weight: 600;
    margin: 3px;
}
.chip-green  { background: rgba(63,185,80,0.18);  color: #3fb950; }
.chip-red    { background: rgba(255,94,122,0.18); color: #ff5e7a; }
.chip-blue   { background: rgba(108,99,255,0.18); color: #6c63ff; }
.chip-yellow { background: rgba(210,153,34,0.18); color: #d29922; }

/* ===== VIDEO STAT ===== */
.video-stat-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px;
}
.video-stat-item {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.video-stat-value {
    font-size: 1.8em; font-weight: 700; color: var(--accent); line-height: 1;
}
.video-stat-label {
    font-size: 0.78em; color: var(--text-muted); margin-top: 4px;
}

/* ===== ABOUT / INFO SECTION ===== */
.info-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.info-card h3 { color: var(--primary); margin-top: 0; font-size: 1.05em; }
.model-badge {
    display: inline-block;
    padding: 3px 10px; border-radius: 6px; font-size: 0.8em; font-weight: 600;
    background: rgba(108,99,255,0.2); color: var(--primary);
    border: 1px solid rgba(108,99,255,0.35); margin-left: 8px;
}

/* ===== PROGRESS BAR ===== */
.gr-progress { background: var(--primary) !important; }
.progress-bar-wrap { background: var(--bg-panel) !important; border-radius: 999px !important; }

/* ===== FOOTER ===== */
footer { display: none !important; }

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary); }
"""

# ============================================================
# GRADIO CALLBACK FUNCTIONS
# ============================================================

def run_image_detection(image, detector_type, classifier_type, conf_threshold):
    """Xu ly anh tinh va tra ve ket qua."""
    if image is None:
        return None, "Vui long tai anh len hoac chup tu webcam."

    try:
        result_img, info = process_image(
            image,
            detector_type=detector_type,
            classifier_type=classifier_type,
            conf_threshold=conf_threshold,
        )

        if info["face_detected"]:
            n = info.get("num_faces", 1)
            face_str = f"**{n} khuon mat**" if n > 1 else "**1 khuon mat**"

            if info["label"] == "With Mask":
                verdict = "CO KHAU TRANG (khuon mat lon nhat)"
            else:
                verdict = "KHONG KHAU TRANG (khuon mat lon nhat)"

            info_md = f"""
**{verdict}**

| Thuoc tinh | Gia tri |
|---|---|
| So khuon mat | {face_str} |
| Phan loai (chinh) | **{info['label']}** |
| Do tin cay | **{info['confidence']*100:.1f}%** |
| Phat hien mat | **{info['face_conf']*100:.0f}%** |
| Mo hinh | **{info['classifier']}** |
| Detector | **{info['detector']}** |
| Thoi gian | **{info['elapsed_ms']:.1f} ms** |
"""
        else:
            info_md = """
**KHONG PHAT HIEN KHUON MAT**

Thu:
- Dam bao khuon mat ro rang trong anh
- Giam nguong confidence
- Doi sang detector HOG
"""

        return result_img, info_md

    except Exception as e:
        import traceback
        return image, f"Loi: {str(e)}\n\n{traceback.format_exc()}"


def run_webcam_detection(frame, detector_type, classifier_type, conf_threshold):
    """Xử lý frame từ webcam stream."""
    if frame is None:
        return frame

    try:
        # Gradio 6 truyền numpy array RGB
        if not isinstance(frame, np.ndarray):
            return frame
        img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result_frame, _, result = process_frame(
            img_bgr, detector_type, classifier_type, conf_threshold
        )
        return cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return frame


def _convert_to_h264(input_path, output_path):
    """Convert video sang H.264 dung ffmpeg (neu co san)."""
    try:
        import subprocess
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', input_path,
             '-c:v', 'libx264', '-preset', 'fast',
             '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
             output_path],
            capture_output=True, timeout=600,
        )
        return result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except (FileNotFoundError, Exception):
        return False


def run_video_processing(video_file, detector_type, classifier_type, conf_threshold, progress=gr.Progress()):
    """Xu ly file video upload."""
    if video_file is None:
        return None, "Vui long tai video len."

    # Gradio 6 co the tra ve string path hoac dict
    if isinstance(video_file, dict):
        input_path = (
            video_file.get("video", {}).get("path")
            or video_file.get("path")
            or str(video_file)
        )
    elif hasattr(video_file, "name"):   # file-like object
        input_path = video_file.name
    else:
        input_path = str(video_file)

    if not input_path or not os.path.exists(input_path):
        return None, f"Khong tim thay file video: {input_path!r}"

    ts = int(time.time())
    # File trung gian (mp4v fallback)
    raw_path    = os.path.join(BASE_DIR, f"output_raw_{ts}.mp4")
    # File dau ra cuoi cung (H.264)
    output_path = os.path.join(BASE_DIR, f"output_{ts}.mp4")

    def prog_cb(p, msg):
        progress(p, desc=msg)

    try:
        progress(0, desc="Dang khoi dong...")
        stats = process_video_file(
            input_path,
            raw_path,
            detector_type=detector_type,
            classifier_type=classifier_type,
            conf_threshold=conf_threshold,
            progress_callback=prog_cb,
        )

        if "error" in stats:
            return None, stats['error']

        # Thu convert sang H.264 de trinh duyet phat duoc
        progress(0.95, desc="Dang chuyen doi sang H.264...")
        if _convert_to_h264(raw_path, output_path):
            try:
                os.remove(raw_path)   # xoa file tam
            except OSError:
                pass
            final_path = output_path
        else:
            # ffmpeg khong co, dung thang file mp4v (co the phat duoc tren Chrome)
            final_path = raw_path

        total = stats["total_frames"]
        mask_pct   = (stats["mask_frames"]    / max(total, 1)) * 100
        nomask_pct = (stats["no_mask_frames"] / max(total, 1)) * 100
        noface_pct = (stats["no_face_frames"] / max(total, 1)) * 100

        stat_md = f"""
### Thong ke xu ly video

| Chi so | Gia tri |
|---|---|
| Tong frames | **{total}** |
| Co khau trang | **{stats['mask_frames']}** ({mask_pct:.1f}%) |
| Khong khau trang | **{stats['no_mask_frames']}** ({nomask_pct:.1f}%) |
| Khong thay mat | **{stats['no_face_frames']}** ({noface_pct:.1f}%) |
| FPS trung binh | **{stats['avg_fps']:.1f}** |
| Mo hinh | **{classifier_type}** / **{detector_type}** |
"""
        return final_path, stat_md

    except Exception as e:
        import traceback
        return None, f"Loi xu ly video: {str(e)}\n\n{traceback.format_exc()}"


def get_model_status():
    """Kiểm tra trạng thái các model đã có sẵn."""
    available = get_available_classifiers()

    dnn_ready = (
        os.path.exists(os.path.join(MODELS_DIR, "deploy.prototxt")) and
        os.path.exists(os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel"))
    )

    try:
        import dlib
        dlib_ready = True
    except ImportError:
        dlib_ready = False

    lines = ["###  Trạng thái Model\n"]
    lines.append(f"- **OpenCV DNN** (Face Detector): {' Sẵn sàng' if dnn_ready else ' Sẽ tự tải khi chạy'}")
    lines.append(f"- **Dlib HOG** (Face Detector): {' Sẵn sàng' if dlib_ready else ' Chưa cài dlib'}")
    lines.append("")
    lines.append("**Classifier Models:**")
    for clf in ["CNN", "SVM", "Random Forest"]:
        is_ok = clf in available
        lines.append(f"- **{clf}**: {' Đã có' if is_ok else ' Chưa có file model'}")

    if not available:
        lines.append("\n>  **Chưa có model classifier nào!**")
        lines.append("> Bạn cần train model từ notebook và lưu vào thư mục `models/`.")
        lines.append("> - `models/cnn_model.h5` hoặc `models/cnn_model.keras`")
        lines.append("> - `models/svm_model.pkl`")
        lines.append("> - `models/rf_model.pkl`")
        lines.append("> \n> **Tạm thời bạn vẫn có thể test face detection với model giả lập bên dưới.**")

    return "\n".join(lines)


def load_dnn_model_ui():
    """Tải DNN model (progress indicator)."""
    try:
        get_dnn_net()
        return " DNN Face Detector đã sẵn sàng!"
    except Exception as e:
        return f" Lỗi: {str(e)}"


# ============================================================
# XÁC ĐỊNH CLASSIFIER OPTIONS THEO MODEL CÓ SẴN
# ============================================================
def get_classifier_choices():
    available = get_available_classifiers()
    all_choices = ["CNN", "SVM", "Random Forest"]
    # Nếu không có model, vẫn show nhưng sẽ báo lỗi
    return all_choices if available else all_choices


# ============================================================
# XÂY DỰNG GIAO DIỆN GRADIO
# ============================================================
def build_app():
    available_classifiers = get_classifier_choices()
    default_clf = available_classifiers[0] if available_classifiers else "CNN"

    # Build theme cho Gradio 6
    theme = gr.themes.Base(
        primary_hue="violet",
        secondary_hue="teal",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0d1117",
        body_text_color="#e6edf3",
        block_background_fill="#161b22",
        block_border_color="#30363d",
        input_background_fill="#1c2230",
        button_primary_background_fill="*primary_500",
        button_primary_text_color="white",
    )

    with gr.Blocks(
        css=CUSTOM_CSS,
        title="Face Mask Detection — Nhom 21",
        theme=theme,
    ) as demo:


        # ── HEADER ──────────────────────────────────────────────
        gr.HTML(f"""
        <div class="main-header">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
                <div>
                    <h1 style="margin:0; font-size:1.8em; font-weight:800;
                               background: linear-gradient(135deg, #6c63ff, #00d4aa);
                               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                         Face Mask Detection
                    </h1>
                    <p style="margin:6px 0 0; color:#8b949e; font-size:0.9em;">
                        Phát hiện khuôn mặt và phân loại khẩu trang — Nhóm 21
                    </p>
                </div>
                <div style="display:flex; gap:8px; flex-wrap:wrap;">
                    <span class="header-badge"> CNN · SVM · Random Forest</span>
                    <span class="header-badge"> OpenCV DNN · Dlib HOG</span>
                    <span class="header-badge"> Real-time</span>
                </div>
            </div>
        </div>
        """)

        # ── MAIN TABS ────────────────────────────────────────────
        with gr.Tabs(elem_classes="main-tabs"):

            # ====================================================
            # TAB 1: ẢNH TĨNH
            # ====================================================
            with gr.TabItem(" Ảnh tĩnh", id="tab_image"):
                with gr.Row(equal_height=False):

                    # LEFT: Controls
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown("###  Cài đặt")

                        detector_radio_img = gr.Radio(
                            choices=["DNN", "HOG"],
                            value="DNN",
                            label="Face Detector",
                            info="DNN: Nhanh & chính xác · HOG: Dlib sliding window"
                        )
                        classifier_radio_img = gr.Radio(
                            choices=available_classifiers,
                            value=default_clf,
                            label="Classifier",
                            info="Mô hình phân loại khẩu trang"
                        )
                        conf_slider_img = gr.Slider(
                            minimum=0.1, maximum=0.99, value=0.5, step=0.05,
                            label="Ngưỡng confidence (Face Detection)",
                            info="Thấp hơn = dễ phát hiện hơn"
                        )

                        gr.Markdown("---")
                        detect_btn = gr.Button(" Nhận diện", variant="primary", size="lg")
                        clear_btn = gr.Button(" Xóa", variant="secondary", size="sm")

                        gr.Markdown("---")
                        result_info = gr.Markdown(
                            value="*Kết quả sẽ xuất hiện ở đây...*",
                            label=""
                        )

                    # RIGHT: Image I/O
                    with gr.Column(scale=2):
                        with gr.Row():
                            input_image = gr.Image(
                                label=" Ảnh đầu vào",
                                sources=["upload", "webcam", "clipboard"],
                                type="numpy",
                                height=380,
                            )
                            output_image = gr.Image(
                                label=" Kết quả",
                                type="numpy",
                                height=380,
                                interactive=False,
                            )

                        gr.Examples(
                            examples=[],  # Sẽ thêm nếu có ảnh mẫu
                            inputs=input_image,
                            label="Ảnh mẫu (nếu có)",
                        )

                # Events
                detect_btn.click(
                    fn=run_image_detection,
                    inputs=[input_image, detector_radio_img, classifier_radio_img, conf_slider_img],
                    outputs=[output_image, result_info],
                )
                input_image.change(
                    fn=run_image_detection,
                    inputs=[input_image, detector_radio_img, classifier_radio_img, conf_slider_img],
                    outputs=[output_image, result_info],
                )
                clear_btn.click(lambda: (None, None, "*Kết quả sẽ xuất hiện ở đây...*"),
                                outputs=[input_image, output_image, result_info])

            # ====================================================
            # TAB 2: WEBCAM REAL-TIME
            # ====================================================
            with gr.TabItem(" Webcam Real-time", id="tab_webcam"):
                gr.Markdown("""
>  **Hướng dẫn**: Nhấn nút **Start** để bắt đầu stream webcam.
> Camera sẽ liên tục phân tích và hiển thị kết quả theo thời gian thực.
""")
                with gr.Row(equal_height=False):

                    # Controls
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown("###  Cài đặt")

                        detector_radio_cam = gr.Radio(
                            choices=["DNN", "HOG"],
                            value="DNN",
                            label="Face Detector",
                        )
                        classifier_radio_cam = gr.Radio(
                            choices=available_classifiers,
                            value=default_clf,
                            label="Classifier",
                        )
                        conf_slider_cam = gr.Slider(
                            minimum=0.1, maximum=0.99, value=0.45, step=0.05,
                            label="Ngưỡng confidence",
                        )
                        gr.Markdown("""
---
** Lưu ý:**
- DNN + CNN: Chính xác nhất
- HOG + SVM: Nhanh nhất
- Cần ánh sáng tốt để phát hiện mặt chính xác
""")

                    # Webcam component
                    with gr.Column(scale=2):
                        webcam_input = gr.Image(
                            label="Camera (Live)",
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                            height=380,
                        )
                        webcam_result = gr.Image(
                            label="Ket qua phan tich",
                            type="numpy",
                            height=380,
                            interactive=False,
                        )

                webcam_input.stream(
                    fn=run_webcam_detection,
                    inputs=[webcam_input, detector_radio_cam, classifier_radio_cam, conf_slider_cam],
                    outputs=webcam_result,
                    time_limit=60,
                    stream_every=0.1,
                )

            # ====================================================
            # TAB 3: XỬ LÝ VIDEO
            # ====================================================
            with gr.TabItem(" Xử lý Video", id="tab_video"):
                gr.Markdown("""
>  Upload file video (MP4, AVI, MOV...) — Hệ thống sẽ xử lý từng frame và xuất video kết quả.
""")
                with gr.Row(equal_height=False):

                    # Controls
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown("###  Cài đặt")

                        detector_radio_vid = gr.Radio(
                            choices=["DNN", "HOG"],
                            value="DNN",
                            label="Face Detector",
                        )
                        classifier_radio_vid = gr.Radio(
                            choices=available_classifiers,
                            value=default_clf,
                            label="Classifier",
                        )
                        conf_slider_vid = gr.Slider(
                            minimum=0.1, maximum=0.99, value=0.5, step=0.05,
                            label="Ngưỡng confidence",
                        )
                        gr.Markdown("---")
                        process_btn = gr.Button(" Bắt đầu xử lý", variant="primary", size="lg")

                        gr.Markdown("---")
                        video_stats = gr.Markdown("*Thống kê sẽ hiển thị sau khi xử lý xong.*")

                    # Video I/O
                    with gr.Column(scale=2):
                        video_input = gr.Video(
                            label="Video dau vao (MP4, AVI, MOV...)",
                            height=300,
                            format=None,   # Khong ep convert khi upload (tranh loi ffmpeg)
                        )
                        video_output = gr.Video(
                            label="Video ket qua",
                            height=300,
                            interactive=False,
                            format=None,
                        )

                process_btn.click(
                    fn=run_video_processing,
                    inputs=[video_input, detector_radio_vid, classifier_radio_vid, conf_slider_vid],
                    outputs=[video_output, video_stats],
                )

            # ====================================================
            # TAB 4: THÔNG TIN & HƯỚNG DẪN
            # ====================================================
            with gr.TabItem(" Thông tin", id="tab_info"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("""
##  Giới thiệu đề tài

Ứng dụng này là **GUI trực quan** cho đồ án môn học **Computer Vision** về bài toán:

> **Phát hiện khuôn mặt và phân loại khẩu trang** (Face Detection & Mask Classification)

###  Pipeline xử lý

```
Ảnh đầu vào
    ↓
[1] Phát hiện khuôn mặt (Face Detection)
    ├── OpenCV DNN (ResNet SSD 300x300)
    └── Dlib HOG (Sliding Window + Image Pyramid)
    ↓
[2] Tiền xử lý (Preprocessing)
    └── Bilateral Filter → Grayscale (cho SVM/RF)
    ↓
[3] Trích xuất đặc trưng
    ├── HOG Features → SVM / Random Forest
    └── RGB Image (64×64) → CNN
    ↓
[4] Phân loại (Classification)
    ├── Linear SVM
    ├── Random Forest (100 trees)
    └── CNN (16→32→64 filters)
    ↓
Kết quả: With Mask / Without Mask + Bounding Box
```
""")

                    with gr.Column(scale=1):
                        gr.Markdown("##  Trạng thái hệ thống")
                        model_status_md = gr.Markdown(get_model_status())

                        gr.Markdown("---")
                        refresh_btn = gr.Button(" Làm mới trạng thái", variant="secondary")
                        load_dnn_btn = gr.Button(" Tải DNN Model (nếu chưa có)", variant="secondary")
                        dnn_status = gr.Textbox(label="Trạng thái DNN", interactive=False, value="")

                        refresh_btn.click(fn=get_model_status, outputs=model_status_md)
                        load_dnn_btn.click(fn=load_dnn_model_ui, outputs=dnn_status)

                gr.Markdown("---")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
##  Cách thêm Model đã train

Sau khi train xong trên Jupyter Notebook, lưu model vào thư mục `models/`:

```python
# Sau khi train xong trong notebook:
import joblib
from tensorflow.keras.models import save_model

# Lưu SVM
joblib.dump(svm_model, 'models/svm_model.pkl')

# Lưu Random Forest
joblib.dump(rf_model, 'models/rf_model.pkl')

# Lưu CNN
cnn_model.save('models/cnn_model.h5')
# Hoặc: cnn_model.save('models/cnn_model.keras')
```

Sau đó **khởi động lại app** để nạp model mới.
""")

                    with gr.Column():
                        gr.Markdown("""
##  So sánh hiệu năng (từ notebook)

| Mô hình | Accuracy | FPS ước tính |
|---|---|---|
| **CNN** | ~98%+ | ~30-60 FPS |
| **SVM** (HOG) | ~95%+ | ~100+ FPS |
| **Random Forest** (HOG) | ~93%+ | ~80+ FPS |

###  Face Detector

| Detector | Ưu điểm | Nhược điểm |
|---|---|---|
| **DNN** | Chính xác, xử lý mặt đeo khẩu trang tốt | Chậm hơn HOG |
| **HOG (Dlib)** | Rất nhanh | Recall thấp hơn với mặt đeo khẩu trang |

---
*Nhóm 21 · Hoàng Gia Mạnh · Computer Vision · 2024*
""")

        # ── FOOTER ──────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding:16px; color:#4a5568; font-size:0.82em;
                    border-top:1px solid #30363d; margin-top:16px;">
             Face Mask Detection GUI &nbsp;·&nbsp;
            Nhóm 21 · Hoàng Gia Mạnh &nbsp;·&nbsp;
            Powered by Gradio + OpenCV + Keras
        </div>
        """)

    return demo


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import sys
    # Fix encoding cho Windows console
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 60)
    print("  FACE MASK DETECTION - LOCAL WEB GUI")
    print("  Nhom 21 - Computer Vision")
    print("=" * 60)
    print()

    # Kiểm tra model sẵn có
    available = get_available_classifiers()
    if available:
        print(f"[OK] Classifier models co san: {', '.join(available)}")
    else:
        print("[WARNING] Chua co classifier model nao trong thu muc models/")
        print("   -> Hay train va luu model tu notebook vao models/")
        print("   -> Ban van co the chay app de test Face Detection")

    print()
    print("[START] Khoi dong server...")
    print("   Trinh duyet se mo tu dong tai: http://127.0.0.1:7860")
    print()
    print("   Nhan Ctrl+C de dung server")
    print("=" * 60)

    demo = build_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,        # Tự động mở trình duyệt
        share=False,           # Không tạo link public
        show_error=True,
        quiet=False,
    )
