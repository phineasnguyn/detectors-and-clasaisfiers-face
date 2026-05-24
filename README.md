# Face Mask Detection - Local Web GUI

Ứng dụng web cục bộ (Local Web GUI) sử dụng Gradio để phát hiện khuôn mặt và phân loại việc đeo khẩu trang (Face Mask Detection). 
Hệ thống hỗ trợ nhận diện nhiều khuôn mặt cùng lúc (Multi-face detection) trên ảnh tĩnh, luồng webcam thời gian thực và video tải lên.

## 🚀 Tính năng

- **Nhận diện đa khuôn mặt**: Phát hiện và vẽ bounding box, dự đoán nhãn cho tất cả các khuôn mặt xuất hiện trong khung hình.
- **Phát hiện khuôn mặt (Detectors)**: 
  - **OpenCV DNN**: Nhanh, độ chính xác cao, tự động tải mô hình nếu chưa có.
  - **Dlib HOG**: Rất nhẹ, phù hợp cho cấu hình yếu (yêu cầu cài đặt thư viện dlib).
- **Phân loại khẩu trang (Classifiers)**: Hỗ trợ linh hoạt các mô hình phân loại: `CNN`, `SVM`, `Random Forest`.
- **Đầu vào đa dạng**:
  - **Hình ảnh tĩnh**: Tải ảnh lên và phân tích.
  - **Webcam Real-time**: Stream trực tiếp từ webcam với độ trễ thấp.
  - **Video**: Xử lý file Video (MP4, AVI) tải lên và xuất ra video kết quả (H.264 MP4).

---

## 🛠 Hướng dẫn cài đặt

### 1. Yêu cầu hệ thống
- **Python 3.8+** (khuyến nghị 3.10 hoặc 3.11).
- Tùy chọn: **FFmpeg** để mã hóa video đầu ra tương thích hoàn toàn với các trình duyệt web. 
  *(Trên Windows, có thể cài bằng lệnh: `winget install ffmpeg`)*

### 2. Cài đặt thư viện

Mở Terminal (hoặc PowerShell / Command Prompt) tại thư mục chứa dự án và chạy lệnh:

```bash
pip install -r requirements.txt
```

> **Lưu ý**: Nếu bạn muốn dùng detector HOG, bạn cần cài đặt thêm `dlib` (`pip install dlib`). Quá trình này trên Windows có thể yêu cầu CMake và C++ Build Tools.

### 3. Chuẩn bị mô hình (Models)

Hệ thống cần các mô hình học máy đã được huấn luyện để phân loại. Do kích thước file lớn, các mô hình này không được tải trực tiếp lên mã nguồn. Bạn có thể lấy mô hình bằng một trong hai cách sau:

#### Cách A: Tải mô hình đã được huấn luyện sẵn (Pre-trained)
1. Truy cập vào mục **[Releases](https://github.com/phineasnguyn/detectors-and-clasaisfiers-face/releases)** của kho lưu trữ này (nếu tác giả đã cung cấp).
2. Tải xuống các file mô hình và đặt chúng vào thư mục `models/` ở gốc thư mục dự án:
   - `cnn_model.h5` (hoặc `.keras`)
   - `svm_model.pkl`
   - `rf_model.pkl`

#### Cách B: Tự huấn luyện (Train) mô hình
1. Mở file `detectedfaceandclassification.ipynb` bằng Jupyter Notebook hoặc Google Colab.
2. Chuẩn bị dataset (chia thư mục `with_mask` và `without_mask`).
3. Chạy toàn bộ các ô lệnh (cells) trong Notebook để tiến hành huấn luyện.
4. Sau khi huấn luyện hoàn tất, chạy file `save_models.py` hoặc các ô lệnh lưu model cuối cùng để xuất ra các file `.h5` và `.pkl`.
5. Đưa các file xuất ra vào thư mục `models/`.

*(Lưu ý: Nếu thiếu mô hình, hệ thống sẽ cảnh báo trên giao diện. Các file của mô hình phát hiện khuôn mặt OpenCV DNN sẽ tự động được tải xuống trong lần chạy đầu tiên).*

---

## 🎯 Hướng dẫn chạy chương trình

Sau khi hoàn tất cài đặt, khởi động server bằng lệnh sau:

```bash
python app.py
```

- Lệnh này sẽ khởi động local server sử dụng Gradio.
- Trình duyệt web sẽ tự động mở trang giao diện tại địa chỉ: `http://127.0.0.1:7860`.
- Chuyển đổi giữa các Tab (Upload Ảnh, Webcam Real-time, Xử lý Video) trên giao diện để sử dụng các tính năng.
- **Để dừng server:** Nhấn tổ hợp phím `Ctrl + C` trong Terminal.

---

## 📝 Thông tin dự án
- **Đề tài**: Phát hiện khuôn mặt và phân loại khẩu trang
- **Môn học**: Computer Vision
- **Nhóm**: 21
- **Thành viên**: Hoàng Gia Mạnh
