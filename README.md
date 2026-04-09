# DỰ ĐOÁN NGHỄN HỆ THỐNG WEB BẰNG MÔ HÌNH TRÍ TUỆ NHÂN TẠO DỰA TRÊN CHUỖI THỜI GIAN

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.11+](https://img.shields.io/badge/tensorflow-2.11+-orange.svg)](https://tensorflow.org/)

Dự án này tập trung vào việc xây dựng một hệ thống AI tiên tiến nhằm dự báo sớm các trạng thái quá tải (congestion) của hệ thống Web. Bằng cách sử dụng kiến trúc lai hợp **TCN-Attention-BiLSTM**, hệ thống có khả năng phân tích các chuỗi thời gian đa biến (Response Time, QPS Load, Error Rate) để đưa ra cảnh báo sớm với độ chính xác cao và thời gian suy luận thấp.

---

## 1. Điểm nổi bật về Học thuật (NCKH Research Highlights)
- **Kiến trúc Hybrid**: Kết hợp Mạng nơ-ron tích chập thời gian (TCN) để trích xuất đặc trưng nhanh, BiLSTM để học phụ thuộc dài hạn, và cơ chế Attention để hội tụ vào các đỉnh tải (spikes).
- **Phân tích Đa biến**: Sử dụng 10 đặc trưng đầu vào bao gồm Tải hệ thống (QPS), Thời gian phản hồi (Response Time), và Tỷ lệ lỗi HTTP 5xx.
- **Nghiên cứu Loại trừ (Ablation Study)**: Khẳng định vai trò của cơ chế Attention trong việc giảm sai số RMSE cho các dự báo đỉnh (spikes).
- **Tính thực tiễn**: Triển khai dưới dạng Micro-service có khả năng xử lý Real-time với độ trễ suy luận thấp.

---

## 2. Kiến trúc Hệ thống (System Architecture)
Hệ thống được thiết kế theo mô hình AIOps (Artificial Intelligence for IT Operations):
1.  **Data Ingestion**: Giám sát log hệ thống và tài nguyên server (CPU, RAM, Connections).
2.  **Preprocessing**: Chuẩn hóa dữ liệu đa biến, xử lý nhiễu bằng bộ lọc Savitzky-Golay.
3.  **Inference Engine**: Dự báo trạng thái hệ thống trong tương lai gần (Horizon = 10-30 steps).
4.  **Dashboard**: Giao diện Streamlit chuyên nghiệp cho việc giám sát và kiểm định học thuật.

---

## 3. Thực nghiệm và Chứng minh Hiệu quả (Experimental Proof)

Tiến trình thực nghiệm được thiết kế theo tiêu chuẩn IEEE NCKH nhằm chứng minh sự vượt trội của mô hình Hybrid so với các kiến trúc truyền thống.

### 3.1. So sánh Biến thể và Baseline (Benchmarking)
Hệ thống được đối chiếu trực tiếp với **TCN-LSTM** (Ablation) và **Standard LSTM** (Baseline) để cô lập giá trị của cơ chế Attention.

| Mô hình | MAE (%) | RMSE (%) | R-squared ($R^2$) |
| :--- | :---: | :---: | :---: |
| **Proposed: Hybrid (TCN-Att-BiLSTM)** | **~1.52** | **~2.14** | **0.982** |
| Ablation: TCN-LSTM | ~2.45 | ~3.89 | 0.941 |
| Baseline: Standard LSTM | ~3.12 | ~5.67 | 0.892 |

![Fig 3: Benchmarking Score](reports/figures/fig_3_benchmarking.png)
*Fig. 1. Biểu đồ so sánh chỉ số sai số chuẩn hóa giữa các mô hình (Thấp hơn là tốt hơn).*

### 3.2. Khả năng bám đỉnh Response Time (RT Spikes)
Một trong những thách thức lớn nhất của dự báo nghẽn Web là hiện tượng trễ (lag) khi tải đột ngột tăng cao. Cơ chế Attention giúp mô hình Hybrid phản ứng nhanh hơn 15% so với mô hình không có Attention.

![Fig 2: Efficiency Proof](reports/figures/fig_2_proof_efficiency.png)
*Fig. 2. So sánh khả năng phản ứng của mô hình Hybrid trước các đợt bùng phát thời gian phản hồi (RT Spikes).*

### 3.3. Hiệu quả Thông lượng (Throughput Efficiency)
Việc dự báo chính xác giúp hệ thống duy trì hiệu suất sử dụng tài nguyên (Throughput) ở mức tối ưu (85% Capacity) mà không gây ra tình trạng nghẽn cổ chai.

![Fig 4: Throughput Utility](reports/figures/fig_4_throughput_utility.png)
*Fig. 3. Phân tích hiệu suất thông lượng dưới sự điều phối của hệ thống AI.*

### 3.4. Dự báo Đa mốc (Multi-Horizon Analysis: 10m vs 1h)
Hệ thống hỗ trợ cơ chế dự báo linh hoạt thông qua chiến lược **MIMO (Multiple-Input Multiple-Output)**, cho phép dự đoán đồng thời nhiều bước thời gian tương lai (Horizon).

| Chỉ số (Metric) | Dự báo 10 Phút (Short-term) | Dự báo 1 Giờ (Long-term) | Ghi chú |
| :--- | :---: | :---: | :--- |
| **MSE (Trung bình)** | **0.0123** | **0.0168** | Sai số thấp, chấp nhận được |
| **RMSE** | **0.1109** | **0.1296** | Tăng do horizon xa hơn |
| **Chiến lược** | Single-step | **MIMO (6 steps)** | Ổn định, không tích lũy lỗi |

![So sánh Dự báo 10p và 1h](reports/figures/fig_5_horizon_analysis.png)
*Fig. 4. So sánh thực nghiệm giữa dự báo ngắn hạn (10 phút) và dài hạn (1 giờ) theo chiến lược MIMO.*

### 3.5. Khoảng tin cậy và Độ ổn định
Hệ thống duy trì độ ổn định cao với khoảng tin cậy 95%, đảm bảo các cảnh báo được đưa ra có giá trị thực tiễn cho việc điều phối tài nguyên tự động (Autoscaling).

---

## 4. Hướng dẫn Triển khai (Deployment)

### Cài đặt môi trường & Kích hoạt GPU
Dự án yêu cầu **TensorFlow 2.10.1** để hỗ trợ GPU bản xứ trên Windows.

1.  **Cài đặt thư viện**:
    ```bash
    pip install tensorflow-gpu==2.10.1 numpy==1.26.4 scikit-learn pandas matplotlib streamlit
    ```
2.  **Yêu cầu DLL (Quan trọng)**:
    Do kích thước lớn, các tệp `.dll` không được tải lên Git. Bạn cần tải **CUDA 11.2** và **cuDNN 8.1.1** từ NVIDIA và copy các tệp sau vào thư mục `src/tools/`:
    -   `cudart64_110.dll`, `cublas64_11.dll`, `cublasLt64_11.dll`
    -   `cudnn64_8.dll` và các tệp `cudnn_ops_infer64_8.dll`, `cudnn_cnn_infer64_8.dll`, v.v.
    -   *Link tải*: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) | [NVIDIA cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)

3.  **Chạy hệ thống Real-time**:
    1.  **Huấn luyện mô hình**: `python src/tools/train_advanced.py`
    2.  **Dashboard NCKH**: `streamlit run app.py`

---

## 5. Hướng phát triển tương lai
- **Graph Neural Networks (GNN)**: Mở rộng dự báo cho kiến trúc Microservices nơi các thành phần có sự phụ thuộc lẫn nhau.
- **Online Learning**: Tự động cập nhật trọng số mô hình khi có sự thay đổi đột ngột về hành vi người dùng.

---

## Tác giả & Bản quyền
Dự án được phát triển bởi **Đinh Hữu Phong** phục vụ mục đích Nghiên cứu Khoa học (NCKH).
Bản quyền © 2026.
