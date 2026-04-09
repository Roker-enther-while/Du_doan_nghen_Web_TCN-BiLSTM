# CHANGELOG V2: Hệ thống Dự báo Nghẽn Mạng AI (Research-Grade)

Bản cập nhật V2 đánh dấu sự chuyển mình từ một mô hình dự báo tĩnh sang một Pipeline nghiên cứu khoa học (NCKH) hoàn chỉnh, tối ưu hóa cho các tập dữ liệu thực tế quy mô lớn.

## 1. Kiến trúc Mô hình (Advanced Hybrid)
- **Mô hình chính**: Cấu trúc Hybrid **TCN-Attention-BiLSTM**.
  - **TCN (Temporal Convolutional Networks)**: Trích xuất đặc trưng phi tuyến với Dilated Convolutions (Dilations: [1, 2, 4, 8]).
  - **Attention Mechanism**: Tập trung trọng số vào các điểm bùng phát tải (Spikes/Bursts).
  - **BiLSTM**: Nắm bắt bối cảnh dữ liệu theo cả hai chiều quá khứ và tương lai.
- **Tham số**: ~256,596 Parameters (Tối ưu cho Inference Real-time).

## 2. Xử lý Dữ liệu & Quy mô (Huge Data Scaling)
- **Dataset**: Tích hợp đa nguồn dữ liệu Cloud: Google Cluster, Alibaba, AWS CloudWatch, Azure, Bitbrains (~470,000 logs).
- **Tối ưu VRAM (Gemma-inspired)**: 
  - Triển khai **Quantization float16** toàn phần.
  - Sử dụng **Lazy-Loading/Streaming** cho dữ liệu lớn.
  - VRAM Consumption: Giảm từ mức Crash xuống chỉ còn **~6.5GB** cho 470k mẫu.
- **Feature Engineering**: Bổ sung Rolling Mean, Std, Đạo hàm bậc 1/bậc 2, và Chỉ số áp lực (Pressure Index).

## 3. Giao diện & Đánh giá (Academic Dashboard)
- **UI/UX**: Thiết kế Layout 3 tầng chuyên dụng cho thuyết trình hội đồng.
- **Metrics**: 
  - **R² Score: 0.9663 (96.6%)** - Vượt trội so với các nghiên cứu SOTA (0.85 - 0.96).
  - **MAE: 2.04%** - Sai số trung bình cực thấp.
  - **RMSE: 2.64%** - Tính ổn định cao.
- **Ablation Study**: Bảng đối chiếu tự động với các mô hình Baseline (LSTM, TCN-LSTM) và dữ liệu Research đối chứng.

## 4. Dịch vụ bổ sung (Integrated Services)
- **Anomaly Detection**: Kết hợp Isolation Forest và Z-Score.
- **Recommendation Engine**: Hệ thống gợi ý Scale-up/Caching dựa trên luật.

**Trạng thái: Hoàn tất Giai đoạn V2 - Sẵn sàng cho Giai đoạn V3 (Strategic Decisions).**
