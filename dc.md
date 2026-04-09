# ĐỀ TÀI: DỰ ĐOÁN NGHẼN HỆ THỐNG WEB BẰNG MÔ HÌNH TRÍ TUỆ NHÂN TẠO DỰA TRÊN CHUỖI THỜI GIAN

## 8. Tính cấp thiết của đề tài
Trong thời đại chuyển đổi số hiện nay, các hệ thống Web ngày càng phải xử lý lượng lớn dữ liệu và phục vụ hàng triệu lượt truy cập đồng thời. Sự gia tăng đột biến của lưu lượng truy cập (ví dụ: các chiến dịch khuyến mãi, tấn công DDoS hoặc tin tức sự kiện) thường dẫn đến tình trạng quá tải (nghẽn mạng), làm tăng thời gian phản hồi (Response Time), lỗi gián đoạn dịch vụ (Error Rate) và gây thất thoát nghiêm trọng về kinh tế cũng như trải nghiệm người dùng. Việc phát hiện sự cố sau khi nó đã xảy ra thông qua các công cụ giám sát truyền thống là chưa đủ để đảm bảo tính sẵn sàng cao (High Availability). Do đó, việc nghiên cứu một mô hình dự báo trước khả năng xảy ra nghẽn hệ thống Web dựa trên dữ liệu chuỗi thời gian là cực kỳ cấp thiết, giúp các nhà quản trị mạng tự động điều phối tài nguyên (Auto-scaling) kịp thời và chủ động ứng phó.

## 9. Mục tiêu của đề tài
### 9.1. Mục tiêu tổng quát
Phát triển và hoàn thiện công cụ phát hiện, dự báo sớm tình trạng nghẽn hệ thống Web (Web Congestion) bằng việc kết hợp các mạng học sâu tiên tiến (Deep Learning) bao gồm: TCN (Temporal Convolutional Network), cơ chế Attention và BiLSTM (Bidirectional Long Short-Term Memory). 

### 9.2. Mục tiêu cụ thể
- Nghiên cứu và tối ưu hóa kiến trúc mạng lai (Hybrid Architecture) nhằm trích xuất đặc trưng không gian (TCN) và chuỗi thời gian (BiLSTM kết hợp Attention).
- Xây dựng 파ipeline xử lý dữ liệu tự động từ các chỉ số đo lường hiệu suất Web như QPS, Response Time, Error Rate.
- Xây dựng mô hình với chiến lược dự báo đa mốc (Multiple-Input Multiple-Output - MIMO) để dự báo tình trạng trong 1 giờ hoặc 1 ngày tiếp theo.
- Tạo báo cáo học thuật và tài liệu hóa quá trình cài đặt, kết quả thực nghiệm đạt tiêu chuẩn nghiên cứu khoa học.

## 10. Tổng quan tình hình nghiên cứu liên quan đến đề tài
- Nghiên cứu về các công cụ dự báo chuỗi thời gian: Các nghiên cứu trước đây thường sử dụng các mô hình thống kê ARIMA/SARIMA, tuy nhiên không còn phụ hợp với đặc tính biến động phi tuyến tính cao của lưu lượng hệ thống hiện đại.
- Nghiên cứu về AI và Machine Learning dự báo: Các nghiên cứu áp dụng mô hình LSTM thuần túy thường mất nhiều thời gian huấn luyện và không bắt được nhanh các biến dạng đỉnh (spikes).
- Những đề xuất gần đây: Sự ra đời của kiến trúc TCN và cơ chế Attention đã mang lại những cải tiến vượt trội khi kết hợp chúng lại với nhau (hybrid models), giúp mở rộng "trường tiếp nhận" dữ liệu và ghi nhớ đặc tính chu kỳ một cách tối ưu. Tuy nhiên, việc áp dụng cụ thể kiến trúc TCN-Attention-BiLSTM vào bối cảnh dự báo nghẽn hệ thống Web với các đặc trưng QPS và Response Time chưa có nhiều sản phẩm đóng gói hoàn thiện để ứng dụng thực tế. 

## 11. Công cụ nghiên cứu
- **Ngôn ngữ lập trình:** Python.
- **Thư viện chính:**
  - TensorFlow & Keras: Thiết lập và huấn luyện mô hình (Model, Input, Conv1D, Bidirectional, LSTM, Dense).
  - SkLearn, Numpy, Pandas: Tiền xử lý, trích xuất dữ liệu, chia tập Train/Test.
  - SciPy (Savgol_filter): Làm mượt dữ liệu (smoothing data) không làm mất các đặc trưng nhiễu đỉnh đột biến (spikes).
- **Môi trường:** IDE (VS Code), Git/GitHub (Quản lý phiên bản), Hệ thống GPU hỗ trợ tăng tốc độ huấn luyện.

## 12.2. Tiến độ thực hiện: Các nội dung, công việc thực hiện
1. **Tuần 1 - Tuần 2:** Nghiên cứu tổng quan cơ sở lý thuyết về TCN, Attention và BiLSTM. Xây dựng tài liệu kiến trúc.
2. **Tuần 3 - Tuần 4:** Thu thập dữ liệu mô phỏng tình trạng quá tải hệ thống; làm sạch và đóng gói dữ liệu (Data Pipeline).
3. **Tuần 5 - Tuần 7:** Thiết kế mô hình, thử nghiệm các hyperparameter, huấn luyện (Train) mô hình với chiến lược MIMO.
4. **Tuần 8 - Tuần 9:** Đánh giá độ chính xác (Accuracy), tối ưu hóa GPU, tinh chỉnh bộ lọc dữ liệu (savgol_filter) để tăng tính ổn định của mô hình.
5. **Tuần 10 - Tuần 11:** Tích hợp mô hình vào các hệ thống Dashboard dự báo, thực hiện quy trình dự báo thời gian thực (Inference).
6. **Tuần 12:** Tổng kết số liệu thực nghiệm, hoàn thiện các tài liệu báo cáo nghiên cứu học thuật và nghiệm thu đề tài.

## 13. Sản phẩm và khả năng ứng dụng
### 13.1. Sản phẩm của đề tài
- **Mã nguồn hoàn chỉnh (Source Code):** Hệ thống Data Pipeline và AI Engine TCN-Attention-BiLSTM (Train & Inference scripts) để dự đoán Load/Congestion hệ thống Web.
- **Mô hình AI đóng gói:** Các file trọng số/model (models files) đã được huấn luyện sẵn, hoạt động độc lập và ổn định.
- **Tài liệu nghiên cứu:** Đề cương chi tiết, file báo cáo học thuật tổng hợp kiến trúc hệ thống hướng dẫn chạy đồ án.

### 13.2. Khả năng ứng dụng
- Tích hợp trực tiếp vào các hệ thống giám sát server/mạng thực tế (như Prometheus/Grafana, Zabbix) để cảnh báo sớm tình trạng nghẽn server.
- Làm cơ sở module thiết yếu (trigger system) cho các hệ thống điều phối điện toán đám mây (Cloud Auto-Scaling) trong việc tự động scale-up / scale-out tài nguyên trước lúc xảy ra quá tải đỉnh điểm.
- Hỗ trợ cho các nhà nghiên cứu, quản trị viên có thêm công cụ tham khảo cấu trúc dự đoán chuỗi thời gian dựa trên các bộ dữ liệu bất kỳ khác.
