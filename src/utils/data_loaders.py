import pandas as pd
import os
import numpy as np

class UniversalDataLoader:
    """
    Dịch vụ nạp dữ liệu đa định dạng (v4)
    Hỗ trợ: CSV, JSON, Excel
    Mục tiêu: Đảm bảo dữ liệu đầu vào luôn khớp với cấu hình 10-features của mô hình.
    """
    def __init__(self):
        pass

    def load(self, file_path):
        if not os.path.exists(file_path):
            return None
        
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.csv':
                df = pd.read_csv(file_path)
            elif ext == '.json':
                df = pd.read_json(file_path)
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                return None
        except Exception:
            return None

        # 1. Đảm bảo tên cột chính là 'value'
        if 'value' not in df.columns:
            # Lấy cột số đầu tiên không phải index/timestamp
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df.rename(columns={numeric_cols[0]: 'value'}, inplace=True)
        
        # 2. Phase 10 FINAL: Academic Alignment (Web Server Context)
        # Chuyển đổi toàn bộ thuật ngữ Network sang Web System
        rename_map = {
            'Latency': 'Response_Time',
            'Provider_Latency': 'Response_Time',
            'provider_latency': 'Response_Time',
            'Packet_Loss': 'Error_Rate_5xx',
            'packet_loss': 'Error_Rate_5xx',
            'Value': 'value'
        }
        df = df.rename(columns=rename_map)
        
        # 3. Đảm bảo các cột đặc trưng bắt buộc hiện diện
        if 'Response_Time' not in df.columns:
            df['Response_Time'] = 0.0
        if 'Error_Rate_5xx' not in df.columns:
            df['Error_Rate_5xx'] = 0.0
        
        return df

    def mix_real_synthetic(self, real_df, synthetic_df, ratio=0.7):
        """
        Giai đoạn 1.3: Tạo bộ kết hợp mix data: 70% Real + 30% Synthetic.
        """
        if real_df is None or synthetic_df is None:
            return real_df if synthetic_df is None else synthetic_df
            
        n_real = int(len(real_df) * ratio)
        n_synth = int(len(real_df) * (1 - ratio))
        
        # Lấy random samples để mix
        real_sample = real_df.sample(n=min(n_real, len(real_df)))
        synth_sample = synthetic_df.sample(n=min(n_synth, len(synthetic_df)))
        
        # Nối và sort lại theo timestamp nếu có
        mixed_df = pd.concat([real_sample, synth_sample]).reset_index(drop=True)
        if 'timestamp' in mixed_df.columns:
            mixed_df['timestamp'] = pd.to_datetime(mixed_df['timestamp'])
            mixed_df = mixed_df.sort_values('timestamp').reset_index(drop=True)
            
        return mixed_df
