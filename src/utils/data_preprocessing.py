import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

def compute_pressure_index(cpu, request, r_time):
    # Normalize locally to avoid massive spikes
    cpu_n = (cpu - np.min(cpu)) / (np.max(cpu) - np.min(cpu) + 1e-7)
    req_n = (request - np.min(request)) / (np.max(request) - np.min(request) + 1e-7)
    rt_n = (r_time - np.min(r_time)) / (np.max(r_time) - np.min(r_time) + 1e-7)
    
    cpu_pressure = cpu_n * req_n
    latency_pressure = rt_n * req_n
    return cpu_pressure, latency_pressure

def prepare_data_v2(
    df, window_size=50, horizon=5, train_ratio=0.8, filter_noise=True
):
    """
    Giai đoạn 2 & 3: V2 Upgrade Data Preprocessing & Feature Engineering
    - Missing value interpolation, noise filtering
    - Statistical, Trend, and Pressure Features
    - Output tensor: (batch_size, 50, features)
    """
    df = df.copy()
    
    # Giai đoạn 2.1: Interpolation
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.fillna(0) # Fallback

    # Khai báo các cột cốt lõi
    for col in ['CPU_usage', 'Response_time', 'Request_rate']:
        if col not in df.columns:
            # Fake column if not exist to prevent crash
            df[col] = df.get('value', df.iloc[:, 1] if len(df.columns) > 1 else np.zeros(len(df)))

    cpu_raw = df['CPU_usage'].values.astype(float)
    rt_raw = df['Response_time'].values.astype(float)
    req_raw = df['Request_rate'].values.astype(float)
    err_raw = df.get('Error_Rate_5xx', df.get('Error_rate', pd.Series(np.zeros(len(df))))).values

    # Giai đoạn 2.1: Lọc nhiễu Savitzky-Golay & Outlier smoothing
    if filter_noise and savgol_filter is not None:
        if len(cpu_raw) > 11:
            cpu_raw = savgol_filter(cpu_raw, window_length=11, polyorder=3)
            rt_raw = savgol_filter(rt_raw, window_length=11, polyorder=3)
            req_raw = savgol_filter(req_raw, window_length=11, polyorder=3)

    cpu_raw = np.clip(cpu_raw, 0, None)
    
    # Giai đoạn 2.2: Standard/MinMax Scaler tích hợp
    cpu_scaler = MinMaxScaler(feature_range=(0, 1))
    rt_scaler = MinMaxScaler(feature_range=(0, 1))
    req_scaler = MinMaxScaler(feature_range=(0, 1))
    
    cpu_n = cpu_scaler.fit_transform(cpu_raw.reshape(-1, 1)).flatten()
    rt_n = rt_scaler.fit_transform(rt_raw.reshape(-1, 1)).flatten()
    req_n = req_scaler.fit_transform(req_raw.reshape(-1, 1)).flatten()
    err_n = (err_raw - np.min(err_raw)) / (np.max(err_raw) - np.min(err_raw) + 1e-7)

    # Giai đoạn 3.1 & 3.2: Statistical & Trend Features (Feature Engineering)
    cpu_series = pd.Series(cpu_n)
    
    cpu_ma10 = cpu_series.rolling(10, min_periods=1).mean().values
    cpu_std30 = cpu_series.rolling(30, min_periods=1).std().fillna(0).values
    
    # First derivative (Tốc độ thay đổi)
    cpu_vel = cpu_series.diff(1).fillna(0).values
    # Second derivative (Gia tốc)
    cpu_accel = pd.Series(cpu_vel).diff(1).fillna(0).values

    # Giai đoạn 3.3: Pressure Features
    # Resource pressure index: CPU * Request_rate
    # Latency pressure score: Response_time * Request_rate
    cpu_pressure = cpu_n * req_n
    latency_pressure = rt_n * req_n

    # Temporal features
    dt_series = None
    if "timestamp" in df.columns:
        dt_series = pd.to_datetime(df["timestamp"], errors='coerce')
    elif isinstance(df.index, pd.DatetimeIndex):
        dt_series = df.index.to_series()

    N = len(df)
    if dt_series is not None and not dt_series.isnull().all():
        dt_series = dt_series.fillna(method='bfill').fillna(method='ffill')
        day_of_week = (dt_series.dt.dayofweek / 6.0).values
        hour = (dt_series.dt.hour / 23.0).values
        is_weekend = dt_series.dt.dayofweek.isin([5, 6]).astype(float).values
    else:
        day_of_week = np.zeros(N)
        hour = np.zeros(N)
        is_weekend = np.zeros(N)
        
    # Gộp toàn bộ Features & ÁP DỤNG QUÁ TRÌNH LƯỢNG TỬ HOÁ (Quantization xuống float16)
    # Giảm 75% dung lượng RAM (Tương đương triết lý nén TurboQuant của Gemma)
    all_features = np.column_stack([
        cpu_n, rt_n, req_n, err_n,                  # Base Normalized
        cpu_ma10, cpu_std30,                        # Statistical (3.1)
        cpu_vel, cpu_accel,                         # Trend (3.2)
        cpu_pressure, latency_pressure,             # Pressure (3.3)
        day_of_week, hour, is_weekend               # Temporal
    ]).astype(np.float16)
    
    # Xây dựng ma trận Target
    congestion_prob = np.where(cpu_raw > 85.0, 1.0, np.clip(cpu_raw/100.0, 0, 1))
    all_targets = np.column_stack([cpu_n, rt_n, req_n, congestion_prob]).astype(np.float16)

    # Nếu dữ liệu cực lớn (>100k dòng), Ưu tiên trả về Lazy Loading Dataset thay vì tạo Array tĩnh
    train_size = int(len(all_features) * train_ratio)
    
    scalers_dict = {
        'cpu': cpu_scaler,
        'rt': rt_scaler,
        'req': req_scaler
    }
    
    # Nén dữ liệu xuống chuẩn float16 (Gemma Quantization philosophy) -> Gọn 75% RAM
    X, y = [], []
    for i in range(window_size, len(all_features) - horizon + 1):
        X.append(all_features[i - window_size : i, :])
        y.append(all_targets[i : i + horizon, :])

    X = np.array(X, dtype=np.float16)
    y = np.array(y, dtype=np.float16)
    
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    scalers_dict = {
        'cpu': cpu_scaler,
        'rt': rt_scaler,
        'req': req_scaler
    }

    return X_train, y_train, X_test, y_test, scalers_dict
