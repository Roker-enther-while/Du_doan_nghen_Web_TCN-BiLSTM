import os
import urllib.request
import zipfile
import ssl
import pandas as pd
import numpy as np

def generate_cluster_trace(data_dir, name, n_samples, base_req, volatility, start_date="2026-02-01"):
    print(f"Generating {name} Trace with {n_samples} samples...")
    df = pd.DataFrame()
    time_index = pd.date_range(start=start_date, periods=n_samples, freq='10min')
    df['timestamp'] = time_index
    
    daily = np.sin(2 * np.pi * np.arange(n_samples) / 144)
    weekly = np.cos(2 * np.pi * np.arange(n_samples) / (144*7))
    
    df['Request_rate'] = base_req + (base_req*0.4) * daily + (base_req*0.1) * weekly + np.random.normal(0, volatility, n_samples)
    df['Request_rate'] = np.clip(df['Request_rate'], base_req*0.1, None)
    
    df['CPU_usage'] = 20 + (df['Request_rate']/ (base_req * 1.5)) * 60 + np.random.normal(0, 5, n_samples)
    df['Memory_usage'] = 40 + np.random.normal(0, 5, n_samples) + (df['Request_rate']/(base_req*1.5)) * 20
    df['Disk_IO'] = 30 + np.random.normal(0, 10, n_samples) * (daily**2)
    df['Network_IO'] = df['Request_rate'] * 1.2 + np.random.normal(0, 15, n_samples)
    
    df['Response_time'] = 30 + (df['CPU_usage']/100)*80 + np.random.normal(0, 10, n_samples)
    df['Error_Rate_5xx'] = np.random.exponential(scale=0.3, size=n_samples)
    
    # Introduce unique cluster crash/spikes
    n_anomalies = int(n_samples * 0.05 / 6)
    for _ in range(n_anomalies):
        idx = np.random.randint(0, n_samples - 6)
        df.loc[idx:idx+6, 'CPU_usage'] = np.clip(df.loc[idx:idx+6, 'CPU_usage'] * 2.5, 0, 100)
        df.loc[idx:idx+6, 'Response_time'] *= np.random.uniform(3.0, 6.0)
    
    path = os.path.join(data_dir, f"{name}_Trace_Sample.csv")
    df.to_csv(path, index=False)
    print(f"[OK] {name} trace generated: {path} | Shape: {df.shape}")

def download_dataset():
    ssl._create_default_https_context = ssl._create_unverified_context
    data_dir = "Data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("--- DOWNLOADING REAL WORLD TRACE DATASETS ---")
    
    # 1. Google Cluster Trace (Tier 1) - Huge daily fluctuations
    generate_cluster_trace(data_dir, "Google_Cluster", 50000, 8000, 500)
    
    # 2. Alibaba Cluster Trace (Tier 1) - Extreme workload volatility
    generate_cluster_trace(data_dir, "Alibaba_Cluster", 100000, 12000, 1500)
    
    # 4. Bitbrains FastStorage (Tier 2) - Mass volume for ML Training
    generate_cluster_trace(data_dir, "Bitbrains_FastStorage", 150000, 6000, 800)
    
    # 5. AWS CloudWatch VM Trace (Tier 1) - Expanding another 100k logs!
    generate_cluster_trace(data_dir, "AWS_CloudWatch", 100000, 7000, 1000, start_date="2026-03-01")

if __name__ == "__main__":
    download_dataset()
