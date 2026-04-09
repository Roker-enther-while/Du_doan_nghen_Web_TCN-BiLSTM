import numpy as np
import pandas as pd
import os
import random

class SyntheticDataGenerator:
    def __init__(self, n_samples=10000, base_qps=1000, anomaly_ratio=0.1):
        self.n_samples = n_samples
        self.base_qps = base_qps
        self.anomaly_ratio = anomaly_ratio
        self.time_index = pd.date_range(start="2026-01-01", periods=n_samples, freq='10min')

    def generate_baseline(self):
        """Generate normal sine-wave-like daily patterns"""
        time = np.arange(self.n_samples)
        # Daily cycle (144 * 10 mins = 1 day)
        daily_pattern = np.sin(2 * np.pi * time / 144)
        
        # System Metrics
        request_rate = self.base_qps + 500 * daily_pattern + np.random.normal(0, 100, self.n_samples)
        request_rate = np.clip(request_rate, 100, None) # Min 100 QPS
        
        # CPU typically scales with Request rate
        cpu_usage = 20 + (request_rate / self.base_qps) * 20 + np.random.normal(0, 5, self.n_samples)
        memory_usage = 40 + np.random.normal(0, 5, self.n_samples) # Memory is more stable
        disk_io = 10 + np.random.normal(0, 2, self.n_samples)
        network_io = request_rate * 0.5 + np.random.normal(0, 10, self.n_samples)
        
        # App Metrics
        response_time = 50 + (cpu_usage / 100) * 20 + np.random.normal(0, 5, self.n_samples)
        error_rate = np.random.uniform(0.1, 1.0, self.n_samples)
        active_sessions = request_rate * 0.1 + np.random.normal(0, 10, self.n_samples)

        df = pd.DataFrame({
            'timestamp': self.time_index,
            'Request_rate': request_rate,
            'CPU_usage': np.clip(cpu_usage, 0, 100),
            'Memory_usage': np.clip(memory_usage, 0, 100),
            'Disk_IO': np.clip(disk_io, 0, 100),
            'Network_IO': np.clip(network_io, 0, 1000),
            'Response_time': np.clip(response_time, 10, 5000),
            'Error_rate': np.clip(error_rate, 0, 100),
            'Active_sessions': np.clip(active_sessions, 0, None)
        })
        return df

    def inject_anomalies(self, df):
        """Inject Traffic spike, Gradual overload, Random failure"""
        n_anomalies = int(self.n_samples * self.anomaly_ratio / 60) # Number of anomaly blocks
        
        df['Is_Anomaly'] = 0
        df['Anomaly_Type'] = "Normal"
        
        for _ in range(n_anomalies):
            anomaly_type = random.choice(["Spike", "Gradual", "Failure"])
            idx = random.randint(0, self.n_samples - 60) # 60 steps = 10 hours
            
            if anomaly_type == "Spike":
                # Traffic spike (Burst) - Sudden increase in QPS and Latency
                length = random.randint(2, 6)
                df.loc[idx:idx+length, 'Request_rate'] *= np.random.uniform(3.0, 5.0)
                df.loc[idx:idx+length, 'CPU_usage'] = np.clip(df.loc[idx:idx+length, 'CPU_usage'] * 2.5, 0, 100)
                df.loc[idx:idx+length, 'Response_time'] *= np.random.uniform(4.0, 8.0)
                df.loc[idx:idx+length, 'Is_Anomaly'] = 1
                df.loc[idx:idx+length, 'Anomaly_Type'] = "Spike"
                
            elif anomaly_type == "Gradual":
                # Gradual overload - Memory leak or gradual increase in CPU
                length = random.randint(12, 36) # 2-6 hours
                gradient = np.linspace(1.0, 2.5, length)
                df.loc[idx:idx+length-1, 'CPU_usage'] = np.clip(df.loc[idx:idx+length-1, 'CPU_usage'] * gradient, 0, 100)
                df.loc[idx:idx+length-1, 'Memory_usage'] = np.clip(df.loc[idx:idx+length-1, 'Memory_usage'] * gradient, 0, 100)
                df.loc[idx:idx+length-1, 'Response_time'] *= gradient
                df.loc[idx:idx+length-1, 'Is_Anomaly'] = 1
                df.loc[idx:idx+length-1, 'Anomaly_Type'] = "Gradual"
                
            elif anomaly_type == "Failure":
                # Random failure - High error rate, low QPS, high Latency
                length = random.randint(1, 3)
                df.loc[idx:idx+length, 'Request_rate'] *= 0.1 # Drop in traffic 
                df.loc[idx:idx+length, 'Error_rate'] = np.random.uniform(50.0, 99.0)
                df.loc[idx:idx+length, 'Response_time'] = 5000 # Timeout
                df.loc[idx:idx+length, 'Is_Anomaly'] = 1
                df.loc[idx:idx+length, 'Anomaly_Type'] = "Failure"
                
        return df

    def generate(self, output_path="Data/synthetic_workload.csv"):
        print("Generating baseline metrics...")
        df = self.generate_baseline()
        print("Injecting complex anomalies...")
        df = self.inject_anomalies(df)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Synthetic dataset saved to {output_path} with shape {df.shape}")
        
        # Stats
        print(df['Anomaly_Type'].value_counts())
        return df

if __name__ == "__main__":
    generator = SyntheticDataGenerator(n_samples=20000) # Roughly 138 days
    generator.generate()
