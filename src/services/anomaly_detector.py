import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class HybridAnomalyDetector:
    def __init__(self, contamination=0.05, std_multiplier=3.0):
        self.iso_forest = IsolationForest(contamination=contamination, random_state=42)
        self.std_multiplier = std_multiplier
        self.mean_dict = {}
        self.std_dict = {}
        
    def fit(self, df):
        """Fit isolation forest and calculate statistical limits based on normal training data"""
        features_for_if = df[['CPU_usage', 'Response_time', 'Request_rate']].fillna(0)
        self.iso_forest.fit(features_for_if)
        
        # Calculate stats for all numeric cols
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.mean_dict[col] = df[col].mean()
            self.std_dict[col] = df[col].std()
            
    def predict(self, df):
        """Returns anomaly masks based on hybrid logic"""
        # 1. Isolation Forest Prediction (-1 is anomaly, 1 is normal)
        features_for_if = df[['CPU_usage', 'Response_time', 'Request_rate']].fillna(0)
        if_preds = self.iso_forest.predict(features_for_if)
        if_anomaly = (if_preds == -1)
        
        # 2. Statistical Deviation Detection (Mean + 3*STD)
        stat_anomaly = np.zeros(len(df), dtype=bool)
        for col in ['CPU_usage', 'Response_time', 'Request_rate']:
            if col in df.columns and col in self.mean_dict:
                threshold = self.mean_dict[col] + (self.std_multiplier * self.std_dict[col])
                # Also detect drops as anomalies for request rate? 
                stat_anomaly = stat_anomaly | (df[col] > threshold).values
                
        # 3. Hybrid Logic (High Confidence)
        # IF both detectors agree -> high confidence anomaly
        high_confidence = if_anomaly & stat_anomaly
        
        result_df = df.copy()
        result_df['is_high_conf_anomaly'] = high_confidence
        result_df['if_anomaly'] = if_anomaly
        result_df['stat_anomaly'] = stat_anomaly
        
        return result_df

if __name__ == "__main__":
    # Test
    data = pd.DataFrame({
        'CPU_usage': np.random.normal(30, 5, 100).tolist() + [95, 99],
        'Response_time': np.random.normal(50, 10, 100).tolist() + [5000, 6000],
        'Request_rate': np.random.normal(1000, 100, 100).tolist() + [5000, 5000]
    })
    
    detector = HybridAnomalyDetector()
    detector.fit(data.iloc[:100]) # Fit on normal
    preds = detector.predict(data)
    
    anomalies = preds[preds['is_high_conf_anomaly'] == True]
    print(f"High confidence anomalies detected: {len(anomalies)}")
    print(anomalies)
