import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

# Path setup
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.data_loaders import UniversalDataLoader
from src.utils.data_preprocessing import prepare_data_v2
from src.models.attention_layer import FeatureAttention, TemporalAttention

from src.services.decision_engine import RuleBasedDecisionEngine

# Constants
MODEL_PATH = "models/checkpoints_advanced/best_attention_model_v3.h5"
DATA_DIR = "Data"
OUTPUT_FILE = "latest_prediction.json"
WINDOW_SIZE = 60
CONGESTION_THRESHOLD = 85.0 # % QPS Load

class InferenceEngine:
    def __init__(self):
        self.loader = UniversalDataLoader()
        self.model = self._load_model()
        self.decider = RuleBasedDecisionEngine()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            return None
        return tf.keras.models.load_model(MODEL_PATH, 
                                         custom_objects={
                                             'Attention': TemporalAttention,
                                             'FeatureAttention': FeatureAttention,
                                             'TemporalAttention': TemporalAttention
                                         },
                                         compile=False)

    def calculate_lead_time(self, current_load, predicted_load):
        """Lead time to 85% overload"""
        if predicted_load >= CONGESTION_THRESHOLD:
            return "IMMEDIATE"
        if predicted_load > current_load:
            rate = (predicted_load - current_load) / 10 # 10 mins window
            if rate <= 0: return ">30 mins"
            time_to_85 = (CONGESTION_THRESHOLD - current_load) / rate
            return f"{round(time_to_85, 1)} mins"
        return ">30 mins"

    def run_inference(self, file_path):
        if not self.model: return
        
        df = self.loader.load(file_path)
        if df is None or len(df) < WINDOW_SIZE:
             return

        try:
            # Phase 10: Advanced Web Features Preparation
            X_all, _, _, _, scaler = prepare_data_v2(df, window_size=WINDOW_SIZE, train_ratio=0.99)
            if len(X_all) == 0: return
            
            last_window = X_all[-1:]
            pred_val = self.model.predict(last_window, verbose=0) # Shape (1, horizon, 4)
            
            # Extract CPU/Load prediction (1st target)
            p_idx = pred_val[0, 0, 0] if len(pred_val.shape) == 3 else pred_val[0, 0]
            pred_unscaled = float(scaler['cpu'].inverse_transform([[p_idx]])[0][0])
            
            # Extract current state for decision
            curr_load = float(df['CPU_usage'].iloc[-1])
            curr_rt = float(df['Response_time'].iloc[-1])
            
            # [ UPGRADE ] - Autonomous Decision Agent
            # Simulate a 5-min lookahead for decision
            pred_rt = float(scaler['rt'].inverse_transform([[pred_val[0, 0, 1] if len(pred_val.shape) == 3 else pred_val[0, 1]]])[0][0])
            cong_prob = float(pred_val[0, 0, 3] if len(pred_val.shape) == 3 else 0.5)
            
            action, reason, reward = self.decider.decide(pred_unscaled, pred_rt, cong_prob)
            
            lead_time = self.calculate_lead_time(curr_load, pred_unscaled)
            is_critical = pred_unscaled >= CONGESTION_THRESHOLD
            
            result = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file": os.path.basename(file_path),
                "current_load": round(curr_load, 2),
                "predicted_load": round(pred_unscaled, 2),
                "response_time": round(curr_rt, 2),
                "lead_time": lead_time,
                "is_critical": is_critical,
                "action": action,
                "reason": reason,
                "policy_reward": reward,
                "risk_level": "CRITICAL" if is_critical else ("WARNING" if pred_unscaled > 70 else "NORMAL")
            }
            
            with open(OUTPUT_FILE, "w") as f:
                json.dump(result, f, indent=4)
            
            status = f"[{result['timestamp']}] Action: {action} | Policy Reward: {reward} | Risk: {result['risk_level']}"
            print(status)
            with open("inference.log", "a") as log: log.write(status + "\n")
                
        except Exception as e:
            print(f"Inference Error: {e}")
            import traceback
            traceback.print_exc()

class DataHandler(FileSystemEventHandler):
    def __init__(self, engine):
        self.engine = engine

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith((".csv", ".json")):
            self.engine.run_inference(event.src_path)

if __name__ == "__main__":
    print("--- PAES: Web Autonomous AI Service (Academic V5.2) ---")
    engine = InferenceEngine()
    
    # Init from last log
    files = []
    for f in os.listdir(DATA_DIR):
        if f.endswith((".csv", ".json")): files.append(os.path.join(DATA_DIR, f))
    
    if files:
        latest_f = max(files, key=os.path.getmtime)
        print(f"[*] Pre-loading live context from: {latest_f}")
        engine.run_inference(latest_f)

    observer = Observer()
    observer.schedule(DataHandler(engine), path=DATA_DIR, recursive=True)
    observer.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

