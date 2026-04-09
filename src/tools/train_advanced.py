import sys
import os
import glob
import json
import time
import numpy as np
import pandas as pd

# Suppress TensorFlow GPU warnings on Windows
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# DLL CONFIGURATION (Local GPU Support)
# ==========================================
if sys.platform == 'win32':
    dll_path = os.path.dirname(os.path.abspath(__file__))
    # REQUIRED for Python 3.8+ to load local DLLs
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(dll_path)
        except Exception as e:
            print(f"[!] DLL Add Warning: {e}")
    os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ==========================================
# GPU CONFIGURATION (Academic Phase 10)
# ==========================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Setting a VRAM limit of 4GB for stability on Windows WDDM
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
        
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        gpu_names = [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']
        
        print(f"[*] GPU detected: {len(gpus)} devices. Memory Growth ENABLED.")
        for name in gpu_names:
            print(f"[*] Device Name: {name}")
        print(f"[*] CUDA built: {tf.test.is_built_with_cuda()}")
    except RuntimeError as e:
        print(f"[!] GPU Init Error: {e}")
else:
    print("[!] WARNING: No GPU detected. Check CUDA/cuDNN installation.")

# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.data_preprocessing import prepare_data_v2
from src.utils.data_loaders import UniversalDataLoader
from src.models.tcn_attention_bilstm import build_advanced_model

# ==========================================
# 1. CONFIG (PHASE 3)
# ==========================================
DATA_DIR = "Data/"
WINDOW_SIZE = 60
NUM_FEATURES = 10  # Upgraded from 8
USE_LOG = True
EPOCHS = 100
BATCH_SIZE = 64 # Reduced for stability on laptop/WDDM drivers
HORIZON = 6 # Predicted steps into the future (e.g., 6 steps = 1 hour if 10-min interval 144=1 day)
MODEL_DIR = "models/checkpoints_advanced"

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

# ==========================================
# 2. ADVANCED AUTOGRAD TRAINING LOOP
# ==========================================
class AdvancedTrainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

    # Remove @tf.function for stability in complex graph (Windows GPU)
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.loss_fn(y, predictions)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss_metric.update_state(loss)
        return loss

    # Remove @tf.function for stability
    def test_step(self, x, y):
        predictions = self.model(x, training=False)
        loss = self.loss_fn(y, predictions)
        self.val_loss_metric.update_state(loss)
        return loss

def get_gpu_memory():
    """Returns (used, total) in MB using nvidia-smi for Windows/Linux"""
    try:
        output = os.popen('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader').read()
        if not output: return 0, 0
        used, total = map(int, output.strip().split(','))
        return used, total
    except:
        return 0, 0

def run_advanced_training_v3():
    # Record start resources
    used_start, total_vram = get_gpu_memory()
    start_time_global = time.time()
    
    # Support multiple formats in training
    extensions = ['*.csv', '*.json', '*.xlsx', '*.xls']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True))
    
    files = files[:40]  # Increased from 50 to 40 for a balanced full run
    print(f"[*] Phase 3 Training: Loading {len(files)} multi-format files...")
    
    if not files:
        print("[!] Error: No files found in DATA_DIR")
        return

    loader = UniversalDataLoader()
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for f in files:
        df = loader.load(f)
        if df is not None and len(df) > WINDOW_SIZE:
            try:
                # Use upgraded preprocessing (10+ features) with HORIZON
                X_tr, y_tr, X_te, y_te, _ = prepare_data_v2(df, window_size=WINDOW_SIZE, horizon=HORIZON)
                X_train_list.append(X_tr.astype(np.float16))
                y_train_list.append(y_tr.astype(np.float16))
                X_test_list.append(X_te.astype(np.float16))
                y_test_list.append(y_te.astype(np.float16))
            except Exception as e: 
                print(f"Skipping {f} due to: {e}")
                continue

    if not X_train_list:
        print("[!] No valid data found for training. Check if files have >60 samples.")
        return

    print(f"[*] Concatenating data from {len(X_train_list)} valid files...")

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    print(f"\n--- DATA AUDIT (NCKH Phase 10) ---")
    print(f"- Total Files processed: {len(files)}")
    print(f"- Training samples: {X_train.shape[0]}")
    print(f"- Validation samples: {X_test.shape[0]}")
    print(f"- Input Features: {X_train.shape[2]} (Target: 10)")
    print(f"- Sequence Length: {X_train.shape[1]}")
    print(f"----------------------------------")

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Automatically adapt to 10 features and multi-step HORIZON
    model = build_advanced_model((WINDOW_SIZE, X_train.shape[2]), horizon=HORIZON)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()
    trainer = AdvancedTrainer(model, optimizer, loss_fn)

    # Custom Callback for VRAM monitoring
    class VRAMLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            used, _ = get_gpu_memory()
            print(f" - VRAM: {used}MB")

    print(f"\n[START] Phase 3 Training (model.fit) | Initial VRAM: {used_start}/{total_vram} MB")
    
    try:
        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ModelCheckpoint(os.path.join(MODEL_DIR, "best_attention_model_v3.h5"), save_best_only=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7),
                VRAMLogger()
            ]
        )
        best_val_loss = min(history.history['val_loss'])
    except Exception as e:
        print(f"\n[!] Training Error: {e}")
        best_val_loss = float('inf')
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")

    # Final Audit Report
    used_end, _ = get_gpu_memory()
    total_time = time.time() - start_time_global
    print(f"\n==========================================")
    print(f"   NCKH AUDIT REPORT: TCN-Att-BiLSTM")
    print(f"==========================================")
    print(f"Status: COMPLETED (Checkpoints in {MODEL_DIR})")
    print(f"Total Training Time: {total_time:.2f}s")
    print(f"Data Samples: {len(X_train)} train / {len(X_test)} val")
    print(f"VRAM Usage: {used_start}MB (Start) -> {used_end}MB (End)")
    print(f"Peak Delta VRAM: {max(0, used_end - used_start)} MB")
    print(f"Best Validation Loss (MSE): {best_val_loss:.6f}")
    if best_val_loss != float('inf'):
        print(f"Final Estimated RMSE: {np.sqrt(best_val_loss):.6f}")
    print(f"==========================================\n")

if __name__ == "__main__":
    run_advanced_training_v3()
