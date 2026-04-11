import sys
import os
import argparse
import time

# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.tools.train_advanced import run_advanced_training_v3

def run_training_command(args):
    """
    Main training command.
    Executes the advanced TCN-Attention-BiLSTM training pipeline.
    """
    print("==========================================")
    print("   AI PREDICTION ENGINE — TRAIN COMMAND")
    print("==========================================")
    print(f"Mode: {args.mode}")
    print(f"Model Architecture: TCN-Attention-BiLSTM (Upgraded)")
    
    if args.mode == "global":
        print(f"Starting Global Training on {args.data_dir} for {args.epochs} epochs...")
        run_advanced_training_v3(epochs=args.epochs)
        print("[OK] Global Model Training completed.")
    elif args.mode == "single":
        print(f"Starting Single Server Training on {args.file}...")
        # For single file, we could adapt run_advanced_training_v3 to take a file list
        # but for now, we'll keep the global logic as the primary check.
        print("Feature pending: Single file training optimization.")
    else:
        print("Error: Unknown mode.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Prediction Engine Command Interface")
    parser.add_argument("mode", choices=["global", "single"], help="Training mode")
    parser.add_argument("--data_dir", default="Data", help="Directory for global training data")
    parser.add_argument("--file", help="Specific CSV file for single training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    
    args = parser.parse_args()
    run_training_command(args)
