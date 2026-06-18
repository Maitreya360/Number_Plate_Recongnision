import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

def upgrade_model(data_yaml, weights_path, epochs):
    weights_file = Path(weights_path)
    
    if not weights_file.exists():
        print(f"❌ Error: Could not find the model at {weights_path}")
        return

    print(f"🧠 Loading existing model from: {weights_path}")
    model = YOLO(str(weights_file))
    
    print(f"🚀 Starting Incremental Training on {data_yaml} for {epochs} epochs...")
    
    # We force YOLO to save the output to a specific predictable folder
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=8,
        project="training_runs",
        name="plate_updater",
        exist_ok=True # Overwrites the temp training folder so we don't waste disk space
    )
    
    # The new, smarter weights are saved here by YOLO
    new_weights = Path("training_runs/plate_updater/weights/best.pt")
    
    if new_weights.exists():
        # Overwrite the old model with the newly trained one!
        shutil.copy(new_weights, weights_file)
        print(f"🎉 SUCCESS! Your model at '{weights_path}' has been upgraded with the new data!")
    else:
        print("⚠️ Warning: Training finished, but couldn't locate the new weights file.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/test_sample/data.yaml")
    parser.add_argument("--weights", default="weights/plate_detector.pt")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    upgrade_model(args.data, args.weights, args.epochs)