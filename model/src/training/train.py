import sys
import os
import re
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from rich.console import Console
from rich.panel import Panel

# Custom class to simultaneously print to the screen AND save to a clean .txt file
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "w", encoding="utf-8")
        # This regex acts as a filter to remove ugly command prompt color codes from the text file
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        self.terminal.write(message)
        clean_msg = self.ansi_escape.sub('', message)
        self.log_file.write(clean_msg)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def isatty(self):
        # Tricks YOLO into drawing its progress bar correctly even though we are capturing the text
        return self.terminal.isatty()

def upgrade_model(data_yaml, weights_path, epochs, dataset_name):
    console = Console()
    
    weights_file = Path(weights_path)
    if not weights_file.exists():
        console.print(f"[bold red]❌ Error: Could not find the model at {weights_path}[/bold red]")
        return

    # 1. Create the timestamped folder first so we can put the logs in it
    timestamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    run_name = f"{dataset_name}_{timestamp}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Activate the Dual Logger
    log_path = run_dir / "terminal_logs.txt"
    sys.stdout = DualLogger(log_path)
    sys.stderr = sys.stdout 
    
    # 3. Clean up the CMD View
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Draw a beautiful UI Dashboard
    console.print(Panel(
        f"[bold white]Target Model:[/bold white] [green]{weights_path}[/green]\n"
        f"[bold white]Dataset:[/bold white] [cyan]{dataset_name}[/cyan]\n"
        f"[bold white]Epochs:[/bold white] [yellow]{epochs}[/yellow]\n"
        f"[bold white]Backup Directory:[/bold white] [magenta]runs/{run_name}[/magenta]",
        title="[bold blue]🚀 ANPR Model Training Initialized[/bold blue]",
        expand=False
    ))
    
    # Load and Train
    model = YOLO(str(weights_file))
    
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=8,
        project="runs",
        name=run_name,
        exist_ok=True 
    )
    
    new_weights = run_dir / "weights" / "best.pt"
    
    console.print("\n")
    if new_weights.exists():
        shutil.copy(new_weights, weights_file)
        console.print(Panel(
            "[bold green]SUCCESS! The active model has been overwritten and upgraded.[/bold green]\n"
            "[bold white]All logs, graphs, and a backup .pt file are safely stored.[/bold white]", 
            title="[bold cyan]Training Complete[/bold cyan]", 
            expand=False
        ))
    else:
        console.print("[bold red]⚠️ Warning: Training finished, but couldn't locate the new weights file.[/bold red]")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/test_samples/data.yaml")
    parser.add_argument("--weights", default="weights/plate_detector.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dataset_name", required=True)
    args = parser.parse_args()
    
    upgrade_model(args.data, args.weights, args.epochs, args.dataset_name)