import argparse
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from engine.zip_manager import extract_and_prepare, cleanup_temp
from engine.processor import BatchProcessor

def run_bulk_pipeline(zip_path, model_path, raw_dir, processed_dir):
    console = Console()
    console.print(f"[bold blue]Initializing Q/A testing pipeline for {zip_path}[/bold blue]")
    
    temp_dir, base_name = extract_and_prepare(zip_path, raw_dir)
    final_output_dir = Path(processed_dir) / base_name
    
    processor = BatchProcessor(model_path=model_path)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task_id = progress.add_task("[cyan]Processing images...", total=None)
        
        def update_ui(data):
            progress.update(task_id, total=data["total"], completed=data["index"])
            
            if data["status"] == "SUCCESS":
                status_text = f"[green]{data['index']} of {data['total']}[/green] - [yellow]{data['file']}[/yellow]\n"
                status_text += f"[bold white]Detected: {', '.join(data['texts']) if data['texts'] else 'None'}[/bold white] "
                status_text += f"| [magenta]Time: {data['time']:.2f}s[/magenta]\n"
            else:
                status_text = f"[red]{data['index']} of {data['total']}[/red] - [yellow]{data['file']}[/yellow]\n"
                status_text += f"[bold red]Status: CRASHED (See batch_logs.txt for details)[/bold red] "
                status_text += f"| [magenta]Time: {data['time']:.2f}s[/magenta]\n"
                
            progress.console.print(status_text)