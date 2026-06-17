import zipfile
import shutil
import time
from pathlib import Path

def extract_and_prepare(zip_path, raw_dir):
    zip_file = Path(zip_path)
    base_name = zip_file.stem
    
    timestamp = time.strftime("(%d_%m_%Y____%H_%M_%S)")
    
    unique_run_name = f"{base_name}_{timestamp}"
    
    temp_dir = Path(raw_dir) / f"temp_{unique_run_name}"
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
        
    return temp_dir, unique_run_name

def cleanup_temp(temp_dir):
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)