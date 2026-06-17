import cv2
import time
import traceback
from pathlib import Path
from detect import PlateDetector
from preprocess import clean_plate_image
from ocr import extract_text

class BatchProcessor:
    def __init__(self, model_path):
        self.detector = PlateDetector(model_path=model_path)
        
    def process_directory(self, input_dir, output_dir, progress_callback=None):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logs_file_path = output_path / "batch_logs.txt"
        results_file_path = output_path / "results.txt"
        
        valid_exts = {".jpg", ".jpeg", ".png"}
        image_files = [f for f in input_path.rglob("*") if f.suffix.lower() in valid_exts]
        total_files = len(image_files)
        
        with open(logs_file_path, "w", encoding="utf-8") as log_file, \
             open(results_file_path, "w", encoding="utf-8") as res_file:
            
            log_file.write(f"--- Batch Processing Started ---\n")
            log_file.write(f"Total files found: {total_files}\n\n")
            log_file.flush()
            
            res_file.write(f"{'FILE NAME':<45} | {'STATUS':<10} | {'DETECTED PLATES'}\n")
            res_file.write("-" * 90 + "\n")
            res_file.flush()
            
            try:
                for idx, img_path in enumerate(image_files, 1):
                    start_time = time.time()
                    
                    try:
                        original_image, plates = self.detector.detect_and_crop(str(img_path))
                        detected_texts = []
                        log_results = []
                        
                        if plates:
                            for plate_idx, (cropped_plate, coords) in enumerate(plates):
                                cleaned_plate = clean_plate_image(cropped_plate)
                                text, conf = extract_text(cleaned_plate)
                                
                                formatted_result = f"{text} ({conf:.1f}%)"
                                detected_texts.append(formatted_result)
                                log_results.append((text, conf))
                                
                                x1, y1, x2, y2 = coords
                                cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(original_image, formatted_result, (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
                        
                        save_path = output_path / img_path.name
                        if original_image is not None:
                            cv2.imwrite(str(save_path), original_image)
                        
                        elapsed = time.time() - start_time
                        plates_str = ", ".join(detected_texts) if detected_texts else "None"
                        
                        log_file.write(f"[{idx}/{total_files}] File: {img_path.name}\n")
                        log_file.write(f"Status: SUCCESS\n")
                        log_file.write(f"Plates Detected: {plates_str}\n")
                        log_file.write(f"Processing Time: {elapsed:.4f} seconds\n")
                        log_file.write(f"----------------------------------------\n")
                        log_file.flush()
                        
                        res_file.write(f"{img_path.name:<45} | {'SUCCESS':<10} | {plates_str}\n")
                        res_file.flush()
                        
                        if progress_callback:
                            result_data = {
                                "file": img_path.name,
                                "texts": detected_texts,
                                "time": elapsed,
                                "index": idx,
                                "total": total_files,
                                "status": "SUCCESS"
                            }
                            progress_callback(result_data)
                            
                    except Exception as img_err:
                        elapsed = time.time() - start_time
                        error_trace = traceback.format_exc()
                        
                        log_file.write(f"[{idx}/{total_files}] File: {img_path.name}\n")
                        log_file.write(f"Status: FAILED\n")
                        log_file.write(f"Error Details:\n{error_trace}\n")
                        log_file.write(f"----------------------------------------\n")
                        log_file.flush()
                        
                        res_file.write(f"{img_path.name:<45} | {'FAILED':<10} | ERROR\n")
                        res_file.flush()
                        
                        if progress_callback:
                            progress_callback({
                                "file": img_path.name,
                                "texts": ["ERROR"],
                                "time": elapsed,
                                "index": idx,
                                "total": total_files,
                                "status": "FAILED"
                            })
                            
            except Exception as fatal_err:
                error_trace = traceback.format_exc()
                log_file.write(f"\n!!! FATAL SYSTEM CRASH !!!\n")
                log_file.write(f"Execution aborted at file {idx} of {total_files}.\n")
                log_file.write(f"Crash Reason:\n{error_trace}\n")
                log_file.flush()
                
                res_file.write(f"{'FATAL CRASH':<45} | {'ABORTED':<10} | SEE BATCH_LOGS.TXT\n")
                res_file.flush()
                raise