```markdown
# ANPR Model Pipeline (YOLO + Tesseract)

This directory contains the core Computer Vision and Machine Learning pipeline for the Automatic Number Plate Recognition (ANPR) system.

## System Requirements & Setup

This project uses Tesseract OCR, which requires a system-level installation before the Python wrapper will work.

1. Download and install Tesseract OCR for your operating system.
2. Note the installation path.

Ensure you have Python installed. Navigate to this directory and install the required Python dependencies.

```bash
pip install -r requirements.txt

```

## Directory Architecture

* **data/**: Stores all image assets. Place unedited vehicle images in raw/ and use test_samples/ for validation.
* **weights/**: Contains the custom-trained YOLO model files.
* **tessdata/**: Contains Tesseract traineddata files for specific fonts or languages.
* **src/**: Contains the modular Python scripts.

## Module Breakdown

* **preprocess.py**: Handles image grayscaling, bilateral filtering, and binarization using OpenCV.
* **detect.py**: Executes the YOLO object detection to localize and crop the license plate.
* **ocr.py**: Runs Pytesseract on the preprocessed image to extract alphanumeric text.
* **pipeline.py**: The main executable that chains the modules together.

## Execution
To run a test image through the pipeline:
python src/pipeline.py --image data/test_samples/vehicle_1.jpg