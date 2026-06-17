# Automatic Number Plate Recognition (ANPR) Pipeline

Welcome to the ANPR project! This directory contains the core Computer Vision and Artificial Intelligence pipeline. It uses a **YOLOv8** object detection model to find the license plate in an image, and **Tesseract OCR** to read the text off that plate.

This guide is written for absolute beginners. Follow the steps below carefully to set up your system, download the required AI models, and run your first test.

---

## 1. Directory Architecture

Before starting, ensure your project folder looks exactly like this. You will need to create some of these folders manually if they are missing.

```text
anpr_system/
└── model/
    ├── data/
    │   ├── raw/                 <-- You will put your ZIP files of images here
    │   └── processed/           <-- The AI will save results here
    ├── weights/                 <-- You will put your downloaded AI model here
    ├── tessdata/                <-- (Optional) Tesseract language files
    ├── src/
    │   ├── engine/
    │   │   ├── processor.py     <-- Handles memory-safe image processing
    │   │   └── zip_manager.py   <-- Unzips files and creates timestamped folders
    │   ├── detect.py            <-- YOLO object detection script
    │   ├── ocr.py               <-- Text extraction script
    │   ├── pipeline.py          <-- The main script you will run
    │   └── preprocess.py        <-- Cleans images (black & white) for better reading
    └── requirements.txt         <-- List of required Python packages





2. Software Installation (Third-Party Resources)
This project relies on external AI software that you must install on your computer before running the Python code.

A. Install Python
If you do not have Python installed, download it from Python.org. Ensure you check the box that says "Add Python to PATH" during installation.

B. Install Tesseract OCR (The Text Reader)
Tesseract is a free, open-source engine developed by Google that reads text from images. Python needs the actual software installed on your Windows machine to work.

Download: Go to the official Windows installer page: Tesseract at UB Mannheim

Select Version: Download the latest 64-bit installer (e.g., tesseract-ocr-w64-setup-5.3.3...exe).

Install: Run the installer.

When asked to select components, leave the defaults checked.

(Optional) If you want to read regional Indian plates, click the + next to "Additional language data" and check Gujarati and Hindi.

Path Check: Ensure it installs to C:\Program Files\Tesseract-OCR\. If you install it somewhere else, you must update the path inside the src/ocr.py file.

C. Download the YOLO Model Weights (The Plate Finder)
Standard YOLO models look for cars and people. We need a specialized model trained specifically to find license plates.

Download: Click this direct link to download the custom model weights: Download best.pt from HuggingFace

Rename: Locate the downloaded file (best.pt) and rename it to plate_detector.pt.

Move: Cut and paste plate_detector.pt into your project's model/weights/ folder.





3. Python Environment Setup
Now that the external software is installed, you need to install the Python libraries that connect everything together.

Open your computer's Terminal (or Command Prompt).

Use the cd command to navigate into the model/ folder of this project.

Run the following command to install all necessary packages:

Bash
pip install -r requirements.txt
(This will install opencv-python for image manipulation, ultralytics for the YOLO AI, pytesseract to talk to Tesseract, and rich for the colorful progress bar).





4. How to Run the Pipeline
The system is designed to process batches of images securely without crashing low-end computers.

Step 1: Prepare your data
Gather a few images of cars/motorcycles with visible license plates.

Compress those images into a .zip file. Let's name it A.zip.

Place A.zip inside the model/data/raw/ folder.

Step 2: Execute the Code
Open your terminal, ensure you are inside the model/ folder, and type the following command:

Bash
python src/pipeline.py --zip data/raw/A.zip
(Note: The system automatically looks for weights/plate_detector.pt. You do not need to specify it).

Step 3: Watch the Process
You will see a live, color-coded progress bar in your terminal showing the system extracting the images, finding the plates, and reading the text in real-time.





5. Understanding the Outputs
Once the terminal says "Batch processing completed successfully," navigate to your model/data/processed/ folder.

You will find a brand new folder named after your zip file with a precise timestamp (e.g., A_(17_06_2026____14_50_05)). Inside this folder, you will find:

The Images: Copies of your original images, but with green boxes drawn around the detected license plates and the extracted text written above them.

results.txt: A clean, highly readable table showing the File Name, the Status (SUCCESS/FAILED), and the specific plate text detected along with the AI's confidence percentage.

batch_logs.txt: A deeply detailed log file. If an image crashes the system or is corrupted, this file will contain the exact error codes needed by a developer to fix the issue.

6. Troubleshooting Common Errors
Error: TesseractNotFoundError: C:\Program Files\Tesseract-OCR\tesseract.exe is not installed

Fix: You skipped Step 2B. Download and install Tesseract. If you installed it on a D:\ drive or a custom folder, open src/ocr.py and change the tesseract_cmd path to match your custom location.

Error: FileNotFoundError: weights/plate_detector.pt

Fix: You skipped Step 2C. Download the best.pt file from HuggingFace, rename it to plate_detector.pt, and ensure it is sitting inside the model/weights/ folder.

Result says "None": * Fix: The AI could not physically find a license plate in the image. Ensure the image is clear, not too blurry, and the plate is fully visible.