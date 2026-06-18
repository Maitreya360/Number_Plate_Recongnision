# Automatic Number Plate Recognition (ANPR) Pipeline

Welcome to the ANPR project! This directory contains the core Computer Vision and Artificial Intelligence pipeline. It uses a **YOLOv8** object detection model to find the license plate in an image, and **Tesseract OCR** to read the text off that plate.

This guide is written for absolute beginners. Follow the steps below carefully to set up your system, download the required AI models, run your first test, and continuously train the AI with your own data.

---

## 1. Directory Architecture

Before starting, ensure your project folder looks exactly like this. You will need to create some of these folders manually if they are missing.

```text
anpr_system/
└── model/
    ├── data/
    │   ├── raw/                 <-- You will put your ZIP files of images here
    │   ├── processed/           <-- The AI will save results here
    │   └── test_sample/         <-- Your growing AI training dataset lives here
    ├── weights/                 <-- You will put your downloaded AI model here
    ├── tessdata/                <-- (Optional) Tesseract language files
    ├── src/
    │   ├── engine/
    │   │   ├── processor.py     <-- Handles memory-safe image processing
    │   │   └── zip_manager.py   <-- Unzips files and creates timestamped folders
    │   ├── training/
    │   │   ├── auto_trainer.bat <-- One-click automatic training pipeline
    │   │   ├── prepare_data.py  <-- Formats new data and adds it to the pile
    │   │   └── train.py         <-- Upgrades the .pt file
    │   ├── detect.py            <-- YOLO object detection script
    │   ├── ocr.py               <-- Text extraction script
    │   ├── pipeline.py          <-- The main QA script you will run
    │   └── preprocess.py        <-- Cleans images (upscales & pads) for better reading
    └── requirements.txt         <-- List of required Python packages

```

---

## 2. Software Installation (Third-Party Resources)

This project relies on external AI software that you must install on your computer before running the Python code.

### A. Install Python

If you do not have Python installed, download it from [Python.org](https://www.python.org/downloads/). Ensure you check the box that says **"Add Python to PATH"** during installation.

### B. Install Tesseract OCR (The Text Reader)

Tesseract is a free, open-source engine developed by Google that reads text from images. Python needs the actual software installed on your Windows machine to work.

1. **Download:** Go to the official Windows installer page: [Tesseract at UB Mannheim](https://www.google.com/search?q=https://github.com/UB-Mannheim/tesseract/wiki)
2. **Select Version:** Download the latest 64-bit installer (e.g., `tesseract-ocr-w64-setup-5.3.3...exe`).
3. **Install:** Run the installer.
* When asked to select components, leave the defaults checked.
* *(Optional)* If you want to read regional Indian plates, click the `+` next to "Additional language data" and check **Gujarati** and **Hindi**.


4. **Path Check:** Ensure it installs to `C:\Program Files\Tesseract-OCR\`. If you install it somewhere else, you **must** update the path inside the `src/ocr.py` file.

### C. Download the YOLO Model Weights (The Plate Finder)

Standard YOLO models look for cars and people. We need a specialized model trained specifically to find license plates.

1. **Download:** Click this direct link to download the custom model weights: [Download best.pt from HuggingFace](https://www.google.com/search?q=https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt)
2. **Rename:** Locate the downloaded file (`best.pt`) and rename it to `plate_detector.pt`.
3. **Move:** Cut and paste `plate_detector.pt` into your project's `model/weights/` folder.

---

## 3. Python Environment Setup

Now that the external software is installed, you need to install the Python libraries that connect everything together.

1. Open your computer's Terminal (or Command Prompt).
2. Use the `cd` command to navigate into the `model/` folder of this project.
3. Run the following command to install all necessary packages:

```bash
pip install -r requirements.txt

```

*(This will install `opencv-python` for image manipulation, `ultralytics` for the YOLO AI, `pytesseract` to talk to Tesseract, and `rich` for the colorful progress bar).*

---

## 4. How to Run the Inference Pipeline

The system is designed to process batches of images securely without crashing low-end computers.

### Step 1: Prepare your data

1. Gather a few images of cars/motorcycles with visible license plates.
2. Compress those images into a `.zip` file. Let's name it `A.zip`.
3. Place `A.zip` inside the `model/data/raw/` folder.

### Step 2: Execute the Code

Open your terminal, ensure you are inside the `model/` folder, and type the following command:

```bash
python src/pipeline.py --zip data/raw/A.zip

```

*(Note: The system automatically looks for `weights/plate_detector.pt`. You do not need to specify it).*

### Step 3: Watch the Process

You will see a live, color-coded progress bar in your terminal showing the system extracting the images, finding the plates, and reading the text in real-time.

---

## 5. How to Train and Upgrade the Model (Continuous Fine-Tuning)

If the AI fails to recognize specific types of license plates (like angled motorcycle plates), you can continuously upgrade its brain by feeding it local data without losing its previous knowledge.

### Step 1: Prepare your Raw Data

1. Create a new folder (e.g., `test_data_set_1`).
2. Inside this folder, place your raw `.jpg` images and their corresponding `.xml` bounding box files.
3. Move this folder into your main `model/` directory.

### Step 2: Run the Auto-Trainer

1. Navigate to your `src/training/` folder and double-click the **`auto_trainer.bat`** file.
2. When the black terminal window opens, it will ask for the name of your raw data folder. Type the folder name (e.g., `test_data_set_1`) and press **Enter**.
3. The script will automatically:
* Convert your `.xml` files into YOLO format.
* Merge the new images into your growing `data/test_sample/` dataset.
* Ask you for the number of epochs to train (default is 30).
* Train the AI and automatically overwrite `weights/plate_detector.pt` with the newly upgraded version.



---

## 6. Understanding the Outputs

Once the terminal says "Batch processing completed successfully," navigate to your `model/data/processed/` folder.

You will find a brand new folder named after your zip file with a precise timestamp (e.g., `A_(17_06_2026____14_50_05)`). Inside this folder, you will find:

1. **The Images:** Copies of your original images, but with green boxes drawn around the detected license plates and the extracted text written above them.
2. **`results.txt`:** A clean, highly readable table showing the File Name, the Status (SUCCESS/FAILED), and the specific plate text detected along with the AI's confidence percentage.
3. **`batch_logs.txt`:** A deeply detailed log file. If an image crashes the system or is corrupted, this file will contain the exact error codes needed by a developer to fix the issue.

---

## 7. Troubleshooting Common Errors

* **Error:** `TesseractNotFoundError: C:\Program Files\Tesseract-OCR\tesseract.exe is not installed`
* **Fix:** You skipped Step 2B. Download and install Tesseract. If you installed it on a `D:\` drive or a custom folder, open `src/ocr.py` and change the `tesseract_cmd` path to match your custom location.


* **Error:** `FileNotFoundError: weights/plate_detector.pt`
* **Fix:** You skipped Step 2C. Download the `best.pt` file from HuggingFace, rename it to `plate_detector.pt`, and ensure it is sitting inside the `model/weights/` folder.


* **Result says "None":** * **Fix:** The AI could not physically find a license plate in the image. Ensure the image is clear, not too blurry, and the plate is fully visible. If the plate is clear but still returning "None", gather more similar images and run them through the **Continuous Training Pipeline** (Section 5) to teach the model what to look for.