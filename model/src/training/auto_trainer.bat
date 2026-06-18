@echo off
title ANPR Continuous Training Pipeline
color 0A

cd /d "%~dp0\..\.."

echo ========================================================
echo       ANPR MODEL UPGRADER (Continuous Fine-Tuning)
echo ========================================================
echo.
echo The system will look for your raw data inside: data\data_set\
echo.
set /p folder_name="Enter the name of your new data folder (e.g., test_data_set_1): "

:: The script now searches inside the new data_set folder
set raw_path=data\data_set\%folder_name%

if not exist "%raw_path%" (
    echo.
    echo [ERROR] Could not find the folder at: %raw_path%
    echo Please make sure you typed the name correctly.
    pause
    exit /b
)

echo.
echo [1/2] Scanning '%folder_name%' to pair Images with XMLs...
:: The script reads from data_set but permanently saves to test_samples
python src\training\prepare_data.py --raw "%raw_path%" --yolo "data\test_samples"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The data preparation crashed. Please read the Python error above.
    pause
    exit /b
)

echo.
echo [2/2] Upgrading the AI model...
set /p epochs="Enter number of Epochs to train (Press ENTER for default 30): "
if "%epochs%"=="" set epochs=30

:: The AI still trains using the master memory bank in test_samples
python src\training\train.py --data "data\test_samples\data.yaml" --epochs %epochs% --dataset_name "%folder_name%"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The training crashed. Please read the Python error above.
    pause
    exit /b
)

echo.
echo ========================================================
echo   SUCCESS! The model has been upgraded.
echo ========================================================
pause