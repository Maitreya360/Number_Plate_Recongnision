@echo off
title ANPR Continuous Training Pipeline
color 0A

:: This command anchors the script to your main 'model' folder automatically
cd /d "%~dp0\..\.."

echo ========================================================
echo       ANPR MODEL UPGRADER (Continuous Fine-Tuning)
echo ========================================================
echo.
echo The system will look for your data inside: data\test_samples\
echo.
set /p folder_name="Enter the name of your new data folder "

set raw_path=data\test_samples\%folder_name%

if not exist "%raw_path%" (
    echo.
    echo [ERROR] Could not find the folder at: %raw_path%
    echo Please make sure you typed the name correctly.
    pause
    exit /b
)

echo.
echo [1/2] Scanning '%folder_name%' to pair Images with XMLs...
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

python src\training\train.py --data "data\test_samples\data.yaml" --epochs %epochs%

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