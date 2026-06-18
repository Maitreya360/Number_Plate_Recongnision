@echo off
title ANPR Continuous Training Pipeline
color 0A

cd ..\..

echo ========================================================
echo       ANPR MODEL UPGRADER (Continuous Fine-Tuning)
echo ========================================================
echo.

set /p raw_folder="Enter the name of the new raw data folder (e.g., test_data_set_1): "

if not exist "%raw_folder%" (
    echo [ERROR] The folder "%raw_folder%" does not exist in the main directory.
    pause
    exit /b
)

echo.
echo [1/2] Processing new data and adding to the training pile...
python src\training\prepare_data.py --raw "%raw_folder%"

if %errorlevel% neq 0 (
    echo [ERROR] Data preparation failed.
    pause
    exit /b
)

echo.
echo [2/2] Upgrading the AI model...
set /p epochs="Enter number of Epochs to train (Press ENTER for default 30): "
if "%epochs%"=="" set epochs=30

python src\training\train.py --epochs %epochs%

if %errorlevel% neq 0 (
    echo [ERROR] Training failed.
    pause
    exit /b
)

echo.
echo ========================================================
echo   PIPELINE COMPLETE! The model has been upgraded.
echo ========================================================
pause