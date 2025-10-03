@echo off
echo Starting Data Generation...
echo.

REM Activate virtual environment
call loan_env\Scripts\activate.bat

REM Generate training data
echo Generating training dataset...
python data\generate_data.py

if errorlevel 1 (
    echo Error: Failed to generate data
    pause
    exit /b 1
)

echo.
echo Data generation completed successfully!
echo Files created:
echo - data\loan_train.csv
echo - data\loan_train_split.csv  
echo - data\loan_test_split.csv
echo.
pause