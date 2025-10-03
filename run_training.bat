@echo off
echo Starting Model Training...
echo.

REM Activate virtual environment
call loan_env\Scripts\activate.bat

REM Train the model
echo Training Decision Tree model...
python models\loan_model.py

if errorlevel 1 (
    echo Error: Failed to train model
    pause
    exit /b 1
)

echo.
echo Model training completed successfully!
echo Model saved to: models\loan_model.pkl
echo.
pause