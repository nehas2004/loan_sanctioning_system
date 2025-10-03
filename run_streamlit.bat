@echo off
echo Starting Streamlit Application...
echo.

REM Activate virtual environment
call loan_env\Scripts\activate.bat

echo Streamlit server starting...
echo The application will open in your default browser
echo Press Ctrl+C to stop the server
echo.

REM Start Streamlit application
streamlit run frontend\streamlit_app.py

pause