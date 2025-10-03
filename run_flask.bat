@echo off
echo Starting Flask Web Application...
echo.

REM Activate virtual environment
call loan_env\Scripts\activate.bat

REM Set environment variables
set FLASK_APP=backend\app.py
set FLASK_ENV=development
set FLASK_DEBUG=1

echo Flask server starting at http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

REM Start Flask application
python backend\app.py

pause