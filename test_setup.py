"""
Quick test script to verify the project structure and basic functionality
Run this after installing dependencies with setup.bat
"""

import os
import sys

def check_project_structure():
    """Check if all required files and directories exist"""
    print("🔍 Checking project structure...")
    
    required_files = [
        'requirements.txt',
        'backend/app.py',
        'models/loan_model.py',
        'data/generate_data.py',
        'frontend/streamlit_app.py',
        'templates/index.html',
        'templates/predict.html',
        'templates/result.html',
        'templates/analytics.html',
        'static/css/style.css',
        'static/js/main.js'
    ]
    
    required_dirs = [
        'backend',
        'models',
        'data',
        'frontend',
        'templates',
        'static',
        'static/css',
        'static/js'
    ]
    
    # Check directories
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directory: {directory}")
        else:
            print(f"❌ Missing directory: {directory}")
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ File: {file_path}")
        else:
            print(f"❌ Missing file: {file_path}")

def check_dependencies():
    """Check if dependencies are available"""
    print("\n🔍 Checking dependencies...")
    
    dependencies = [
        'pandas',
        'numpy',
        'flask',
        'streamlit'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} is available")
        except ImportError:
            print(f"❌ {dep} is not installed")

def test_basic_imports():
    """Test basic imports without sklearn"""
    print("\n🔍 Testing basic imports...")
    
    try:
        import json
        import os
        import sys
        from datetime import datetime
        print("✅ Standard library imports work")
    except Exception as e:
        print(f"❌ Standard library import error: {e}")
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ Data science libraries work")
    except Exception as e:
        print(f"❌ Data science libraries not available: {e}")
    
    try:
        from flask import Flask
        print("✅ Flask is available")
    except Exception as e:
        print(f"❌ Flask not available: {e}")

def main():
    print("🏦 Loan Sanctioning System - Project Verification")
    print("=" * 50)
    
    check_project_structure()
    check_dependencies()
    test_basic_imports()
    
    print("\n" + "=" * 50)
    print("📋 Next Steps:")
    print("1. Run setup.bat to install dependencies")
    print("2. Run run_data_generation.bat to create training data")
    print("3. Run run_training.bat to train the model")
    print("4. Run run_flask.bat to start the web application")
    print("   OR run run_streamlit.bat for the dashboard")
    print("\n🎉 Project setup verification complete!")

if __name__ == "__main__":
    main()