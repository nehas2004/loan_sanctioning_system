#!/usr/bin/env python3
"""
Test script to verify all dependencies are working
"""
import sys
print(f"Python version: {sys.version}")

try:
    import flask
    print(f"✅ Flask {flask.__version__}")
except ImportError as e:
    print(f"❌ Flask: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas: {e}")

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn: {e}")

try:
    import joblib
    print(f"✅ Joblib {joblib.__version__}")
except ImportError as e:
    print(f"❌ Joblib: {e}")

try:
    from flask_cors import CORS
    print(f"✅ Flask-CORS imported successfully")
except ImportError as e:
    print(f"❌ Flask-CORS: {e}")

print("\n🎯 Dependency check completed!")