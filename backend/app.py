from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loan_model import LoanPredictionModel

app = Flask(__name__, template_folder='../templates', static_folder='../static')
CORS(app)

# Production configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'loan-ml-secret-key-2024')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# Initialize model
loan_model = LoanPredictionModel()

# Try to load existing model, otherwise train a new one
def initialize_model():
    if os.path.exists('models/loan_model.pkl'):
        loan_model.load_model('models/loan_model.pkl')
        print("Model loaded successfully!")
    else:
        print("No trained model found. Training a new model...")
        try:
            # Check if data exists
            if os.path.exists('data/loan_train.csv'):
                # Load and train model
                df = loan_model.load_data('data/loan_train.csv') 
                if df is not None:
                    processed_data = loan_model.preprocess_data(df)
                    X, y = loan_model.prepare_features(processed_data)
                    loan_model.train_model(X, y)
                    
                    # Save the model
                    os.makedirs('models', exist_ok=True)
                    loan_model.save_model('models/loan_model.pkl')
                    print("Model trained and saved successfully!")
                else:
                    print("Error loading training data")
            else:
                print("Training data not found. Please run generate_data.py first.")
        except Exception as e:
            print(f"Error training model: {e}")

# Initialize model on startup
initialize_model()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Loan prediction page"""
    if request.method == 'POST':
        try:
            # Get form data
            loan_amount_rupees = float(request.form['loan_amount'])
            # Convert rupees to thousands for model compatibility
            loan_amount_thousands = loan_amount_rupees / 1000
            
            applicant_data = {
                'Gender': request.form['gender'],
                'Married': request.form['married'],
                'Dependents': request.form['dependents'],
                'Education': request.form['education'],
                'Self_Employed': request.form['self_employed'],
                'ApplicantIncome': float(request.form['applicant_income']),
                'CoapplicantIncome': float(request.form['coapplicant_income']),
                'LoanAmount': loan_amount_thousands,  # Store in thousands for model
                'Loan_Amount_Term': int(request.form['loan_amount_term']),
                'Credit_History': int(request.form['credit_history']),
                'Property_Area': request.form['property_area']
            }
            
            # For display purposes, store the original rupee amount
            applicant_data_display = applicant_data.copy()
            applicant_data_display['LoanAmount'] = loan_amount_rupees
            
            # Make prediction - initialize model if not available
            if loan_model.model is None:
                print("Model not loaded, attempting to initialize...")
                initialize_model()
            
            if loan_model.model is not None:
                result, confidence, probabilities = loan_model.predict_single(applicant_data)
                
                return render_template('result.html', 
                                     result=result,
                                     confidence=round(confidence, 2),
                                     applicant_data=applicant_data_display,
                                     probabilities=probabilities)
            else:
                return render_template('predict.html', error="Model not available. Please train the model first.")
        
        except Exception as e:
            return render_template('predict.html', error=f"Error making prediction: {str(e)}")
    
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    try:
        data = request.json
        
        # Initialize model if not available
        if loan_model.model is None:
            print("API: Model not loaded, attempting to initialize...")
            initialize_model()
            
        if loan_model.model is not None:
            result, confidence, probabilities = loan_model.predict_single(data)
            
            return jsonify({
                'success': True,
                'prediction': result,
                'confidence': round(confidence, 2),
                'probabilities': {
                    'rejected': round(probabilities[0] * 100, 2),
                    'approved': round(probabilities[1] * 100, 2)
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Model not available'
            }), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    try:
        # Check if model is available
        if loan_model.model is None:
            return render_template('analytics.html', error="Model not available")
        
        # Initialize model if not available
        if loan_model.model is None:
            print("Analytics: Model not loaded, attempting to initialize...")
            initialize_model()
        
        # Get feature importance using the model's method
        feature_importance = loan_model.get_feature_importance()
        
        # Get tree rules using the model's method
        tree_rules = loan_model.get_tree_rules(max_depth=3)
        
        return render_template('analytics.html', 
                             feature_importance=feature_importance,
                             tree_rules=tree_rules)
    
    except Exception as e:
        return render_template('analytics.html', error=f"Error loading analytics: {str(e)}")

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/train')
def train():
    """Model training page"""
    return render_template('train.html')

@app.route('/api/train', methods=['POST'])
def api_train():
    """API endpoint for model training"""
    try:
        # Check if training data exists
        if not os.path.exists('data/loan_train.csv'):
            return jsonify({
                'success': False,
                'error': 'Training data not found. Please generate data first.'
            }), 400
        
        # Load and train model
        df = loan_model.load_data('data/loan_train.csv')
        processed_data = loan_model.preprocess_data(df)
        X, y = loan_model.prepare_features(processed_data)
        results = loan_model.train_model(X, y)
        
        # Save model
        loan_model.save_model()
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully!',
            'metrics': {
                'accuracy': round(results['accuracy'], 4),
                'precision': round(results['precision'], 4),
                'recall': round(results['recall'], 4),
                'f1_score': round(results['f1_score'], 4)
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Initialize the model before starting the server
    print("Initializing loan prediction model...")
    initialize_model()
    
    # Start the Flask server
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)