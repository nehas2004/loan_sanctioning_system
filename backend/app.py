from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simple_loan_model import SimpleRuleBasedLoanModel

app = Flask(__name__, template_folder='../templates', static_folder='../static')
CORS(app)

# Production configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'loan-ml-secret-key-2024')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# Initialize model
loan_model = SimpleRuleBasedLoanModel()

# Initialize rule-based model (no training needed)
def initialize_model():
    print("Rule-based loan model initialized successfully!")
    print("Model uses financial industry standards for loan approval decisions.")

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
            # Rule-based model is always ready
            result, confidence, probabilities = loan_model.predict_single(applicant_data)
            
            return render_template('result.html', 
                                 result=result,
                                 confidence=round(confidence, 2),
                                 applicant_data=applicant_data_display,
                                 probabilities=probabilities)
        
        except Exception as e:
            return render_template('predict.html', error=f"Error making prediction: {str(e)}")
    
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    try:
        data = request.json
        
        # Rule-based model is always ready
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
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    try:
        # Rule-based model is always available
        
        # Get feature importance using the model's method
        feature_importance = loan_model.get_feature_importance()
        
        # Get decision rules using the model's method
        decision_rules = loan_model.get_decision_rules()
        
        return render_template('analytics.html', 
                             feature_importance=feature_importance,
                             decision_rules=decision_rules)
    
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
    """API endpoint for model training (rule-based model doesn't need training)"""
    try:        
        return jsonify({
            'success': True,
            'message': 'Rule-based model is ready to use!',
            'info': 'This model uses financial industry standards and does not require training.',
            'rules': loan_model.get_decision_rules()['rules_summary']
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