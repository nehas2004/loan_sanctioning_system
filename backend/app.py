from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.decision_tree_model import DecisionTreeModel
from models.risk_assessment import RiskAssessment

app = Flask(__name__, template_folder='../templates', static_folder='../static')
CORS(app)

# Production configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'loan-ml-secret-key-2024')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'decision_tree_with_scores.joblib')

# Initialize model with credit scores using absolute path
loan_model = DecisionTreeModel(model_path=MODEL_PATH)
risk_assessor = RiskAssessment()

@app.route('/decision-trees')
def decision_trees():
    """Route to display decision tree visualizations"""
    return render_template('decision_trees.html')

def initialize_model():
    if loan_model.pipeline is not None:
        print("Decision tree model loaded successfully!")
        try:
            info = loan_model.get_decision_rules()
            print('Model info:', info)
            
            # Train risk assessor on historical data
            try:
                train_data_path = os.path.join(PROJECT_ROOT, 'data', 'loan_train_split_with_scores.csv')
                train_data = pd.read_csv(train_data_path)
                risk_assessor.train(train_data)
                print("Risk assessment model trained successfully!")
            except Exception as e:
                print("Error training risk assessment model:", e)
                
        except Exception:
            pass
    else:
        print("Decision tree model not found - falling back to limited behavior")

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
            # Get form data - loan amount is entered in rupees in the form
            loan_amount_rupees = float(request.form['loan_amount'])
            # Convert to thousands for model input (model was trained with LoanAmount in thousands)
            loan_amount_thousands = loan_amount_rupees / 1000.0
            
            applicant_data = {
                'Gender': request.form['gender'],
                'Married': request.form['married'],
                'Dependents': request.form['dependents'],
                'Education': request.form['education'],
                'Self_Employed': request.form['self_employed'],
                'Purpose': request.form.get('purpose'),
                'ApplicantIncome': float(request.form['applicant_income']),
                'CoapplicantIncome': float(request.form['coapplicant_income']),
                'LoanAmount': loan_amount_thousands,  # Store in thousands for model
                'Loan_Amount_Term': int(request.form['loan_amount_term']),
                'Credit_Score': int(request.form['credit_score']),
                'Property_Area': request.form['property_area']
            }
            
            # For display purposes, store the original rupee amount
            applicant_data_display = applicant_data.copy()
            applicant_data_display['LoanAmount'] = loan_amount_rupees
            
            # Make prediction using decision tree model (pipeline handles preprocessing)
            result, confidence, probabilities = loan_model.predict_single(applicant_data)
            
            # Assess risk level
            risk_level = risk_assessor.assess_risk(applicant_data)
            risk_color = risk_assessor.get_risk_color(risk_level)
            
            # Compute EMI for display (loan_amount_thousands -> rupees)
            try:
                # loan_amount_rupees already holds rupee amount
                term = int(request.form['loan_amount_term'])
                rate = 0.08
                monthly_rate = rate / 12
                emi = (loan_amount_rupees * monthly_rate * (1 + monthly_rate) ** term) / ((1 + monthly_rate) ** term - 1)
                total_income = float(request.form.get('applicant_income', 0)) + float(request.form.get('coapplicant_income', 0) or 0)
                emi_to_income = (emi / total_income) * 100 if total_income > 0 else None
            except Exception:
                emi = None
                emi_to_income = None

            return render_template('result.html', 
                                result=result,
                                confidence=round(confidence, 2),
                                applicant_data=applicant_data_display,
                                probabilities=probabilities,
                                emi=emi,
                                emi_to_income=emi_to_income,
                                risk_level=risk_level,
                                risk_color=risk_color)
        except Exception as e:
            return render_template('predict.html', error=f"Error making prediction: {str(e)}")
    
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    try:
        data = request.json

        # Normalize incoming data so model receives LoanAmount in thousands
        try:
            # Determine loan amount in rupees and thousands
            if data.get('loan_amount') is not None:
                loan_amount_rupees = float(data.get('loan_amount') or 0)
                loan_amount_thousands = loan_amount_rupees / 1000.0
            else:
                # If caller passed LoanAmount (likely in thousands), use that
                loan_amount_thousands = float(data.get('LoanAmount', 0) or 0)
                loan_amount_rupees = loan_amount_thousands * 1000

            # Build applicant_data dict expected by the model
            applicant_data = {
                'Gender': data.get('Gender', data.get('gender')),
                'Married': data.get('Married', data.get('married')),
                'Dependents': data.get('Dependents', data.get('dependents')),
                'Education': data.get('Education', data.get('education')),
                'Self_Employed': data.get('Self_Employed', data.get('self_employed')),
                'Purpose': data.get('Purpose', data.get('purpose')),
                'ApplicantIncome': float(data.get('ApplicantIncome', data.get('applicant_income', 0)) or 0),
                'CoapplicantIncome': float(data.get('CoapplicantIncome', data.get('coapplicant_income', 0)) or 0),
                'LoanAmount': float(loan_amount_thousands),
                'Loan_Amount_Term': int(data.get('Loan_Amount_Term', data.get('loan_amount_term', 360)) or 360),
                'Credit_Score': int(data.get('Credit_Score', data.get('credit_score', 0)) or 0),
                'Property_Area': data.get('Property_Area', data.get('property_area'))
            }

        except Exception:
            # Fallback to raw data if normalization fails
            applicant_data = data

        # Use Decision Tree model if available
        result, confidence, probabilities = loan_model.predict_single(applicant_data)
        
        # Assess risk level
        risk_level = risk_assessor.assess_risk(applicant_data)
        risk_color = risk_assessor.get_risk_color(risk_level)
        
        # Compute EMI if enough data provided
        emi = None
        emi_to_income = None
        try:
            term = int(applicant_data.get('Loan_Amount_Term', 360) or 360)
            applicant_income = float(applicant_data.get('ApplicantIncome', 0) or 0)
            coapplicant_income = float(applicant_data.get('CoapplicantIncome', 0) or 0)
            if loan_amount_rupees > 0 and term > 0 and (applicant_income + coapplicant_income) > 0:
                monthly_rate = 0.08 / 12
                emi = (loan_amount_rupees * monthly_rate * (1 + monthly_rate) ** term) / ((1 + monthly_rate) ** term - 1)
                total_income = applicant_income + coapplicant_income
                emi_to_income = (emi / total_income) * 100 if total_income > 0 else None
        except Exception:
            pass

        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': round(confidence, 2),
            'probabilities': {
                'rejected': round(probabilities[0] * 100, 2),
                'approved': round(probabilities[1] * 100, 2)
            },
            'emi': round(emi, 2) if emi is not None else None,
            'emi_to_income_percent': round(emi_to_income, 2) if emi_to_income is not None else None,
            'risk_level': risk_level,
            'risk_color': risk_color
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
        # Get feature importance using the model's method
        feature_importance_raw = loan_model.get_feature_importance()

        # Normalize feature importance into a list of {feature, importance}
        feature_importance = None
        try:
            if isinstance(feature_importance_raw, dict):
                # If dict contains numeric importances, convert to sorted list
                numeric_items = [(k, v) for k, v in feature_importance_raw.items() if isinstance(v, (int, float))]
                if numeric_items:
                    feature_importance = [
                        {'feature': k, 'importance': round(float(v), 1)}
                        for k, v in sorted(numeric_items, key=lambda kv: kv[1], reverse=True)
                    ]
                else:
                    # Not numeric data (e.g., a note), leave as None so template can hide section
                    feature_importance = None
            else:
                # If already a list-like structure, pass through
                feature_importance = feature_importance_raw
        except Exception:
            feature_importance = None

        # Get decision rules / metadata using the model's method
        decision_rules = loan_model.get_decision_rules()

        # Provide a textual version of rules for the template (template expects `tree_rules`)
        tree_rules = None
        try:
            if isinstance(decision_rules, dict):
                # If rules are encapsulated under a key, try to extract
                tree_rules = decision_rules.get('rules') or decision_rules.get('text') or str(decision_rules)
            else:
                tree_rules = str(decision_rules)
        except Exception:
            tree_rules = str(decision_rules)

        return render_template('analytics.html', 
                             feature_importance=feature_importance,
                             decision_rules=decision_rules,
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
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.tree import DecisionTreeClassifier
        import joblib
        
        # Load the training data
        train_data_path = os.path.join(PROJECT_ROOT, 'data', 'loan_train_split_with_scores.csv')
        test_data_path = os.path.join(PROJECT_ROOT, 'data', 'loan_test_split_with_scores.csv')
        
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        
        # Define the features
        numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                           'Loan_Amount_Term', 'Credit_Score']
        categorical_features = ['Gender', 'Married', 'Dependents', 'Education',
                               'Self_Employed', 'Property_Area', 'Purpose']
        
        # Create preprocessing transformers
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42
            ))
        ])
        
        # Prepare the data
        X_train = train_data.drop(['Loan_Status'], axis=1)
        y_train = train_data['Loan_Status']
        
        X_test = test_data.drop(['Loan_Status'], axis=1)
        y_test = test_data['Loan_Status']
        
        # Fit the pipeline
        pipeline.fit(X_train, y_train)
        
        # Calculate accuracies
        train_accuracy = pipeline.score(X_train, y_train)
        test_accuracy = pipeline.score(X_test, y_test)
        
        # Save feature names
        feature_names = (numeric_features + 
                        [f"{feat}_{val}" for feat, vals in 
                         zip(categorical_features,
                             pipeline.named_steps['preprocessor']
                             .named_transformers_['cat'].categories_) 
                         for val in vals[1:]])
        
        # Save the model
        model_save_path = os.path.join(PROJECT_ROOT, 'models', 'decision_tree_with_scores.joblib')
        joblib.dump({
            'pipeline': pipeline,
            'feature_names': feature_names,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features
        }, model_save_path)
        
        # Calculate predictions for metrics
        y_pred = pipeline.predict(X_test)
        
        # Calculate detailed metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_test, y_pred, pos_label='Y')
        recall = recall_score(y_test, y_pred, pos_label='Y')
        f1 = f1_score(y_test, y_pred, pos_label='Y')
        
        # Reload the model in the app
        global loan_model
        loan_model = DecisionTreeModel(model_path=model_save_path)
        
        # Retrain risk assessor
        risk_assessor.train(train_data)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully!',
            'metrics': {
                'accuracy': round(test_accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4)
            },
            'train_accuracy': round(train_accuracy * 100, 2),
            'test_accuracy': round(test_accuracy * 100, 2),
            'samples_trained': len(train_data),
            'samples_tested': len(test_data)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    # Initialize the model before starting the server
    print("Initializing loan prediction model...")
    initialize_model()
    
    # Start the Flask server
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)