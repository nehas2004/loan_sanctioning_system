import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class LoanPredictionModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.target_names = ['Rejected', 'Approved']
    
    def load_data(self, filepath):
        """Load the loan dataset"""
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, data):
        """Preprocess the data"""
        # Handle missing values
        data = data.fillna({
            'Gender': 'Male',
            'Married': 'No',
            'Dependents': '0',
            'Self_Employed': 'No',
            'LoanAmount': data['LoanAmount'].median(),
            'Loan_Amount_Term': 360,
            'Credit_History': 1
        })
        
        # Encode categorical variables
        categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 
                             'Self_Employed', 'Property_Area']
        
        for col in categorical_columns:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                self.label_encoders[col] = le
        
        # Encode target variable
        if 'Loan_Status' in data.columns:
            le_target = LabelEncoder()
            data['Loan_Status'] = le_target.fit_transform(data['Loan_Status'])
            self.label_encoders['Loan_Status'] = le_target
        
        return data
    
    def prepare_features(self, data):
        """Prepare features and target variables"""
        # Feature columns
        feature_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                          'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                          'Loan_Amount_Term', 'Credit_History', 'Property_Area']
        
        X = data[feature_columns]
        y = data['Loan_Status'] if 'Loan_Status' in data.columns else None
        
        self.feature_names = feature_columns
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train the decision tree model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Hyperparameter tuning
        param_grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
        
        # Grid search with cross-validation
        dt = DecisionTreeClassifier(random_state=random_state)
        grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Best model
        self.model = grid_search.best_estimator_
        
        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validation score:", grid_search.best_score_)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        target_names = ['Rejected', 'Approved']
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict_single(self, applicant_data):
        """Make prediction for a single applicant"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([applicant_data])
            
            # Encode categorical variables
            for col, encoder in self.label_encoders.items():
                if col in df.columns and col != 'Loan_Status':
                    # Handle missing values
                    if df[col].isnull().any():
                        df[col] = df[col].fillna('Unknown')
                    df[col] = encoder.transform(df[col])
            
            # Ensure we have all required features
            features_df = df[self.feature_names]
            
            # Make prediction
            prediction = self.model.predict(features_df)[0]
            probability = self.model.predict_proba(features_df)[0]
            
            # Convert prediction to scalar if it's an array
            if hasattr(prediction, 'item'):
                prediction = prediction.item()
            
            # Decode prediction
            result = self.label_encoders['Loan_Status'].inverse_transform([prediction])[0]
            confidence = float(max(probability)) * 100
            
            # Convert probability array to list for JSON serialization
            probability_list = [float(p) for p in probability]
            
            return result, confidence, probability_list
            
        except Exception as e:
            print(f"Error in predict_single: {e}")
            raise e
    
    def save_model(self, filepath='models/loan_model.pkl'):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/loan_model.pkl'):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            self.target_names = model_data['target_names']
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def get_tree_rules(self, max_depth=3):
        """Extract decision tree rules for analytics"""
        try:
            if self.model is None:
                return []
            
            from sklearn.tree import export_text
            
            # Get text representation of the tree
            tree_rules = export_text(
                self.model, 
                feature_names=self.feature_names,
                max_depth=max_depth,
                spacing=2,
                decimals=2,
                show_weights=True
            )
            
            # Parse rules into a more readable format
            rules_list = []
            lines = tree_rules.split('\n')
            
            for i, line in enumerate(lines[:20]):  # Limit to first 20 lines
                if line.strip() and not line.strip().startswith('|'):
                    rule_text = line.strip()
                    if 'class:' in rule_text:
                        # Extract class and value
                        parts = rule_text.split('class:')
                        if len(parts) > 1:
                            class_info = parts[1].strip()
                            rules_list.append({
                                'rule': rule_text,
                                'type': 'decision',
                                'description': f"Decision: {class_info}"
                            })
                    elif '<=' in rule_text or '>' in rule_text:
                        rules_list.append({
                            'rule': rule_text,
                            'type': 'condition',
                            'description': f"Condition: {rule_text}"
                        })
            
            return rules_list
            
        except Exception as e:
            print(f"Error extracting tree rules: {e}")
            return [{'rule': f'Error: {str(e)}', 'type': 'error', 'description': 'Unable to extract tree rules'}]
    
    def get_feature_importance(self):
        """Get feature importance for analytics"""
        try:
            if self.model is None or not hasattr(self.model, 'feature_importances_'):
                return []
            
            importance = self.model.feature_importances_
            feature_importance = [
                {
                    'feature': name, 
                    'importance': round(imp * 100, 2),
                    'description': self._get_feature_description(name)
                }
                for name, imp in zip(self.feature_names, importance)
            ]
            
            # Sort by importance
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            return feature_importance
            
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return []
    
    def _get_feature_description(self, feature_name):
        """Get human-readable description for features"""
        descriptions = {
            'Gender': 'Applicant gender (Male/Female)',
            'Married': 'Marital status (Yes/No)',
            'Dependents': 'Number of dependents (0/1/2/3+)',
            'Education': 'Education level (Graduate/Not Graduate)',
            'Self_Employed': 'Employment type (Self-employed/Salaried)',
            'ApplicantIncome': 'Primary applicant monthly income',
            'CoapplicantIncome': 'Co-applicant monthly income',
            'LoanAmount': 'Requested loan amount',
            'Loan_Amount_Term': 'Loan repayment term in months',
            'Credit_History': 'Credit history (Good/Poor)',
            'Property_Area': 'Property location (Urban/Semiurban/Rural)'
        }
        return descriptions.get(feature_name, feature_name)

# Example usage
if __name__ == "__main__":
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize model
    loan_model = LoanPredictionModel()
    
    # Load data (you'll need to run generate_data.py first)
    try:
        df = loan_model.load_data('data/loan_train.csv')
        if df is not None:
            # Preprocess data
            processed_data = loan_model.preprocess_data(df)
            
            # Prepare features
            X, y = loan_model.prepare_features(processed_data)
            
            # Train model
            results = loan_model.train_model(X, y)
            
            # Save model
            loan_model.save_model()
            
            print("\nModel training completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("Please run 'python data/generate_data.py' first to create the training data.")