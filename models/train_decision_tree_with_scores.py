"""Train decision tree model with credit scores"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the data
train_data = pd.read_csv('data/loan_train_split_with_scores.csv')
test_data = pd.read_csv('data/loan_test_split_with_scores.csv')

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

# Save the model
joblib.dump(pipeline, 'models/decision_tree_with_scores.joblib')

# Print model performance
train_accuracy = pipeline.score(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)

print(f"Train accuracy: {train_accuracy:.2%}")
print(f"Test accuracy: {test_accuracy:.2%}")

# Save feature names for later use
feature_names = (numeric_features + 
                [f"{feat}_{val}" for feat, vals in 
                 zip(categorical_features,
                     pipeline.named_steps['preprocessor']
                     .named_transformers_['cat'].categories_) 
                 for val in vals[1:]])

joblib.dump({
    'pipeline': pipeline,
    'feature_names': feature_names,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features
}, 'models/decision_tree_with_scores.joblib')