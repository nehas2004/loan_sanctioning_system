import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from joblib import load
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'decision_tree_with_scores.joblib')
model_dict = load(model_path)

# Extract the actual decision tree classifier from the pipeline
clf = model_dict['pipeline'].named_steps['classifier']
feature_names = model_dict['feature_names']

# Load the training data
train_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'loan_train_split_with_scores.csv')
data = pd.read_csv(train_data_path)

# Preprocess the data using the pipeline's preprocessor
preprocessor = model_dict['pipeline'].named_steps['preprocessor']

# Create a figure with a larger size
plt.figure(figsize=(20,10))

# Plot the decision tree
plot_tree(clf, 
          feature_names=feature_names,
          class_names=['Not Approved', 'Approved'],
          filled=True,
          rounded=True,
          fontsize=10)

# Save the plot
output_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'decision_tree_visualization.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

# Now let's create separate trees for different loan amounts
amount_ranges = [(0, 100000), (100000, 300000), (300000, float('inf'))]
amount_labels = ['Small', 'Medium', 'Large']

for (min_amount, max_amount), label in zip(amount_ranges, amount_labels):
    # Filter data for this amount range
    mask = (data['LoanAmount'] >= min_amount) & (data['LoanAmount'] < max_amount)
    subset_data = data[mask]
    
    if len(subset_data) > 0:
        # Prepare features for the subset
        X = subset_data[model_dict['numeric_features'] + model_dict['categorical_features']]
        y = subset_data['Loan_Status']
        
        # Preprocess the data
        X_transformed = preprocessor.transform(X)
        
        # Train a new decision tree for this subset
        clf_subset = DecisionTreeClassifier(max_depth=4, random_state=42)
        clf_subset.fit(X_transformed, y)
        
        # Create and save visualization
        plt.figure(figsize=(20,10))
        plot_tree(clf_subset,
                 feature_names=feature_names,
                 class_names=['Not Approved', 'Approved'],
                 filled=True,
                 rounded=True,
                 fontsize=10)
        
        output_path = os.path.join(os.path.dirname(__file__), '..', 'static', f'decision_tree_{label.lower()}_loans.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

print("Decision tree visualizations have been created in the static directory.")