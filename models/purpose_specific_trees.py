"""Train and visualize separate decision trees for each loan purpose"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
import os
import joblib

def train_purpose_specific_trees(data_path='data/loan_train_split_with_scores.csv'):
    """Train separate decision trees for each loan purpose and save visualizations"""
    
    # Read training data
    train_data = pd.read_csv(data_path)
    
    # Define features
    numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                       'Loan_Amount_Term', 'Credit_Score']
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education',
                           'Self_Employed', 'Property_Area']
    
    # Create output directory
    os.makedirs('models/purpose_trees', exist_ok=True)
    
    # Get unique purposes
    purposes = train_data['Purpose'].unique()
    
    purpose_models = {}
    
    for purpose in purposes:
        print(f"\nTraining tree for {purpose} loans:")
        
        # Filter data for this purpose
        purpose_data = train_data[train_data['Purpose'] == purpose]
        
        if len(purpose_data) < 20:  # Skip if too few samples
            print(f"Insufficient data for {purpose} (only {len(purpose_data)} samples)")
            continue
            
        # Prepare features
        X = purpose_data[numeric_features + categorical_features].copy()
        y = purpose_data['Loan_Status']
        
        # Convert categorical features
        for col in categorical_features:
            X[col] = pd.Categorical(X[col]).codes
        
        # Train a decision tree (slightly more complex than main model)
        clf = DecisionTreeClassifier(
            max_depth=4,  # Smaller depth for interpretability
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        
        # Fit the tree
        clf.fit(X, y)
        
        # Save the rules as text
        rules = export_text(clf, feature_names=list(X.columns))
        with open(f'models/purpose_trees/{purpose.lower().replace(" ", "_")}_rules.txt', 'w') as f:
            f.write(f"Decision Rules for {purpose} Loans\n")
            f.write("=" * 50 + "\n\n")
            f.write(rules)
        
        # Create and save tree visualization
        plt.figure(figsize=(20,10))
        plot_tree(clf, 
                 feature_names=list(X.columns),
                 class_names=['N', 'Y'],
                 filled=True,
                 rounded=True,
                 fontsize=10)
        plt.title(f'Decision Tree for {purpose} Loans')
        plt.savefig(f'models/purpose_trees/{purpose.lower().replace(" ", "_")}_tree.png',
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # Store some metrics
        purpose_models[purpose] = {
            'samples': len(purpose_data),
            'approval_rate': (y == 'Y').mean() * 100,
            'feature_importance': dict(zip(X.columns, clf.feature_importances_))
        }
        
        print(f"- Samples: {len(purpose_data)}")
        print(f"- Approval rate: {purpose_models[purpose]['approval_rate']:.1f}%")
        print("- Top 3 important features:")
        imp = purpose_models[purpose]['feature_importance']
        for feat, imp in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  - {feat}: {imp:.3f}")
    
    # Save summary
    with open('models/purpose_trees/summary.txt', 'w') as f:
        f.write("Loan Purpose Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for purpose, metrics in purpose_models.items():
            f.write(f"\n{purpose}\n")
            f.write("-" * len(purpose) + "\n")
            f.write(f"Number of applications: {metrics['samples']}\n")
            f.write(f"Approval rate: {metrics['approval_rate']:.1f}%\n")
            f.write("Top 3 important features:\n")
            imp = metrics['feature_importance']
            for feat, imp in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:3]:
                f.write(f"- {feat}: {imp:.3f}\n")
            f.write("\n")
    
    return purpose_models

if __name__ == "__main__":
    # Train trees
    purpose_models = train_purpose_specific_trees()
    print("\nAnalysis complete! Check models/purpose_trees/ for visualizations and rules.")