"""Script to convert binary credit history to credit scores in dataset"""
import pandas as pd
import numpy as np

def convert_credit_history_to_score(credit_history, loan_amount, applicant_income):
    """Convert binary credit history to a more realistic credit score"""
    base_score = 650  # Base score
    
    # Add randomness based on loan amount and income ratio
    loan_to_income = loan_amount / (applicant_income + 1)  # Adding 1 to avoid division by zero
    score_variance = np.random.normal(0, 30)  # Random variance
    
    if credit_history == 1:  # Good credit history
        score = base_score + 150 + score_variance
        # Higher loan-to-income ratio slightly reduces score for good credit
        score -= loan_to_income * 10
    else:  # Poor credit history
        score = base_score - 150 + score_variance
        # Higher loan-to-income ratio significantly reduces score for poor credit
        score -= loan_to_income * 20
    
    # Ensure score is within valid range (300-900)
    return min(max(int(round(score)), 300), 900)

# Read the training data
train_data = pd.read_csv('loan_train_split.csv')
test_data = pd.read_csv('loan_test_split.csv')
sample_data = pd.read_csv('sample_loan_data.csv')

# Convert credit history to credit score for each dataset
for dataset in [train_data, test_data, sample_data]:
    credit_scores = []
    for _, row in dataset.iterrows():
        score = convert_credit_history_to_score(
            row['Credit_History'],
            row['LoanAmount'],
            row['ApplicantIncome']
        )
        credit_scores.append(score)
    
    # Replace Credit_History with Credit_Score
    dataset['Credit_Score'] = credit_scores
    dataset.drop('Credit_History', axis=1, inplace=True)

# Save the updated datasets
train_data.to_csv('loan_train_split_with_scores.csv', index=False)
test_data.to_csv('loan_test_split_with_scores.csv', index=False)
sample_data.to_csv('sample_loan_data_with_scores.csv', index=False)

print("Datasets updated with credit scores")