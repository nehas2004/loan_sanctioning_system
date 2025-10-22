import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def generate_loan_dataset(n_samples=1000):
    """Generate synthetic loan dataset for training"""
    np.random.seed(42)
    
    # Generate features
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
        'Married': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.75, 0.25]),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
        'ApplicantIncome': np.random.exponential(5000, n_samples).astype(int),
        'CoapplicantIncome': np.random.exponential(2000, n_samples).astype(int),
        'LoanAmount': np.random.normal(150, 50, n_samples).clip(20, 500).astype(int),
        'Loan_Amount_Term': np.random.choice([120, 180, 240, 300, 360, 480], n_samples, p=[0.05, 0.1, 0.15, 0.2, 0.45, 0.05]),
        'Credit_History': np.random.choice([0, 1], n_samples, p=[0.15, 0.85]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples, p=[0.4, 0.35, 0.25])
        ,
        # Purpose of loan (new feature)
        'Purpose': np.random.choice([
            'Home Purchase', 'Home Renovation', 'Education', 'Business', 'Medical', 'Car', 'Debt Consolidation', 'Wedding', 'Other'
        ], n_samples, p=[0.2, 0.15, 0.15, 0.15, 0.05, 0.1, 0.1, 0.05, 0.05])
    }
    
    df = pd.DataFrame(data)
    
    # Generate loan status based on realistic criteria
    loan_approved_prob = 0.7  # Base probability
    
    # Adjust probability based on features
    prob_adjustments = np.ones(n_samples) * loan_approved_prob
    
    # Income factor
    prob_adjustments += (df['ApplicantIncome'] + df['CoapplicantIncome']) / 20000 * 0.2
    
    # Credit history factor
    prob_adjustments += df['Credit_History'] * 0.3
    
    # Education factor
    prob_adjustments += (df['Education'] == 'Graduate').astype(int) * 0.1
    
    # Loan amount factor (higher amount = lower approval chance)
    prob_adjustments -= df['LoanAmount'] / 500 * 0.15
    
    # Self employed factor (slightly negative)
    prob_adjustments -= (df['Self_Employed'] == 'Yes').astype(int) * 0.05
    
    # Married factor (positive)
    prob_adjustments += (df['Married'] == 'Yes').astype(int) * 0.05
    
    # Dependents factor (more dependents = slightly negative)
    dependents_numeric = df['Dependents'].replace('3+', '3').astype(int)
    prob_adjustments -= dependents_numeric * 0.02
    
    # Property area factor
    area_factor = df['Property_Area'].map({'Urban': 0.05, 'Semiurban': 0, 'Rural': -0.05})
    prob_adjustments += area_factor

    # Purpose factor: some purposes (education, essential medical) may increase approval likelihood
    purpose_factor = df['Purpose'].map({
        'Education': 0.05,
        'Medical': 0.08,
        'Home Purchase': 0.03,
        'Home Renovation': 0.01,
        'Car': -0.01,
        'Business': 0.0,
        'Debt Consolidation': -0.02,
        'Wedding': -0.02,
        'Other': 0.0
    })
    prob_adjustments += purpose_factor
    
    # Clip probabilities
    prob_adjustments = np.clip(prob_adjustments, 0.1, 0.9)
    
    # Generate loan status
    df['Loan_Status'] = np.random.binomial(1, prob_adjustments, n_samples)
    df['Loan_Status'] = df['Loan_Status'].map({1: 'Y', 0: 'N'})
    
    return df

def create_training_data():
    """Create and save training dataset"""
    # Generate dataset
    df = generate_loan_dataset(1000)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    df.to_csv('data/loan_train.csv', index=False)
    
    # Create train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Loan_Status'])
    
    # Save splits
    train_df.to_csv('data/loan_train_split.csv', index=False)
    test_df.to_csv('data/loan_test_split.csv', index=False)
    
    print(f"Dataset created successfully!")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Approval rate: {(df['Loan_Status'] == 'Y').mean():.2%}")
    
    return df, train_df, test_df

if __name__ == "__main__":
    df, train_df, test_df = create_training_data()
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())