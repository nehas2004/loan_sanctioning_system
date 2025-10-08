"""Risk Assessment module using Apriori algorithm"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

class RiskAssessment:
    def __init__(self):
        self.rules = None
        self.risk_thresholds = {
            'high': 0.7,    # confidence threshold for high risk
            'medium': 0.4,  # confidence threshold for medium risk
            'low': 0.0      # anything below medium is low risk
        }
    
    def prepare_data(self, df):
        """Convert loan data into binary format for apriori"""
        binary_data = pd.DataFrame()
        
        # Income brackets
        binary_data['high_income'] = (df['ApplicantIncome'] > df['ApplicantIncome'].quantile(0.75)).astype(int)
        binary_data['low_income'] = (df['ApplicantIncome'] < df['ApplicantIncome'].quantile(0.25)).astype(int)
        
        # Loan amount brackets
        binary_data['high_loan'] = (df['LoanAmount'] > df['LoanAmount'].quantile(0.75)).astype(int)
        binary_data['low_loan'] = (df['LoanAmount'] < df['LoanAmount'].quantile(0.25)).astype(int)
        
        # Credit score - convert to good credit based on score >= 650
        binary_data['good_credit'] = (df['Credit_Score'] >= 650).astype(int)
        
        # Education
        binary_data['graduate'] = (df['Education'] == 'Graduate').astype(int)
        
        # Employment
        binary_data['self_employed'] = (df['Self_Employed'] == 'Yes').astype(int)
        
        # Property area
        binary_data['urban'] = (df['Property_Area'] == 'Urban').astype(int)
        binary_data['rural'] = (df['Property_Area'] == 'Rural').astype(int)
        
        # Loan approved/rejected (only include if Loan_Status exists in training data)
        if 'Loan_Status' in df.columns:
            binary_data['loan_approved'] = (df['Loan_Status'] == 'Y').astype(int)
        
        return binary_data

    def train(self, df):
        """Train the risk assessment model using apriori algorithm"""
        binary_data = self.prepare_data(df)
        
        # Generate frequent itemsets
        frequent_itemsets = apriori(binary_data, min_support=0.1, use_colnames=True)
        
        # Generate association rules
        self.rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
        
        # Sort rules by confidence
        self.rules = self.rules.sort_values(['confidence', 'lift'], ascending=[False, False])
    
    def assess_risk(self, applicant_data):
        """Assess risk category for a loan application"""
        if self.rules is None:
            return 'unknown'
        
        # Convert applicant data to binary format
        binary_data = self.prepare_data(pd.DataFrame([applicant_data]))
        
        # Calculate risk score based on matching rules
        risk_score = 0
        matches = 0
        
        for _, rule in self.rules.iterrows():
            antecedent = rule['antecedents']
            matches_rule = True
            
            # Check if applicant matches rule antecedents
            for item in antecedent:
                if item in binary_data.columns and binary_data[item].iloc[0] != 1:
                    matches_rule = False
                    break
            
            if matches_rule:
                risk_score += rule['confidence']
                matches += 1
        
        if matches > 0:
            avg_risk_score = risk_score / matches
        else:
            avg_risk_score = 0.5  # Default to medium risk if no rules match
        
        # Categorize risk based on score
        if avg_risk_score >= self.risk_thresholds['high']:
            return 'high'
        elif avg_risk_score >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'

    def get_risk_color(self, risk_level):
        """Get bootstrap color class for risk level"""
        colors = {
            'high': 'danger',
            'medium': 'warning',
            'low': 'success',
            'unknown': 'secondary'
        }
        return colors.get(risk_level, 'secondary')