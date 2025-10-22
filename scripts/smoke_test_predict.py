import sys, os
sys.path.append(r"c:\Users\user\OneDrive\Desktop\Bee_trap\loan_sanctioning_prediction")
from models.decision_tree_model import DecisionTreeModel
m=DecisionTreeModel(model_path=r"c:\Users\user\OneDrive\Desktop\Bee_trap\loan_sanctioning_prediction\models\decision_tree_with_scores.joblib")
print('pipeline loaded:', m.pipeline is not None)
print('numeric_features:', m.numeric_features)
print('categorical_features:', m.categorical_features)
sample={'Gender':'Male','Married':'Yes','Dependents':'0','Education':'Graduate','Self_Employed':'No','ApplicantIncome':50000,'CoapplicantIncome':0,'LoanAmount':150.0,'Loan_Amount_Term':360,'Credit_Score':750,'Property_Area':'Urban','Purpose':'Home Purchase'}
print('predict:', m.predict_single(sample))
