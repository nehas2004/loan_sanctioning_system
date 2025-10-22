import os
import joblib
import pandas as pd

class DecisionTreeModel:
    """Wrapper around a scikit-learn Pipeline stored in models/decision_tree_with_scores.joblib
    Provides a predict_single interface compatible with the previous model but using credit scores.
    """

    def __init__(self, model_path='models/decision_tree_with_scores.joblib'):
        self.model_path = model_path
        self.model_type = 'decision_tree'
        self.pipeline = None
        self.classifier = None
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None
        self._load()

    def _load(self):
        if os.path.exists(self.model_path):
            obj = joblib.load(self.model_path)
            # file stores a dict with pipeline and feature information
            if isinstance(obj, dict):
                self.pipeline = obj.get('pipeline')
                self.feature_names = obj.get('feature_names')
                self.numeric_features = obj.get('numeric_features')
                self.categorical_features = obj.get('categorical_features')
            else:
                self.pipeline = obj

            # try to access classifier
            try:
                self.classifier = self.pipeline.named_steps.get('classifier')
            except Exception:
                self.classifier = None
        else:
            print(f"Decision tree model file not found at {self.model_path}")

    def predict_single(self, applicant_data):
        """Accepts a dict of applicant features and returns (decision, confidence, [reject_prob, approve_prob])
        Expects numeric fields: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_Score
        Categorical fields: Gender, Married, Dependents, Education, Self_Employed, Property_Area
        LoanAmount should be in the same unit used when training (this project uses thousands).
        Credit_Score should be between 300 and 900.
        """
        if self.pipeline is None:
            # fallback: return rejection
            return 'N', 50.0, [0.5, 0.5]

        # Handle old model compatibility
        if 'Credit_Score' not in applicant_data and 'credit_score' in applicant_data:
            applicant_data['Credit_Score'] = applicant_data['credit_score']

        # Build a single-row DataFrame
        features = (self.numeric_features or 
                   ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                    'Loan_Amount_Term', 'Credit_Score']) + \
                                    (self.categorical_features or 
                                     ['Gender', 'Married', 'Dependents', 'Education',
                                        'Self_Employed', 'Property_Area', 'Purpose'])

        row = {f: applicant_data.get(f) for f in features}
        
        # Ensure numeric types for numeric columns
        numeric_cols = (self.numeric_features or 
                       ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                        'Loan_Amount_Term', 'Credit_Score'])
        for col in numeric_cols:
            try:
                row[col] = float(row.get(col, 0) or 0)
            except Exception:
                row[col] = 0.0

        # Validate credit score range
        if 'Credit_Score' in row:
            row['Credit_Score'] = min(max(row['Credit_Score'], 300), 900)

        X = pd.DataFrame([row], columns=features)

        try:
            proba = self.pipeline.predict_proba(X)[0]
            pred = self.pipeline.predict(X)[0]

            # classifier classes might be ['N','Y'] or [0,1]
            classes = None
            try:
                classes = self.classifier.classes_.tolist()
            except Exception:
                classes = None

            # Interpret output
            if classes is not None:
                # find index of positive class (Y or 1)
                if 'Y' in classes:
                    pos_idx = classes.index('Y')
                elif 1 in classes:
                    pos_idx = classes.index(1)
                else:
                    # fallback to last index
                    pos_idx = len(classes) - 1
            else:
                pos_idx = 1 if len(proba) > 1 else 0

            approve_prob = float(proba[pos_idx])
            reject_prob = float(1.0 - approve_prob)

            # Decision label normalization
            decision = 'Y' if (str(pred) == 'Y' or str(pred) == '1' or pred == 1) else 'N'
            confidence = approve_prob * 100 if decision == 'Y' else reject_prob * 100

            return decision, float(confidence), [reject_prob, approve_prob]

        except Exception as e:
            print('Error during DT prediction:', e)
            return 'N', 60.0, [0.6, 0.4]

    def get_feature_importance(self):
        """Return aggregated importance per original feature (attempt)."""
        if self.pipeline is None or self.classifier is None:
            return {}

        try:
            # Attempt to recover feature names after ColumnTransformer
            preprocessor = self.pipeline.named_steps.get('preprocessor')
            classifier = self.classifier

            full_feature_names = []
            if preprocessor is not None:
                for name, transformer, cols in preprocessor.transformers:
                    if name == 'num':
                        full_feature_names.extend(cols)
                    elif name == 'cat':
                        # transformer is a pipeline with onehot
                        try:
                            onehot = transformer.named_steps.get('onehot')
                            cats = onehot.categories_
                            for col, categories in zip(cols, cats):
                                for cat in categories:
                                    full_feature_names.append(f"{col}_{cat}")
                        except Exception:
                            # fallback: include original col names
                            full_feature_names.extend(cols)

            importances = getattr(classifier, 'feature_importances_', None)
            if importances is None or len(importances) != len(full_feature_names):
                # fallback: return empty or approximate weights per numeric features
                return { 'note': 'Feature importances unavailable or mismatch' }

            # Map importances to full_feature_names
            feat_imp = dict(zip(full_feature_names, importances.tolist()))

            # Aggregate by original feature (split on underscore)
            agg = {}
            for fname, imp in feat_imp.items():
                orig = fname.split('_')[0]
                agg[orig] = agg.get(orig, 0.0) + imp

            # Normalize to percentages
            total = sum(agg.values()) or 1.0
            agg_percent = {k: round((v/total)*100, 1) for k,v in agg.items()}
            return agg_percent
        except Exception as e:
            print('Error computing feature importance:', e)
            return {}

    def get_decision_rules(self):
        """Return high-level metadata about the trained tree and a human-readable rule text when possible."""
        if self.pipeline is None or self.classifier is None:
            return {'note': 'Decision tree model not loaded'}

        try:
            clf = self.classifier
            depth = getattr(clf, 'get_depth', lambda: None)()
            n_nodes = getattr(clf, 'tree_', None)
            if n_nodes is not None:
                n_nodes = int(n_nodes.node_count)

            # Attempt to recover feature names for export_text
            feature_names = None
            try:
                preprocessor = self.pipeline.named_steps.get('preprocessor')
                full_feature_names = []
                if preprocessor is not None:
                    for name, transformer, cols in preprocessor.transformers:
                        if name == 'num':
                            full_feature_names.extend(cols)
                        elif name == 'cat':
                            try:
                                onehot = transformer.named_steps.get('onehot')
                                cats = onehot.categories_
                                for col, categories in zip(cols, cats):
                                    for cat in categories:
                                        full_feature_names.append(f"{col}_{cat}")
                            except Exception:
                                full_feature_names.extend(cols)
                if full_feature_names:
                    feature_names = full_feature_names
            except Exception:
                feature_names = None

            # Build rules text using sklearn's export_text when possible
            rules_text = None
            try:
                from sklearn.tree import export_text
                if feature_names is not None and len(feature_names) > 0:
                    rules_text = export_text(clf, feature_names=list(feature_names))
                else:
                    # Export without feature names if unavailable
                    rules_text = export_text(clf)
            except Exception:
                rules_text = None

            result = {
                'model_type': 'DecisionTreeClassifier',
                'tree_depth': depth,
                'n_nodes': n_nodes,
                'notes': f'Model loaded from {os.path.basename(self.model_path)}'
            }
            if rules_text:
                result['rules'] = rules_text

            return result
        except Exception as e:
            print('Error getting decision rules:', e)
            return {'error': str(e)}
