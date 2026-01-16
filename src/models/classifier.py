"""
Performance category classification module.
Classifies cars into: High-Performance, Mid-Range, Economy
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class PerformanceClassifier:
    """
    Car performance category classifier.
    Categories: High-Performance, Mid-Range, Economy
    """
    
    CATEGORIES = ['High-Performance', 'Mid-Range', 'Economy']
    
    def __init__(self, use_xgboost: bool = True, random_state: int = 42):
        """
        Initialize the performance classifier.
        
        Args:
            use_xgboost: Whether to use XGBoost (if available)
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.use_xgboost = use_xgboost and HAS_XGBOOST
        self.models: Dict[str, object] = {}
        self.best_model_name: Optional[str] = None
        self.feature_names: List[str] = []
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the classification models."""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                multi_class='multinomial',
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state
            )
        }
        
        if self.use_xgboost:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
    
    def train(self,
              X: pd.DataFrame,
              y: pd.Series,
              validate: bool = True,
              validation_size: float = 0.2) -> Dict[str, Dict]:
        """
        Train all models and select the best one.
        
        Args:
            X: Feature DataFrame
            y: Target Series (performance categories)
            validate: Whether to perform validation
            validation_size: Proportion of data for validation
            
        Returns:
            Dictionary with training results for each model
        """
        self.feature_names = list(X.columns)
        results = {}
        
        # Handle missing values
        X_clean = X.fillna(X.median())
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        if validate:
            X_train, X_val, y_train, y_val = train_test_split(
                X_clean, y_encoded, test_size=validation_size, 
                random_state=self.random_state, stratify=y_encoded
            )
        else:
            X_train, y_train = X_clean, y_encoded
            X_val, y_val = None, None
        
        best_f1 = -np.inf
        
        for name, model in self.models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Calculate training metrics
                train_pred = model.predict(X_train)
                train_metrics = self._calculate_metrics(y_train, train_pred)
                
                results[name] = {
                    'train_metrics': train_metrics,
                    'status': 'success'
                }
                
                # Validation metrics
                if validate and X_val is not None:
                    val_pred = model.predict(X_val)
                    val_metrics = self._calculate_metrics(y_val, val_pred)
                    results[name]['val_metrics'] = val_metrics
                    
                    # Track best model based on F1 score
                    if val_metrics['f1_weighted'] > best_f1:
                        best_f1 = val_metrics['f1_weighted']
                        self.best_model_name = name
                
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Fallback if no model performed well
        if self.best_model_name is None:
            self.best_model_name = 'random_forest'
        
        self.is_fitted = True
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Dict]:
        """
        Perform stratified cross-validation for all models.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            cv: Number of folds
            
        Returns:
            Cross-validation results for each model
        """
        X_clean = X.fillna(X.median())
        y_encoded = self.label_encoder.fit_transform(y)
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X_clean, y_encoded, cv=skf, scoring='f1_weighted')
                cv_results[name] = {
                    'mean_f1': scores.mean(),
                    'std_f1': scores.std(),
                    'scores': scores.tolist()
                }
            except Exception as e:
                cv_results[name] = {'error': str(e)}
        
        return cv_results
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        Predict performance categories.
        
        Args:
            X: Feature DataFrame
            model_name: Name of model to use (uses best if None)
            
        Returns:
            Array of predicted category labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        model_name = model_name or self.best_model_name
        model = self.models[model_name]
        
        X_clean = X.fillna(X.median())
        predictions_encoded = model.predict(X_clean)
        
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def predict_proba(self, X: pd.DataFrame, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get class probabilities.
        
        Args:
            X: Feature DataFrame
            model_name: Name of model to use
            
        Returns:
            DataFrame with class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        model_name = model_name or self.best_model_name
        model = self.models[model_name]
        
        X_clean = X.fillna(X.median())
        probas = model.predict_proba(X_clean)
        
        return pd.DataFrame(
            probas, 
            columns=self.label_encoder.classes_
        )
    
    def get_confusion_matrix(self, X: pd.DataFrame, y: pd.Series, 
                              model_name: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Get confusion matrix for predictions.
        
        Args:
            X: Feature DataFrame
            y: True labels
            model_name: Name of model to use
            
        Returns:
            Tuple of (confusion matrix, class labels)
        """
        predictions = self.predict(X, model_name)
        cm = confusion_matrix(y, predictions, labels=self.CATEGORIES)
        return cm, self.CATEGORIES
    
    def get_classification_report(self, X: pd.DataFrame, y: pd.Series,
                                   model_name: Optional[str] = None) -> str:
        """
        Get detailed classification report.
        
        Args:
            X: Feature DataFrame
            y: True labels
            model_name: Name of model to use
            
        Returns:
            Classification report as string
        """
        predictions = self.predict(X, model_name)
        return classification_report(y, predictions, target_names=self.CATEGORIES)
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Args:
            model_name: Name of model to use
            
        Returns:
            DataFrame with feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        model_name = model_name or self.best_model_name
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str, model_name: Optional[str] = None):
        """Save the trained model to disk."""
        model_name = model_name or self.best_model_name
        model = self.models[model_name]
        
        save_data = {
            'model': model,
            'model_name': model_name,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder
        }
        
        joblib.dump(save_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'PerformanceClassifier':
        """Load a trained model from disk."""
        save_data = joblib.load(filepath)
        
        classifier = cls()
        classifier.models[save_data['model_name']] = save_data['model']
        classifier.best_model_name = save_data['model_name']
        classifier.feature_names = save_data['feature_names']
        classifier.label_encoder = save_data['label_encoder']
        classifier.is_fitted = True
        
        return classifier
