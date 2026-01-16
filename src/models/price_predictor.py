"""
Price prediction module using ensemble regression models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class PricePredictor:
    """
    Car price prediction using ensemble regression models.
    """
    
    def __init__(self, use_xgboost: bool = True, random_state: int = 42):
        """
        Initialize the price predictor.
        
        Args:
            use_xgboost: Whether to use XGBoost (if available)
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.use_xgboost = use_xgboost and HAS_XGBOOST
        self.models: Dict[str, object] = {}
        self.best_model_name: Optional[str] = None
        self.feature_names: List[str] = []
        self.is_fitted = False
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the regression models."""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            'ridge': Ridge(alpha=1.0)
        }
        
        if self.use_xgboost:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
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
            y: Target Series (prices)
            validate: Whether to perform validation
            validation_size: Proportion of data for validation
            
        Returns:
            Dictionary with training results for each model
        """
        self.feature_names = list(X.columns)
        results = {}
        
        # Handle missing values
        X_clean = X.fillna(X.median())
        
        if validate:
            X_train, X_val, y_train, y_val = train_test_split(
                X_clean, y, test_size=validation_size, random_state=self.random_state
            )
        else:
            X_train, y_train = X_clean, y
            X_val, y_val = None, None
        
        best_r2 = -np.inf
        
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
                    
                    # Track best model based on validation R²
                    if val_metrics['r2'] > best_r2:
                        best_r2 = val_metrics['r2']
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
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Dict]:
        """
        Perform cross-validation for all models.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            cv: Number of folds
            
        Returns:
            Cross-validation results for each model
        """
        X_clean = X.fillna(X.median())
        cv_results = {}
        
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X_clean, y, cv=cv, scoring='r2')
                cv_results[name] = {
                    'mean_r2': scores.mean(),
                    'std_r2': scores.std(),
                    'scores': scores.tolist()
                }
            except Exception as e:
                cv_results[name] = {'error': str(e)}
        
        return cv_results
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make price predictions.
        
        Args:
            X: Feature DataFrame
            model_name: Name of model to use (uses best if None)
            
        Returns:
            Array of predicted prices
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        model_name = model_name or self.best_model_name
        model = self.models[model_name]
        
        X_clean = X.fillna(X.median())
        return model.predict(X_clean)
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Args:
            model_name: Name of model to use (uses best if None)
            
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
            importance = np.abs(model.coef_)
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
            'feature_names': self.feature_names
        }
        
        joblib.dump(save_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'PricePredictor':
        """Load a trained model from disk."""
        save_data = joblib.load(filepath)
        
        predictor = cls()
        predictor.models[save_data['model_name']] = save_data['model']
        predictor.best_model_name = save_data['model_name']
        predictor.feature_names = save_data['feature_names']
        predictor.is_fitted = True
        
        return predictor
