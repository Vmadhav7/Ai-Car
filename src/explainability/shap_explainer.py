"""
SHAP-based model explainability module.
Provides feature importance analysis and individual prediction explanations.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path

import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class SHAPExplainer:
    """
    SHAP-based model explainer for tree and linear models.
    """
    
    def __init__(self, model, feature_names: list, model_type: str = 'tree'):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Trained model (sklearn, XGBoost, etc.)
            feature_names: List of feature names
            model_type: 'tree' for tree-based models, 'linear' for linear models
        """
        if not HAS_SHAP:
            raise ImportError("SHAP library not installed. Run: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.base_value = None
        self._X_sample = None
    
    def fit(self, X: pd.DataFrame, sample_size: int = 100):
        """
        Fit the SHAP explainer on training data.
        
        Args:
            X: Training feature DataFrame
            sample_size: Number of samples for background
        """
        # Sample background data
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        self._X_sample = X_sample.fillna(X_sample.median())
        
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, self._X_sample)
        else:
            # Fallback to KernelExplainer (slower but model-agnostic)
            self.explainer = shap.KernelExplainer(self.model.predict, self._X_sample)
    
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for given data.
        
        Args:
            X: Feature DataFrame to explain
            
        Returns:
            Array of SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        X_clean = X.fillna(X.median())
        self.shap_values = self.explainer.shap_values(X_clean)
        
        # Get base value
        if hasattr(self.explainer, 'expected_value'):
            self.base_value = self.explainer.expected_value
            if isinstance(self.base_value, np.ndarray):
                self.base_value = self.base_value[0]
        
        return self.shap_values
    
    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Handle multi-class output
        if isinstance(self.shap_values, list):
            # Average across classes for classification
            shap_array = np.abs(np.array(self.shap_values)).mean(axis=0)
        else:
            shap_array = self.shap_values
        
        # Mean absolute SHAP value per feature
        importance = np.abs(shap_array).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'std': np.abs(shap_array).std(axis=0)
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def explain_prediction(self, 
                            X_row: pd.DataFrame,
                            prediction: float) -> Dict:
        """
        Explain a single prediction.
        
        Args:
            X_row: Single row DataFrame with features
            prediction: The model's prediction for this row
            
        Returns:
            Dictionary with explanation details
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        X_clean = X_row.fillna(X_row.median())
        shap_vals = self.explainer.shap_values(X_clean)
        
        # Handle multi-class
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        
        shap_vals = shap_vals.flatten()
        
        # Create contribution breakdown
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_clean.values.flatten(),
            'contribution': shap_vals
        }).sort_values('contribution', key=abs, ascending=False)
        
        explanation = {
            'prediction': prediction,
            'base_value': self.base_value,
            'top_positive_factors': contributions[contributions['contribution'] > 0].head(5).to_dict('records'),
            'top_negative_factors': contributions[contributions['contribution'] < 0].head(5).to_dict('records'),
            'all_contributions': contributions.to_dict('records')
        }
        
        return explanation
    
    def plot_summary(self, 
                      X: pd.DataFrame,
                      plot_type: str = 'bar',
                      max_display: int = 10,
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot SHAP summary visualization.
        
        Args:
            X: Feature DataFrame
            plot_type: 'bar' for bar chart, 'dot' for beeswarm plot
            max_display: Maximum features to display
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        X_clean = X.fillna(X.median())
        
        fig = plt.figure(figsize=(10, 8))
        
        # Handle multi-class
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        
        if plot_type == 'bar':
            shap.summary_plot(shap_vals, X_clean, 
                            feature_names=self.feature_names,
                            plot_type='bar', 
                            max_display=max_display,
                            show=False)
        else:
            shap.summary_plot(shap_vals, X_clean,
                            feature_names=self.feature_names,
                            max_display=max_display,
                            show=False)
        
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_force(self,
                    X_row: pd.DataFrame,
                    save_path: Optional[str] = None) -> None:
        """
        Create force plot for a single prediction.
        
        Args:
            X_row: Single row DataFrame
            save_path: Optional path to save HTML
        """
        X_clean = X_row.fillna(X_row.median())
        shap_vals = self.explainer.shap_values(X_clean)
        
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        
        force_plot = shap.force_plot(
            self.base_value,
            shap_vals[0],
            X_clean.iloc[0],
            feature_names=self.feature_names
        )
        
        if save_path:
            shap.save_html(save_path, force_plot)
        
        return force_plot
    
    def plot_dependence(self,
                         X: pd.DataFrame,
                         feature: str,
                         interaction_feature: Optional[str] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot SHAP dependence for a specific feature.
        
        Args:
            X: Feature DataFrame
            feature: Feature name to analyze
            interaction_feature: Optional feature for interaction coloring
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        X_clean = X.fillna(X.median())
        
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        
        fig = plt.figure(figsize=(10, 6))
        
        shap.dependence_plot(
            feature,
            shap_vals,
            X_clean,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        
        plt.title(f'SHAP Dependence: {feature}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def get_model_based_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance directly from model (fallback if SHAP unavailable).
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importances
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
        if len(importance) != len(feature_names):
            # Multi-class: average across classes
            importance = np.abs(model.coef_).mean(axis=0)
    else:
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    importance_df['importance_normalized'] = (
        importance_df['importance'] / importance_df['importance'].sum()
    )
    
    return importance_df


def format_explanation_text(explanation: Dict, 
                            currency_format: bool = True) -> str:
    """
    Format explanation dictionary as readable text.
    
    Args:
        explanation: Explanation dictionary from explain_prediction
        currency_format: Whether to format as currency
        
    Returns:
        Formatted explanation string
    """
    lines = []
    
    if currency_format:
        lines.append(f"Predicted Price: ${explanation['prediction']:,.0f}")
        lines.append(f"Base Value: ${explanation['base_value']:,.0f}")
    else:
        lines.append(f"Prediction: {explanation['prediction']:.2f}")
        lines.append(f"Base Value: {explanation['base_value']:.2f}")
    
    lines.append("\n🔼 Top Positive Factors:")
    for factor in explanation['top_positive_factors'][:3]:
        contrib = factor['contribution']
        if currency_format:
            lines.append(f"  • {factor['feature']}: +${contrib:,.0f}")
        else:
            lines.append(f"  • {factor['feature']}: +{contrib:.2f}")
    
    lines.append("\n🔽 Top Negative Factors:")
    for factor in explanation['top_negative_factors'][:3]:
        contrib = factor['contribution']
        if currency_format:
            lines.append(f"  • {factor['feature']}: -${abs(contrib):,.0f}")
        else:
            lines.append(f"  • {factor['feature']}: {contrib:.2f}")
    
    return '\n'.join(lines)
