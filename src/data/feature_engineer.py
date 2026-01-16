"""
Feature engineering module for the Car Intelligence System.
Creates derived features and prepares data for modeling.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import classify_performance, is_luxury_brand


def create_performance_category(df: pd.DataFrame, 
                                 acceleration_col: str = 'Acceleration_Sec') -> pd.Series:
    """
    Create performance category based on acceleration time.
    
    Categories:
    - High-Performance: < 4.5 seconds
    - Mid-Range: 4.5 - 8.0 seconds
    - Economy: > 8.0 seconds
    
    Args:
        df: DataFrame with acceleration column
        acceleration_col: Name of the acceleration column
        
    Returns:
        Series with performance category labels
    """
    return df[acceleration_col].apply(classify_performance)


def create_luxury_brand_feature(df: pd.DataFrame, 
                                 company_col: str = 'Company_Clean') -> pd.Series:
    """
    Create binary feature indicating if brand is luxury/premium.
    
    Args:
        df: DataFrame with company column
        company_col: Name of the company column
        
    Returns:
        Series with boolean values (True = luxury brand)
    """
    return df[company_col].apply(is_luxury_brand)


def create_price_per_hp(df: pd.DataFrame,
                        price_col: str = 'Price_USD',
                        hp_col: str = 'HorsePower_Numeric') -> pd.Series:
    """
    Calculate price per horsepower ratio.
    
    Args:
        df: DataFrame with price and horsepower columns
        price_col: Name of the price column
        hp_col: Name of the horsepower column
        
    Returns:
        Series with price per HP values
    """
    return df[price_col] / df[hp_col].replace(0, np.nan)


def create_power_to_speed_ratio(df: pd.DataFrame,
                                 hp_col: str = 'HorsePower_Numeric',
                                 speed_col: str = 'Speed_KMH') -> pd.Series:
    """
    Calculate horsepower to max speed ratio (efficiency metric).
    
    Args:
        df: DataFrame with horsepower and speed columns
        hp_col: Name of the horsepower column
        speed_col: Name of the speed column
        
    Returns:
        Series with HP per km/h values
    """
    return df[hp_col] / df[speed_col].replace(0, np.nan)


def create_is_electric(df: pd.DataFrame,
                       fuel_col: str = 'Fuel_Type_Std') -> pd.Series:
    """
    Create binary feature for electric vehicles.
    
    Args:
        df: DataFrame with fuel type column
        fuel_col: Name of the standardized fuel type column
        
    Returns:
        Series with boolean values (True = electric/hybrid)
    """
    electric_types = {'Electric', 'Hybrid', 'Plug-in Hybrid'}
    return df[fuel_col].isin(electric_types)


def encode_categorical_features(df: pd.DataFrame,
                                 columns: List[str],
                                 method: str = 'label') -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical features using specified method.
    
    Args:
        df: DataFrame with categorical columns
        columns: List of column names to encode
        method: 'label' for label encoding, 'onehot' for one-hot encoding
        
    Returns:
        Tuple of (encoded DataFrame, dict of encoders/mapping)
    """
    result = df.copy()
    encoders = {}
    
    if method == 'label':
        for col in columns:
            le = LabelEncoder()
            # Handle missing values
            mask = result[col].notna()
            result.loc[mask, f'{col}_Encoded'] = le.fit_transform(result.loc[mask, col].astype(str))
            encoders[col] = le
            
    elif method == 'onehot':
        for col in columns:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
            result = pd.concat([result, dummies], axis=1)
            encoders[col] = list(dummies.columns)
    
    return result, encoders


def scale_numerical_features(df: pd.DataFrame,
                              columns: List[str],
                              method: str = 'standard') -> Tuple[pd.DataFrame, object]:
    """
    Scale numerical features using specified method.
    
    Args:
        df: DataFrame with numerical columns
        columns: List of column names to scale
        method: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
        
    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    result = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    # Only scale non-null values
    data_to_scale = result[columns].copy()
    valid_mask = data_to_scale.notna().all(axis=1)
    
    if valid_mask.sum() > 0:
        scaled_values = scaler.fit_transform(data_to_scale[valid_mask])
        
        for i, col in enumerate(columns):
            result.loc[valid_mask, f'{col}_Scaled'] = scaled_values[:, i]
    
    return result, scaler


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all derived features for the dataset.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with all derived features added
    """
    result = df.copy()
    
    # Create derived features
    result['Performance_Category'] = create_performance_category(result)
    result['Is_Luxury_Brand'] = create_luxury_brand_feature(result)
    result['Price_Per_HP'] = create_price_per_hp(result)
    result['Power_To_Speed_Ratio'] = create_power_to_speed_ratio(result)
    result['Is_Electric'] = create_is_electric(result)
    
    return result


def prepare_features_for_modeling(df: pd.DataFrame,
                                   target_col: str,
                                   feature_cols: Optional[List[str]] = None,
                                   categorical_cols: Optional[List[str]] = None,
                                   numerical_cols: Optional[List[str]] = None,
                                   scale: bool = True) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Prepare features for ML modeling by encoding and scaling.
    
    Args:
        df: DataFrame with all features
        target_col: Name of the target column
        feature_cols: List of feature columns to use (optional, uses defaults if None)
        categorical_cols: Categorical columns for encoding
        numerical_cols: Numerical columns for scaling
        scale: Whether to scale numerical features
        
    Returns:
        Tuple of (features DataFrame, target Series, preprocessing info dict)
    """
    result = df.copy()
    preprocessing_info = {}
    
    # Default feature columns for regression
    if feature_cols is None:
        feature_cols = [
            'HorsePower_Numeric', 'Speed_KMH', 'Acceleration_Sec',
            'Torque_Nm', 'CC_Numeric', 'Seats_Numeric',
            'Is_Luxury_Brand', 'Is_Electric'
        ]
    
    if categorical_cols is None:
        categorical_cols = ['Fuel_Type_Std', 'Company_Clean']
    
    if numerical_cols is None:
        numerical_cols = [
            'HorsePower_Numeric', 'Speed_KMH', 'Acceleration_Sec',
            'Torque_Nm', 'CC_Numeric', 'Seats_Numeric'
        ]
    
    # Encode categorical features
    result, encoders = encode_categorical_features(result, categorical_cols, method='label')
    preprocessing_info['encoders'] = encoders
    
    # Scale numerical features if requested
    if scale:
        result, scaler = scale_numerical_features(result, numerical_cols, method='standard')
        preprocessing_info['scaler'] = scaler
        preprocessing_info['scaled_cols'] = [f'{col}_Scaled' for col in numerical_cols]
    
    # Extract target
    target = result[target_col]
    
    return result, target, preprocessing_info


def get_feature_matrix_for_recommendation(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Create a normalized feature matrix for similarity-based recommendations.
    
    Args:
        df: DataFrame with features
        
    Returns:
        Tuple of (feature matrix, list of feature names used)
    """
    # Features to use for similarity computation
    feature_cols = [
        'HorsePower_Numeric', 'Speed_KMH', 'Acceleration_Sec',
        'Torque_Nm', 'CC_Numeric', 'Seats_Numeric', 'Price_USD'
    ]
    
    # Extract and normalize features
    feature_df = df[feature_cols].copy()
    
    # Impute missing values with median
    for col in feature_cols:
        median_val = feature_df[col].median()
        feature_df[col] = feature_df[col].fillna(median_val)
    
    # Min-max scale for cosine similarity
    scaler = MinMaxScaler()
    feature_matrix = scaler.fit_transform(feature_df)
    
    return feature_matrix, feature_cols
