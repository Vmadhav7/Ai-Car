"""
Exploratory Data Analysis module for the Car Intelligence System.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def generate_basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate basic statistics for all columns.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        DataFrame with statistics for each column
    """
    # Numerical columns
    numerical_stats = df.describe().T
    numerical_stats['missing'] = df.isnull().sum()
    numerical_stats['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2)
    
    return numerical_stats


def get_categorical_summary(df: pd.DataFrame, 
                            columns: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Get summary of categorical columns.
    
    Args:
        df: DataFrame to analyze
        columns: Categorical columns to summarize (auto-detected if None)
        
    Returns:
        Dictionary with value counts for each categorical column
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    summaries = {}
    for col in columns:
        value_counts = df[col].value_counts()
        pct_counts = (df[col].value_counts(normalize=True) * 100).round(2)
        
        summaries[col] = pd.DataFrame({
            'count': value_counts,
            'percentage': pct_counts
        })
    
    return summaries


def analyze_price_distribution(df: pd.DataFrame, 
                                price_col: str = 'Price_USD') -> Dict:
    """
    Analyze price distribution.
    
    Args:
        df: DataFrame with price column
        price_col: Name of the price column
        
    Returns:
        Dictionary with price distribution statistics
    """
    prices = df[price_col].dropna()
    
    return {
        'mean': prices.mean(),
        'median': prices.median(),
        'std': prices.std(),
        'min': prices.min(),
        'max': prices.max(),
        'q25': prices.quantile(0.25),
        'q75': prices.quantile(0.75),
        'iqr': prices.quantile(0.75) - prices.quantile(0.25),
        'skewness': prices.skew(),
        'kurtosis': prices.kurtosis(),
        'count': len(prices),
        'price_segments': {
            'budget (< $30k)': (prices < 30000).sum(),
            'mid-range ($30k-$100k)': ((prices >= 30000) & (prices < 100000)).sum(),
            'premium ($100k-$500k)': ((prices >= 100000) & (prices < 500000)).sum(),
            'luxury ($500k+)': (prices >= 500000).sum()
        }
    }


def analyze_by_brand(df: pd.DataFrame,
                     company_col: str = 'Company_Clean',
                     price_col: str = 'Price_USD',
                     hp_col: str = 'HorsePower_Numeric') -> pd.DataFrame:
    """
    Analyze cars grouped by brand.
    
    Args:
        df: DataFrame with car data
        company_col: Company column name
        price_col: Price column name
        hp_col: HorsePower column name
        
    Returns:
        DataFrame with brand-level statistics
    """
    brand_stats = df.groupby(company_col).agg({
        price_col: ['count', 'mean', 'median', 'min', 'max'],
        hp_col: ['mean', 'max']
    }).round(2)
    
    # Flatten column names
    brand_stats.columns = [
        'car_count', 'avg_price', 'median_price', 'min_price', 'max_price',
        'avg_hp', 'max_hp'
    ]
    
    return brand_stats.sort_values('car_count', ascending=False)


def analyze_by_fuel_type(df: pd.DataFrame,
                          fuel_col: str = 'Fuel_Type_Std',
                          price_col: str = 'Price_USD',
                          hp_col: str = 'HorsePower_Numeric',
                          acc_col: str = 'Acceleration_Sec') -> pd.DataFrame:
    """
    Analyze cars grouped by fuel type.
    
    Args:
        df: DataFrame with car data
        fuel_col: Fuel type column name
        price_col: Price column name
        hp_col: HorsePower column name
        acc_col: Acceleration column name
        
    Returns:
        DataFrame with fuel type statistics
    """
    available_cols = [col for col in [price_col, hp_col, acc_col] if col in df.columns]
    
    fuel_stats = df.groupby(fuel_col).agg({
        **{col: ['count', 'mean'] for col in available_cols}
    }).round(2)
    
    return fuel_stats


def calculate_correlation_matrix(df: pd.DataFrame,
                                  columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate correlation matrix for numerical columns.
    
    Args:
        df: DataFrame to analyze
        columns: Specific columns to correlate (uses all numerical if None)
        
    Returns:
        Correlation matrix DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return df[columns].corr().round(3)


def identify_outliers(df: pd.DataFrame,
                       column: str,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> Tuple[pd.DataFrame, Dict]:
    """
    Identify outliers in a column.
    
    Args:
        df: DataFrame to analyze
        column: Column to check for outliers
        method: 'iqr' for IQR method, 'zscore' for Z-score method
        threshold: Threshold for outlier detection
        
    Returns:
        Tuple of (DataFrame with outlier rows, outlier statistics dict)
    """
    data = df[column].dropna()
    
    if method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    else:  # zscore
        mean = data.mean()
        std = data.std()
        z_scores = (df[column] - mean) / std
        outlier_mask = z_scores.abs() > threshold
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
    
    outliers = df[outlier_mask]
    
    stats = {
        'total_outliers': len(outliers),
        'outlier_percentage': (len(outliers) / len(df) * 100),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
    
    return outliers, stats


def analyze_performance_vs_price(df: pd.DataFrame,
                                  acc_col: str = 'Acceleration_Sec',
                                  price_col: str = 'Price_USD') -> Dict:
    """
    Analyze relationship between performance and price.
    
    Args:
        df: DataFrame with car data
        acc_col: Acceleration column name
        price_col: Price column name
        
    Returns:
        Dictionary with performance-price relationship statistics
    """
    # Remove rows with missing values
    valid_data = df[[acc_col, price_col]].dropna()
    
    # Correlation
    correlation = valid_data[acc_col].corr(valid_data[price_col])
    
    # Group by performance category
    def categorize_performance(acc):
        if acc < 4.5:
            return 'High-Performance'
        elif acc <= 8.0:
            return 'Mid-Range'
        else:
            return 'Economy'
    
    valid_data = valid_data.copy()
    valid_data['category'] = valid_data[acc_col].apply(categorize_performance)
    
    category_prices = valid_data.groupby('category')[price_col].agg(['mean', 'median', 'count'])
    
    return {
        'correlation': correlation,
        'interpretation': 'Negative correlation means faster cars (lower time) cost more',
        'category_prices': category_prices.to_dict()
    }


def generate_eda_report(df: pd.DataFrame) -> Dict:
    """
    Generate a comprehensive EDA report.
    
    Args:
        df: Cleaned DataFrame with car data
        
    Returns:
        Dictionary containing all EDA insights
    """
    report = {
        'dataset_shape': {
            'rows': len(df),
            'columns': len(df.columns)
        },
        'basic_statistics': generate_basic_statistics(df).to_dict(),
        'price_analysis': analyze_price_distribution(df),
        'correlation_matrix': calculate_correlation_matrix(df).to_dict()
    }
    
    # Add brand analysis if column exists
    if 'Company_Clean' in df.columns:
        report['brand_analysis'] = analyze_by_brand(df).to_dict()
    
    # Add fuel type analysis if column exists
    if 'Fuel_Type_Std' in df.columns:
        report['fuel_type_analysis'] = analyze_by_fuel_type(df).to_dict()
    
    # Add performance-price relationship
    if 'Acceleration_Sec' in df.columns and 'Price_USD' in df.columns:
        report['performance_price'] = analyze_performance_vs_price(df)
    
    return report
