"""
Data loading module for the Car Intelligence System.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_cars_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the cars dataset from a CSV file.
    
    Args:
        filepath: Path to the CSV file. If None, uses the default path.
        
    Returns:
        DataFrame containing the raw car data
    """
    if filepath is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent.parent
        filepath = project_root / 'data' / 'Cars Datasets 2025.csv'
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    
    # Load the CSV file with encoding fallback
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')
    
    # Validate expected columns
    expected_columns = [
        'Company Names', 'Cars Names', 'Engines', 'CC/Battery Capacity',
        'HorsePower', 'Total Speed', 'Performance(0 - 100 )KM/H',
        'Cars Prices', 'Fuel Types', 'Seats', 'Torque'
    ]
    
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")
    
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate the loaded dataset and return a summary of data quality.
    
    Args:
        df: Raw DataFrame to validate
        
    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'column_dtypes': df.dtypes.astype(str).to_dict(),
        'unique_companies': df['Company Names'].nunique(),
        'unique_fuel_types': df['Fuel Types'].nunique(),
        'issues': []
    }
    
    # Check for excessive missing values
    for col, missing_count in validation_results['missing_values'].items():
        if missing_count > len(df) * 0.1:  # More than 10% missing
            validation_results['issues'].append(
                f"Column '{col}' has {missing_count} missing values ({missing_count/len(df)*100:.1f}%)"
            )
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        validation_results['duplicates'] = duplicate_count
        validation_results['issues'].append(
            f"Found {duplicate_count} duplicate rows"
        )
    
    return validation_results


def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of the dataset for quick inspection.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Summary DataFrame with statistics for each column
    """
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'non_null_count': df.count(),
        'null_count': df.isnull().sum(),
        'null_percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'unique_values': df.nunique(),
        'sample_value': df.iloc[0] if len(df) > 0 else None
    })
    
    return summary
