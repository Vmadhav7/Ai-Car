"""
Data cleaning module for the Car Intelligence System.
Handles parsing, standardization, and cleaning of raw data.
"""

import pandas as pd
import numpy as np
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import (
    parse_price,
    extract_numeric,
    parse_acceleration,
    standardize_fuel_type
)


def clean_prices(df: pd.DataFrame, column: str = 'Cars Prices') -> pd.Series:
    """
    Clean price column by parsing currency strings to numeric values.
    
    Args:
        df: DataFrame containing the price column
        column: Name of the price column
        
    Returns:
        Series with cleaned numeric price values
    """
    return df[column].apply(parse_price)


def clean_horsepower(df: pd.DataFrame, column: str = 'HorsePower') -> pd.Series:
    """
    Clean horsepower column by extracting numeric values.
    Handles formats like '963 hp', '70-85 hp'
    
    Args:
        df: DataFrame containing the horsepower column
        column: Name of the horsepower column
        
    Returns:
        Series with cleaned numeric horsepower values
    """
    def parse_hp(val):
        if not isinstance(val, str):
            return np.nan
        
        val = val.lower().replace('hp', '').strip()
        
        # Check for range
        if '-' in val:
            parts = val.split('-')
            try:
                values = [float(p.strip().replace(',', '')) for p in parts if p.strip()]
                return np.mean(values)
            except ValueError:
                pass
        
        return extract_numeric(val)
    
    return df[column].apply(parse_hp)


def clean_speed(df: pd.DataFrame, column: str = 'Total Speed') -> pd.Series:
    """
    Clean speed column by extracting numeric values.
    
    Args:
        df: DataFrame containing the speed column
        column: Name of the speed column
        
    Returns:
        Series with cleaned numeric speed values in km/h
    """
    return df[column].apply(extract_numeric)


def clean_acceleration(df: pd.DataFrame, column: str = 'Performance(0 - 100 )KM/H') -> pd.Series:
    """
    Clean acceleration time column.
    
    Args:
        df: DataFrame containing the acceleration column
        column: Name of the acceleration column
        
    Returns:
        Series with cleaned acceleration times in seconds
    """
    return df[column].apply(parse_acceleration)


def clean_torque(df: pd.DataFrame, column: str = 'Torque') -> pd.Series:
    """
    Clean torque column by extracting numeric values.
    Handles formats like '800 Nm', '100 - 140 Nm'
    
    Args:
        df: DataFrame containing the torque column
        column: Name of the torque column
        
    Returns:
        Series with cleaned numeric torque values in Nm
    """
    def parse_torque(val):
        if not isinstance(val, str):
            return np.nan
        
        val = val.lower().replace('nm', '').strip()
        
        # Check for range
        if ' - ' in val or '-' in val:
            parts = val.replace(' - ', '-').split('-')
            try:
                values = [float(p.strip().replace(',', '')) for p in parts if p.strip()]
                if len(values) >= 2:
                    return np.mean(values)
            except ValueError:
                pass
        
        return extract_numeric(val)
    
    return df[column].apply(parse_torque)


def clean_cc(df: pd.DataFrame, column: str = 'CC/Battery Capacity') -> pd.Series:
    """
    Clean engine displacement / battery capacity column.
    
    Args:
        df: DataFrame containing the CC column
        column: Name of the CC column
        
    Returns:
        Series with cleaned numeric CC values
    """
    return df[column].apply(extract_numeric)


def clean_seats(df: pd.DataFrame, column: str = 'Seats') -> pd.Series:
    """
    Clean seats column, handling non-numeric values.
    
    Args:
        df: DataFrame containing the seats column
        column: Name of the seats column
        
    Returns:
        Series with cleaned seat count values
    """
    def parse_seats(val):
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return np.nan
    
    return df[column].apply(parse_seats)


def standardize_fuel_types(df: pd.DataFrame, column: str = 'Fuel Types') -> pd.Series:
    """
    Standardize fuel types to consistent categories.
    
    Args:
        df: DataFrame containing the fuel type column
        column: Name of the fuel type column
        
    Returns:
        Series with standardized fuel type values
    """
    return df[column].apply(standardize_fuel_type)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning transformations to the dataset.
    
    Args:
        df: Raw DataFrame to clean
        
    Returns:
        Cleaned DataFrame with standardized data types
    """
    # Create a copy to avoid modifying the original
    cleaned = df.copy()
    
    # Clean numerical columns
    cleaned['Price_USD'] = clean_prices(cleaned)
    cleaned['HorsePower_Numeric'] = clean_horsepower(cleaned)
    cleaned['Speed_KMH'] = clean_speed(cleaned)
    cleaned['Acceleration_Sec'] = clean_acceleration(cleaned)
    cleaned['Torque_Nm'] = clean_torque(cleaned)
    cleaned['CC_Numeric'] = clean_cc(cleaned)
    cleaned['Seats_Numeric'] = clean_seats(cleaned)
    
    # Standardize categorical columns
    cleaned['Fuel_Type_Std'] = standardize_fuel_types(cleaned)
    
    # Clean company name (trim whitespace, uppercase)
    cleaned['Company_Clean'] = cleaned['Company Names'].str.strip().str.upper()
    
    # Clean car name
    cleaned['Car_Name_Clean'] = cleaned['Cars Names'].str.strip()
    
    return cleaned


def remove_invalid_rows(df: pd.DataFrame, 
                         price_col: str = 'Price_USD',
                         min_price: float = 1000,
                         max_price: float = 50_000_000) -> pd.DataFrame:
    """
    Remove rows with invalid or missing critical values.
    
    Args:
        df: Cleaned DataFrame
        price_col: Name of the price column
        min_price: Minimum valid price
        max_price: Maximum valid price
        
    Returns:
        DataFrame with invalid rows removed
    """
    # Remove rows with missing prices
    valid_df = df.dropna(subset=[price_col])
    
    # Remove rows with prices outside valid range
    valid_df = valid_df[
        (valid_df[price_col] >= min_price) & 
        (valid_df[price_col] <= max_price)
    ]
    
    return valid_df.reset_index(drop=True)
