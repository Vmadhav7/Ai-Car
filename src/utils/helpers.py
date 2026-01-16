"""
Utility helper functions for data processing and parsing.
"""

import re
import numpy as np


def parse_price(price_str: str) -> float:
    """
    Parse price string and return numeric value.
    Handles formats like '$1,100,000', '$15,000 - $18,000', '? 33,000'
    
    For ranges, returns the midpoint value.
    
    Args:
        price_str: Raw price string from dataset
        
    Returns:
        Numeric price value in USD
    """
    if not isinstance(price_str, str):
        return np.nan
    
    # Clean the string
    price_str = price_str.strip()
    
    # Remove currency symbols and common characters
    price_str = price_str.replace('$', '').replace('?', '').replace('€', '')
    
    # Check if it's a range (contains '-' or 'to')
    if ' - ' in price_str or ' to ' in price_str:
        # Split by range separator
        parts = re.split(r'\s*[-–]\s*|\s+to\s+', price_str)
        if len(parts) >= 2:
            low = extract_numeric(parts[0])
            high = extract_numeric(parts[1])
            if not np.isnan(low) and not np.isnan(high):
                return (low + high) / 2
            elif not np.isnan(low):
                return low
            elif not np.isnan(high):
                return high
    
    # Single value
    return extract_numeric(price_str)


def extract_numeric(value_str: str) -> float:
    """
    Extract numeric value from a string with units.
    Handles formats like '963 hp', '340 km/h', '800 Nm', '3,990 cc'
    
    Args:
        value_str: String containing a numeric value with possible units
        
    Returns:
        Extracted numeric value as float
    """
    if not isinstance(value_str, str):
        try:
            return float(value_str)
        except (ValueError, TypeError):
            return np.nan
    
    # Clean the string
    value_str = value_str.strip()
    
    # Remove common units and text
    value_str = re.sub(r'\s*(hp|HP|km/h|KM/H|Nm|nm|cc|CC|sec|SEC|kWh|kwh)\s*', '', value_str, flags=re.IGNORECASE)
    
    # Remove commas used as thousand separators
    value_str = value_str.replace(',', '')
    
    # Try to find the first numeric value (including decimals)
    match = re.search(r'[-+]?\d*\.?\d+', value_str)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return np.nan
    
    return np.nan


def parse_acceleration(acc_str: str) -> float:
    """
    Parse acceleration time string (0-100 km/h).
    Handles formats like '2.5 sec', '10.5 sec', '8.0 - 9.2 sec'
    
    For ranges, returns the average value.
    
    Args:
        acc_str: Raw acceleration string from dataset
        
    Returns:
        Acceleration time in seconds
    """
    if not isinstance(acc_str, str):
        return np.nan
    
    # Clean the string
    acc_str = acc_str.strip().lower()
    
    # Remove 'sec' and other text
    acc_str = re.sub(r'\s*sec(onds?)?\s*', '', acc_str)
    
    # Check if it's a range
    if ' - ' in acc_str or ' to ' in acc_str:
        parts = re.split(r'\s*[-–]\s*|\s+to\s+', acc_str)
        if len(parts) >= 2:
            low = extract_numeric(parts[0])
            high = extract_numeric(parts[1])
            if not np.isnan(low) and not np.isnan(high):
                return (low + high) / 2
    
    return extract_numeric(acc_str)


def standardize_fuel_type(fuel_str: str) -> str:
    """
    Standardize fuel type to one of: Petrol, Diesel, Hybrid, Electric, Plug-in Hybrid, Other
    
    Args:
        fuel_str: Raw fuel type string from dataset
        
    Returns:
        Standardized fuel type category
    """
    if not isinstance(fuel_str, str):
        return 'Other'
    
    fuel_lower = fuel_str.lower().strip()
    
    # Electric
    if 'electric' in fuel_lower and 'hybrid' not in fuel_lower:
        return 'Electric'
    
    # Plug-in Hybrid (check before regular hybrid)
    if 'plug' in fuel_lower or 'plug-in' in fuel_lower:
        return 'Plug-in Hybrid'
    
    # Hybrid (but not plug-in)
    if 'hybrid' in fuel_lower or 'hyrbrid' in fuel_lower:  # Handle typo
        return 'Hybrid'
    
    # Diesel
    if 'diesel' in fuel_lower and 'petrol' not in fuel_lower:
        return 'Diesel'
    
    # Petrol (and variations)
    if 'petrol' in fuel_lower or 'gas' in fuel_lower or 'gasoline' in fuel_lower:
        return 'Petrol'
    
    # CNG
    if 'cng' in fuel_lower:
        return 'CNG'
    
    # Hydrogen
    if 'hydrogen' in fuel_lower:
        return 'Hydrogen'
    
    return 'Other'


def classify_performance(acceleration_time: float) -> str:
    """
    Classify car into performance category based on 0-100 km/h time.
    
    Categories:
    - High-Performance: < 4.5 seconds
    - Mid-Range: 4.5 - 8.0 seconds
    - Economy: > 8.0 seconds
    
    Args:
        acceleration_time: 0-100 km/h time in seconds
        
    Returns:
        Performance category string
    """
    if np.isnan(acceleration_time):
        return 'Unknown'
    
    if acceleration_time < 4.5:
        return 'High-Performance'
    elif acceleration_time <= 8.0:
        return 'Mid-Range'
    else:
        return 'Economy'


def is_luxury_brand(brand: str) -> bool:
    """
    Check if a brand is considered luxury/premium.
    
    Args:
        brand: Car manufacturer name
        
    Returns:
        True if luxury brand, False otherwise
    """
    luxury_brands = {
        'ferrari', 'rolls royce', 'rolls-royce', 'lamborghini', 'bentley',
        'aston martin', 'maserati', 'bugatti', 'mclaren', 'porsche',
        'mercedes', 'mercedes-benz', 'bmw', 'audi', 'lexus', 'jaguar',
        'land rover', 'range rover', 'cadillac', 'lincoln', 'genesis',
        'infinity', 'infiniti', 'acura', 'alfa romeo', 'maybach', 'pagani',
        'koenigsegg', 'rimac', 'lotus'
    }
    
    if not isinstance(brand, str):
        return False
    
    return brand.lower().strip() in luxury_brands
