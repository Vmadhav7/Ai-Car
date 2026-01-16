"""
Unit tests for data processing modules.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.helpers import (
    parse_price, extract_numeric, parse_acceleration,
    standardize_fuel_type, classify_performance, is_luxury_brand
)
from data.data_cleaner import clean_prices, clean_horsepower


class TestParsePrice:
    """Tests for price parsing function."""
    
    def test_simple_price(self):
        assert parse_price("$100,000") == 100000
    
    def test_price_with_spaces(self):
        assert parse_price("$100,000 ") == 100000
    
    def test_price_range_returns_midpoint(self):
        result = parse_price("$15,000 - $18,000")
        assert result == 16500  # midpoint
    
    def test_invalid_price_returns_nan(self):
        assert np.isnan(parse_price(""))
    
    def test_euro_symbol(self):
        # Should still extract numeric
        result = parse_price("€33,000")
        assert result == 33000 or not np.isnan(result)


class TestExtractNumeric:
    """Tests for numeric extraction function."""
    
    def test_with_hp_suffix(self):
        assert extract_numeric("963 hp") == 963
    
    def test_with_kmh_suffix(self):
        assert extract_numeric("340 km/h") == 340
    
    def test_with_nm_suffix(self):
        assert extract_numeric("800 Nm") == 800
    
    def test_with_commas(self):
        assert extract_numeric("3,990 cc") == 3990
    
    def test_decimal_values(self):
        assert extract_numeric("2.5 sec") == 2.5
    
    def test_integer_input(self):
        assert extract_numeric(100) == 100


class TestParseAcceleration:
    """Tests for acceleration parsing function."""
    
    def test_simple_acceleration(self):
        assert parse_acceleration("2.5 sec") == 2.5
    
    def test_acceleration_range(self):
        result = parse_acceleration("8.0 - 9.2 sec")
        assert abs(result - 8.6) < 0.01  # midpoint
    
    def test_uppercase(self):
        assert parse_acceleration("3.5 SEC") == 3.5


class TestStandardizeFuelType:
    """Tests for fuel type standardization."""
    
    def test_petrol(self):
        assert standardize_fuel_type("Petrol") == "Petrol"
    
    def test_diesel(self):
        assert standardize_fuel_type("Diesel") == "Diesel"
    
    def test_electric(self):
        assert standardize_fuel_type("Electric") == "Electric"
    
    def test_hybrid(self):
        assert standardize_fuel_type("Hybrid") == "Hybrid"
    
    def test_plug_in_hybrid(self):
        assert standardize_fuel_type("plug in hyrbrid") == "Plug-in Hybrid"
        assert standardize_fuel_type("Plug-in Hybrid") == "Plug-in Hybrid"
    
    def test_typo_handling(self):
        # Should handle common typos
        assert standardize_fuel_type("hyrbrid") == "Hybrid"


class TestClassifyPerformance:
    """Tests for performance classification."""
    
    def test_high_performance(self):
        assert classify_performance(2.5) == "High-Performance"
        assert classify_performance(4.4) == "High-Performance"
    
    def test_mid_range(self):
        assert classify_performance(4.5) == "Mid-Range"
        assert classify_performance(6.0) == "Mid-Range"
        assert classify_performance(8.0) == "Mid-Range"
    
    def test_economy(self):
        assert classify_performance(8.1) == "Economy"
        assert classify_performance(12.0) == "Economy"
    
    def test_nan_input(self):
        assert classify_performance(np.nan) == "Unknown"


class TestIsLuxuryBrand:
    """Tests for luxury brand detection."""
    
    def test_ferrari(self):
        assert is_luxury_brand("Ferrari") == True
        assert is_luxury_brand("FERRARI") == True
    
    def test_toyota(self):
        assert is_luxury_brand("Toyota") == False
    
    def test_mercedes(self):
        assert is_luxury_brand("Mercedes") == True
    
    def test_none_input(self):
        assert is_luxury_brand(None) == False


class TestDataCleanerIntegration:
    """Integration tests for data cleaner."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'Cars Prices': ['$100,000', '$50,000 - $60,000', '$25,000'],
            'HorsePower': ['300 hp', '200-250 hp', '150 hp']
        })
    
    def test_clean_prices(self, sample_df):
        result = clean_prices(sample_df)
        assert len(result) == 3
        assert result.iloc[0] == 100000
        assert result.iloc[1] == 55000  # midpoint
        assert result.iloc[2] == 25000
    
    def test_clean_horsepower(self, sample_df):
        result = clean_horsepower(sample_df)
        assert len(result) == 3
        assert result.iloc[0] == 300
        assert result.iloc[1] == 225  # average of range


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
