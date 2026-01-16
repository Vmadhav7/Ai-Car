"""
Unit tests for ML models.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.price_predictor import PricePredictor
from models.classifier import PerformanceClassifier
from models.recommender import CarRecommender


class TestPricePredictor:
    """Tests for PricePredictor class."""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'HorsePower': np.random.uniform(100, 800, n),
            'Speed': np.random.uniform(150, 350, n),
            'Acceleration': np.random.uniform(2, 15, n),
            'Torque': np.random.uniform(100, 1000, n),
        }), pd.Series(np.random.uniform(20000, 500000, n))
    
    def test_initialization(self):
        predictor = PricePredictor(use_xgboost=False)
        assert not predictor.is_fitted
        assert len(predictor.models) >= 3
    
    def test_training(self, sample_data):
        X, y = sample_data
        predictor = PricePredictor(use_xgboost=False)
        results = predictor.train(X, y)
        
        assert predictor.is_fitted
        assert predictor.best_model_name is not None
        assert 'random_forest' in results
    
    def test_prediction(self, sample_data):
        X, y = sample_data
        predictor = PricePredictor(use_xgboost=False)
        predictor.train(X, y, validate=False)
        
        predictions = predictor.predict(X.head(5))
        assert len(predictions) == 5
        assert all(p > 0 for p in predictions)
    
    def test_feature_importance(self, sample_data):
        X, y = sample_data
        predictor = PricePredictor(use_xgboost=False)
        predictor.train(X, y, validate=False)
        
        importance = predictor.get_feature_importance()
        assert len(importance) == 4
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns


class TestPerformanceClassifier:
    """Tests for PerformanceClassifier class."""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'HorsePower': np.random.uniform(100, 800, n),
            'Speed': np.random.uniform(150, 350, n),
            'Torque': np.random.uniform(100, 1000, n),
        })
        y = pd.Series(np.random.choice(
            ['High-Performance', 'Mid-Range', 'Economy'], n
        ))
        return X, y
    
    def test_initialization(self):
        classifier = PerformanceClassifier(use_xgboost=False)
        assert not classifier.is_fitted
        assert len(classifier.models) >= 3
    
    def test_training(self, sample_data):
        X, y = sample_data
        classifier = PerformanceClassifier(use_xgboost=False)
        results = classifier.train(X, y)
        
        assert classifier.is_fitted
        assert classifier.best_model_name is not None
    
    def test_prediction(self, sample_data):
        X, y = sample_data
        classifier = PerformanceClassifier(use_xgboost=False)
        classifier.train(X, y, validate=False)
        
        predictions = classifier.predict(X.head(5))
        assert len(predictions) == 5
        assert all(p in PerformanceClassifier.CATEGORIES for p in predictions)
    
    def test_predict_proba(self, sample_data):
        X, y = sample_data
        classifier = PerformanceClassifier(use_xgboost=False)
        classifier.train(X, y, validate=False)
        
        probas = classifier.predict_proba(X.head(5))
        assert probas.shape[0] == 5
        assert all(probas.sum(axis=1).round(3) == 1.0)


class TestCarRecommender:
    """Tests for CarRecommender class."""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            'Company_Clean': ['BRAND_A'] * 25 + ['BRAND_B'] * 25,
            'Car_Name_Clean': [f'Car_{i}' for i in range(n)],
            'HorsePower_Numeric': np.random.uniform(100, 800, n),
            'Speed_KMH': np.random.uniform(150, 350, n),
            'Acceleration_Sec': np.random.uniform(2, 15, n),
            'Torque_Nm': np.random.uniform(100, 1000, n),
            'CC_Numeric': np.random.uniform(1000, 6000, n),
            'Seats_Numeric': np.random.choice([2, 4, 5, 7], n),
            'Price_USD': np.random.uniform(20000, 500000, n),
        })
    
    def test_initialization(self):
        recommender = CarRecommender()
        assert not recommender.is_fitted
    
    def test_fit(self, sample_data):
        recommender = CarRecommender()
        recommender.fit(sample_data)
        
        assert recommender.is_fitted
        assert recommender.feature_matrix is not None
    
    def test_get_similar_cars(self, sample_data):
        recommender = CarRecommender()
        recommender.fit(sample_data)
        
        recommendations = recommender.get_similar_cars(0, n_recommendations=5)
        assert len(recommendations) == 5
        assert 'similarity_score' in recommendations.columns
        assert all(recommendations['similarity_score'] <= 1.0)
    
    def test_exclude_same_brand(self, sample_data):
        recommender = CarRecommender()
        recommender.fit(sample_data)
        
        # Car 0 is BRAND_A
        recommendations = recommender.get_similar_cars(0, n_recommendations=5, exclude_same_brand=True)
        assert all(recommendations['Company_Clean'] != 'BRAND_A')
    
    def test_coverage(self, sample_data):
        recommender = CarRecommender()
        recommender.fit(sample_data)
        
        coverage = recommender.get_coverage()
        assert 0 <= coverage <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
