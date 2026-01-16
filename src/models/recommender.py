"""
Car recommendation module using content-based filtering.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class CarRecommender:
    """
    Content-based car recommendation system using cosine similarity.
    """
    
    def __init__(self, feature_columns: Optional[List[str]] = None):
        """
        Initialize the recommender.
        
        Args:
            feature_columns: List of numerical columns to use for similarity.
                           If None, uses default features.
        """
        self.feature_columns = feature_columns or [
            'HorsePower_Numeric', 'Speed_KMH', 'Acceleration_Sec',
            'Torque_Nm', 'CC_Numeric', 'Seats_Numeric', 'Price_USD'
        ]
        self.scaler = MinMaxScaler()
        self.feature_matrix: Optional[np.ndarray] = None
        self.car_data: Optional[pd.DataFrame] = None
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame):
        """
        Fit the recommender on the car dataset.
        
        Args:
            df: DataFrame with car data and features
        """
        # Store car data for lookup
        self.car_data = df.copy()
        
        # Extract features for similarity computation
        feature_df = df[self.feature_columns].copy()
        
        # Impute missing values with column medians
        for col in self.feature_columns:
            median_val = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(median_val)
        
        # Scale features to [0, 1] range
        self.feature_matrix = self.scaler.fit_transform(feature_df)
        self.is_fitted = True
    
    def get_similar_cars(self, 
                          car_index: int, 
                          n_recommendations: int = 5,
                          exclude_same_brand: bool = False) -> pd.DataFrame:
        """
        Get similar cars for a given car index.
        
        Args:
            car_index: Index of the car to find similar ones for
            n_recommendations: Number of recommendations to return
            exclude_same_brand: Whether to exclude cars from the same brand
            
        Returns:
            DataFrame with recommended cars and similarity scores
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        if car_index < 0 or car_index >= len(self.feature_matrix):
            raise ValueError(f"Invalid car index: {car_index}")
        
        # Get the feature vector for the target car
        car_features = self.feature_matrix[car_index].reshape(1, -1)
        
        # Calculate similarity with all cars
        similarities = cosine_similarity(car_features, self.feature_matrix)[0]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'index': range(len(similarities)),
            'similarity': similarities
        })
        
        # Exclude the target car itself
        results = results[results['index'] != car_index]
        
        # Optionally exclude same brand
        if exclude_same_brand and 'Company_Clean' in self.car_data.columns:
            target_brand = self.car_data.iloc[car_index]['Company_Clean']
            brand_mask = self.car_data['Company_Clean'] != target_brand
            results = results[brand_mask.values[results['index']]]
        
        # Sort by similarity and get top N
        results = results.nlargest(n_recommendations, 'similarity')
        
        # Add car information
        recommendations = self.car_data.iloc[results['index'].values].copy()
        recommendations['similarity_score'] = results['similarity'].values
        
        return recommendations
    
    def get_recommendations_by_name(self,
                                      car_name: str,
                                      n_recommendations: int = 5,
                                      exclude_same_brand: bool = False) -> pd.DataFrame:
        """
        Get similar cars by car name (partial match).
        
        Args:
            car_name: Name or partial name of the target car
            n_recommendations: Number of recommendations to return
            exclude_same_brand: Whether to exclude cars from the same brand
            
        Returns:
            DataFrame with recommended cars and similarity scores
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        # Find matching car
        name_col = 'Car_Name_Clean' if 'Car_Name_Clean' in self.car_data.columns else 'Cars Names'
        mask = self.car_data[name_col].str.lower().str.contains(car_name.lower(), na=False)
        
        if not mask.any():
            raise ValueError(f"No car found matching: {car_name}")
        
        # Use the first match
        car_index = self.car_data[mask].index[0]
        
        return self.get_similar_cars(car_index, n_recommendations, exclude_same_brand)
    
    def get_recommendations_by_specs(self,
                                       specs: dict,
                                       n_recommendations: int = 5) -> pd.DataFrame:
        """
        Get recommendations based on desired specifications.
        
        Args:
            specs: Dictionary with feature names and target values
                  e.g., {'HorsePower_Numeric': 500, 'Price_USD': 100000}
            n_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommended cars and similarity scores
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        # Build target feature vector using median for missing specs
        target_features = []
        for col in self.feature_columns:
            if col in specs:
                target_features.append(specs[col])
            else:
                target_features.append(self.car_data[col].median())
        
        # Scale the target features
        target_scaled = self.scaler.transform([target_features])
        
        # Calculate similarity with all cars
        similarities = cosine_similarity(target_scaled, self.feature_matrix)[0]
        
        # Get top N
        top_indices = np.argsort(similarities)[-n_recommendations:][::-1]
        
        recommendations = self.car_data.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarities[top_indices]
        
        return recommendations
    
    def get_diversity_score(self, recommendations: pd.DataFrame) -> float:
        """
        Calculate diversity score for recommendations.
        Measures how varied the recommendations are.
        
        Args:
            recommendations: DataFrame with recommended cars
            
        Returns:
            Diversity score between 0 and 1
        """
        if len(recommendations) <= 1:
            return 0.0
        
        # Calculate diversity based on brand variety and price range
        brand_col = 'Company_Clean' if 'Company_Clean' in recommendations.columns else 'Company Names'
        
        unique_brands = recommendations[brand_col].nunique()
        brand_diversity = unique_brands / len(recommendations)
        
        # Price range diversity
        if 'Price_USD' in recommendations.columns:
            price_range = recommendations['Price_USD'].max() - recommendations['Price_USD'].min()
            avg_price = recommendations['Price_USD'].mean()
            price_diversity = min(price_range / (avg_price + 1e-6), 1.0)
        else:
            price_diversity = 0.5
        
        return (brand_diversity + price_diversity) / 2
    
    def get_coverage(self) -> float:
        """
        Calculate catalog coverage - what percentage of cars can be recommended.
        
        Returns:
            Coverage score between 0 and 1
        """
        if not self.is_fitted:
            return 0.0
        
        # Cars with valid feature vectors
        valid_features = ~np.isnan(self.feature_matrix).any(axis=1)
        return valid_features.sum() / len(self.feature_matrix)
    
    def explain_recommendation(self, 
                                 source_idx: int, 
                                 target_idx: int) -> pd.DataFrame:
        """
        Explain why a car was recommended by comparing features.
        
        Args:
            source_idx: Index of the source car
            target_idx: Index of the recommended car
            
        Returns:
            DataFrame comparing feature values and contributions
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        source_features = self.feature_matrix[source_idx]
        target_features = self.feature_matrix[target_idx]
        
        # Calculate feature-wise contribution to similarity
        contributions = source_features * target_features
        
        # Get raw values for display
        source_raw = self.car_data.iloc[source_idx][self.feature_columns]
        target_raw = self.car_data.iloc[target_idx][self.feature_columns]
        
        explanation = pd.DataFrame({
            'feature': self.feature_columns,
            'source_value': source_raw.values,
            'target_value': target_raw.values,
            'contribution': contributions
        }).sort_values('contribution', ascending=False)
        
        return explanation
