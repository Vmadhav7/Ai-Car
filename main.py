"""
Main entry point for the AI-Powered Car Intelligence System.

This script demonstrates the full pipeline:
1. Data loading and cleaning
2. Feature engineering
3. Price prediction (regression)
4. Performance classification
5. Car recommendations
6. Model explainability

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from data.data_loader import load_cars_data, validate_data, get_data_summary
from data.data_cleaner import clean_dataset, remove_invalid_rows
from data.feature_engineer import (
    create_all_features, 
    prepare_features_for_modeling,
    get_feature_matrix_for_recommendation
)
from models.price_predictor import PricePredictor
from models.classifier import PerformanceClassifier
from models.recommender import CarRecommender
from analytics.eda import generate_eda_report, analyze_price_distribution
from analytics.visualizer import create_all_visualizations
from explainability.shap_explainer import get_model_based_importance


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    """Run the full car intelligence pipeline."""
    
    print_section("AI-POWERED CAR INTELLIGENCE SYSTEM")
    
    # ============================================================
    # 1. DATA LOADING
    # ============================================================
    print_section("1. Loading Data")
    
    try:
        df = load_cars_data()
        print(f"[OK] Loaded {len(df)} cars from dataset")
        
        # Validate data
        validation = validate_data(df)
        print(f"[OK] Dataset has {validation['total_columns']} columns")
        print(f"[OK] {validation['unique_companies']} unique car manufacturers")
        
        if validation['issues']:
            print("\n[!] Data quality issues found:")
            for issue in validation['issues']:
                print(f"  - {issue}")
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return
    
    # ============================================================
    # 2. DATA CLEANING
    # ============================================================
    print_section("2. Cleaning Data")
    
    df_cleaned = clean_dataset(df)
    print(f"[OK] Parsed numeric values from all columns")
    print(f"[OK] Standardized fuel types to {df_cleaned['Fuel_Type_Std'].nunique()} categories")
    
    # Remove invalid rows
    df_valid = remove_invalid_rows(df_cleaned)
    removed = len(df_cleaned) - len(df_valid)
    print(f"[OK] Removed {removed} rows with invalid prices")
    print(f"[OK] Valid dataset: {len(df_valid)} cars")
    
    # ============================================================
    # 3. FEATURE ENGINEERING
    # ============================================================
    print_section("3. Feature Engineering")
    
    df_features = create_all_features(df_valid)
    print(f"[OK] Created Performance_Category feature")
    print(f"[OK] Created Is_Luxury_Brand feature")
    print(f"[OK] Created Price_Per_HP and Power_To_Speed_Ratio features")
    print(f"[OK] Created Is_Electric feature")
    
    # Show category distribution
    category_counts = df_features['Performance_Category'].value_counts()
    print(f"\nPerformance Category Distribution:")
    for cat, count in category_counts.items():
        print(f"  - {cat}: {count} cars ({count/len(df_features)*100:.1f}%)")
    
    # ============================================================
    # 4. EXPLORATORY DATA ANALYSIS
    # ============================================================
    print_section("4. Exploratory Data Analysis")
    
    price_stats = analyze_price_distribution(df_features)
    print(f"Price Statistics:")
    print(f"  - Mean: ${price_stats['mean']:,.0f}")
    print(f"  - Median: ${price_stats['median']:,.0f}")
    print(f"  - Min: ${price_stats['min']:,.0f}")
    print(f"  - Max: ${price_stats['max']:,.0f}")
    
    print(f"\nPrice Segments:")
    for segment, count in price_stats['price_segments'].items():
        print(f"  - {segment}: {count} cars")
    
    # Generate visualizations
    print(f"\n[OK] Generating visualizations...")
    try:
        output_dir = Path(__file__).parent / 'outputs' / 'figures'
        saved_files = create_all_visualizations(df_features, str(output_dir))
        print(f"[OK] Saved {len(saved_files)} visualization(s) to outputs/figures/")
    except Exception as e:
        print(f"[!] Could not generate some visualizations: {e}")
    
    # ============================================================
    # 5. PRICE PREDICTION MODEL
    # ============================================================
    print_section("5. Price Prediction (Regression)")
    
    # Prepare features
    feature_cols = ['HorsePower_Numeric', 'Speed_KMH', 'Acceleration_Sec',
                    'Torque_Nm', 'CC_Numeric', 'Seats_Numeric']
    
    # Filter valid rows for modeling
    model_df = df_features.dropna(subset=feature_cols + ['Price_USD'])
    X = model_df[feature_cols]
    y = model_df['Price_USD']
    
    print(f"Training on {len(model_df)} cars with complete data...\n")
    
    # Train price predictor
    price_predictor = PricePredictor(use_xgboost=True)
    results = price_predictor.train(X, y, validate=True)
    
    print("Model Performance (Validation Set):")
    for model_name, model_results in results.items():
        if model_results['status'] == 'success' and 'val_metrics' in model_results:
            metrics = model_results['val_metrics']
            print(f"\n  {model_name}:")
            print(f"    - R² Score: {metrics['r2']:.3f}")
            print(f"    - RMSE: ${metrics['rmse']:,.0f}")
            print(f"    - MAE: ${metrics['mae']:,.0f}")
    
    print(f"\n[OK] Best model: {price_predictor.best_model_name}")
    
    # Feature importance
    importance = price_predictor.get_feature_importance()
    print(f"\nTop Features for Price Prediction:")
    for _, row in importance.head(5).iterrows():
        print(f"  - {row['feature']}: {row['importance']:.3f}")
    
    # ============================================================
    # 6. PERFORMANCE CLASSIFICATION
    # ============================================================
    print_section("6. Performance Classification")
    
    # Prepare classification target
    class_df = model_df[model_df['Performance_Category'] != 'Unknown']
    X_class = class_df[feature_cols]
    y_class = class_df['Performance_Category']
    
    print(f"Training on {len(class_df)} cars...\n")
    
    # Train classifier
    classifier = PerformanceClassifier(use_xgboost=True)
    class_results = classifier.train(X_class, y_class, validate=True)
    
    print("Model Performance (Validation Set):")
    for model_name, model_results in class_results.items():
        if model_results['status'] == 'success' and 'val_metrics' in model_results:
            metrics = model_results['val_metrics']
            print(f"\n  {model_name}:")
            print(f"    - Accuracy: {metrics['accuracy']:.3f}")
            print(f"    - F1 Score (weighted): {metrics['f1_weighted']:.3f}")
    
    print(f"\n[OK] Best model: {classifier.best_model_name}")
    
    # ============================================================
    # 7. CAR RECOMMENDATIONS
    # ============================================================
    print_section("7. Car Recommendation System")
    
    # Initialize recommender
    recommender = CarRecommender()
    recommender.fit(df_features)
    
    print(f"[OK] Fitted recommender on {len(df_features)} cars")
    print(f"[OK] Catalog coverage: {recommender.get_coverage()*100:.1f}%")
    
    # Demo recommendation
    sample_idx = 0
    sample_car = df_features.iloc[sample_idx]
    print(f"\nExample: Cars similar to {sample_car['Company_Clean']} {sample_car['Car_Name_Clean']}:")
    
    recommendations = recommender.get_similar_cars(sample_idx, n_recommendations=5)
    for _, car in recommendations.iterrows():
        print(f"  - {car['Company_Clean']} {car['Car_Name_Clean']} "
              f"(${car['Price_USD']:,.0f}, {car['HorsePower_Numeric']:.0f} HP) "
              f"- Similarity: {car['similarity_score']:.3f}")
    
    # ============================================================
    # 8. SAVE MODELS
    # ============================================================
    print_section("8. Saving Models")
    
    models_dir = Path(__file__).parent / 'outputs' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    price_predictor.save_model(str(models_dir / 'price_predictor.joblib'))
    classifier.save_model(str(models_dir / 'performance_classifier.joblib'))
    
    print(f"[OK] Saved price predictor to outputs/models/price_predictor.joblib")
    print(f"[OK] Saved classifier to outputs/models/performance_classifier.joblib")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print_section("PIPELINE COMPLETE")
    
    print("Summary:")
    print(f"  - Processed {len(df_valid)} cars from {df_valid['Company_Clean'].nunique()} manufacturers")
    print(f"  - Price Prediction R² Score: {results[price_predictor.best_model_name]['val_metrics']['r2']:.3f}")
    print(f"  - Classification Accuracy: {class_results[classifier.best_model_name]['val_metrics']['accuracy']:.3f}")
    print(f"  - Recommendation System: Ready with {recommender.get_coverage()*100:.1f}% coverage")
    
    print("\nNext Steps:")
    print("  1. Explore notebooks/ for detailed analysis")
    print("  2. Review outputs/figures/ for visualizations")
    print("  3. Use saved models for inference")
    
    return df_features, price_predictor, classifier, recommender


if __name__ == '__main__':
    main()
