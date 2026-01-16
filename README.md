# AI-Powered Car Intelligence System

An end-to-end machine learning project for car price prediction, performance classification, and similar car recommendations.

## Live Demo (Deployed)
https://ai-car.streamlit.app/

##  Project Overview

This project provides an AI-powered system to:
- **Predict car prices** based on specifications (regression)
- **Classify cars** into performance categories: High-Performance, Mid-Range, Economy
- **Recommend similar cars** using content-based filtering
- **Analyze and visualize** car market data

##  Project Structure

```
cars/
├── data/
│   └── Cars Datasets 2025.csv      # Raw dataset (1,218 cars)
├── src/
│   ├── data/                        # Data processing modules
│   │   ├── data_loader.py
│   │   ├── data_cleaner.py
│   │   └── feature_engineer.py
│   ├── models/                      # ML models
│   │   ├── price_predictor.py
│   │   ├── classifier.py
│   │   └── recommender.py
│   ├── analytics/                   # EDA and visualization
│   │   ├── eda.py
│   │   └── visualizer.py
│   ├── explainability/             # Model explanations
│   │   └── shap_explainer.py
│   └── utils/
│       └── helpers.py
├── notebooks/                       # Jupyter notebooks
├── outputs/
│   ├── models/                      # Saved model files
│   └── figures/                     # Generated visualizations
├── tests/                           # Unit tests
├── main.py                          # Main pipeline script
├── requirements.txt                 # Python dependencies
└── README.md
```

##  Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

##  Quick Start

```bash
# Run the full pipeline
python main.py
```

##  Dataset

The dataset contains **1,218 cars** from **37 manufacturers** with the following features:

| Feature | Description |
|---------|-------------|
| Company Names | Manufacturer (Ferrari, Toyota, etc.) |
| Cars Names | Model name |
| Engines | Engine type (V8, V12, Electric, etc.) |
| CC/Battery Capacity | Engine displacement or battery capacity |
| HorsePower | Power output in HP |
| Total Speed | Maximum speed in km/h |
| Performance (0-100) | Acceleration time (seconds) |
| Cars Prices | Price in USD |
| Fuel Types | Petrol, Diesel, Hybrid, Electric, etc. |
| Seats | Seating capacity |
| Torque | Engine torque in Nm |

##  Models

### 1. Price Prediction (Regression)
- **Models**: Random Forest, XGBoost, Gradient Boosting, Ridge
- **Metrics**: R², RMSE, MAE, MAPE
- **Best Performance**: ~0.80+ R² score

### 2. Performance Classification
- **Categories**: High-Performance (<4.5s), Mid-Range (4.5-8s), Economy (>8s)
- **Models**: Random Forest, XGBoost, Logistic Regression
- **Metrics**: Accuracy, Precision, Recall, F1-score

### 3. Recommendation System
- **Approach**: Content-based filtering with cosine similarity
- **Features**: Horsepower, speed, acceleration, price, torque, seats

##  Usage Examples

### Price Prediction
```python
from src.models.price_predictor import PricePredictor

predictor = PricePredictor.load_model('outputs/models/price_predictor.joblib')
predicted_price = predictor.predict(features_df)
```

### Performance Classification
```python
from src.models.classifier import PerformanceClassifier

classifier = PerformanceClassifier.load_model('outputs/models/performance_classifier.joblib')
category = classifier.predict(features_df)
```

### Car Recommendations
```python
from src.models.recommender import CarRecommender

recommender = CarRecommender()
recommender.fit(cars_df)
similar_cars = recommender.get_similar_cars(car_index=0, n_recommendations=5)
```

