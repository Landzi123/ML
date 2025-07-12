"""
Test script for XGBoost Housing Price Predictor

This script demonstrates the fixed XGBoost implementation with sample data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from xgb_housing_predictor import HousingPricePredictor


def create_sample_housing_data(n_samples=1000, n_features=8):
    """
    Create sample housing data for testing.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        
    Returns:
        tuple: (X, y) - Features and target
    """
    # Generate base regression data
    X_base, y_base = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=10000,
        random_state=42
    )
    
    # Create realistic housing features
    feature_names = [
        'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars',
        '1stFlrSF', 'FullBath', 'YearBuilt', 'TotRmsAbvGrd'
    ]
    
    # Scale features to realistic ranges
    X_scaled = X_base.copy()
    X_scaled[:, 0] = np.clip(X_scaled[:, 0] * 0.5 + 6, 1, 10)  # OverallQual (1-10)
    X_scaled[:, 1] = np.clip(X_scaled[:, 1] * 200 + 1500, 500, 5000)  # GrLivArea
    X_scaled[:, 2] = np.clip(X_scaled[:, 2] * 150 + 1000, 0, 3000)  # TotalBsmtSF
    X_scaled[:, 3] = np.clip(X_scaled[:, 3] * 0.3 + 2, 0, 4)  # GarageCars
    X_scaled[:, 4] = np.clip(X_scaled[:, 4] * 150 + 1000, 500, 3000)  # 1stFlrSF
    X_scaled[:, 5] = np.clip(X_scaled[:, 5] * 0.3 + 2, 1, 4)  # FullBath
    X_scaled[:, 6] = np.clip(X_scaled[:, 6] * 10 + 1980, 1950, 2020)  # YearBuilt
    X_scaled[:, 7] = np.clip(X_scaled[:, 7] * 0.5 + 6, 3, 12)  # TotRmsAbvGrd
    
    # Create DataFrame
    X = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Create categorical features
    neighborhoods = ['Downtown', 'Suburbs', 'Uptown', 'Riverside', 'Hills']
    qualities = ['Excellent', 'Good', 'Fair', 'Poor']
    
    X['Neighborhood'] = np.random.choice(neighborhoods, size=n_samples)
    X['KitchenQual'] = np.random.choice(qualities, size=n_samples)
    X['GarageFinish'] = np.random.choice(['Finished', 'Unfinished', 'Do_not_have_this_feature'], 
                                        size=n_samples, p=[0.6, 0.3, 0.1])
    X['BsmtQual'] = np.random.choice(qualities + ['Do_not_have_this_feature'], 
                                    size=n_samples, p=[0.3, 0.4, 0.2, 0.05, 0.05])
    X['ExterQual'] = np.random.choice(qualities, size=n_samples)
    X['HeatingQC'] = np.random.choice(qualities, size=n_samples)
    
    # Adjust target to be realistic house prices
    y = np.clip(y_base * 10 + 200000, 50000, 800000)
    
    return X, y


def test_xgb_predictor():
    """
    Test the XGBoost Housing Price Predictor with sample data.
    """
    print("=" * 60)
    print("Testing XGBoost Housing Price Predictor")
    print("=" * 60)
    
    # Create sample data
    print("Creating sample housing data...")
    X, y = create_sample_housing_data(n_samples=1000)
    
    print(f"Data shape: {X.shape}")
    print(f"Target range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"Sample features:\n{X.head()}")
    
    # Initialize predictor
    predictor = HousingPricePredictor(random_state=0)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_processed = predictor.preprocess_data(X)
    
    # Split data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_processed, y, test_size=0.2, random_state=0
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_valid.shape[0]}")
    
    # Test XGBoost without early stopping
    print("\n" + "=" * 40)
    print("Testing XGBoost without early stopping")
    print("=" * 40)
    
    xgb_model = predictor.train_xgboost(
        X_train, y_train, 
        n_estimators=100,  # Fewer estimators for faster testing
        learning_rate=0.1
    )
    
    mae_no_early = predictor.evaluate(X_valid, y_valid, 'xgboost')
    print(f"MAE without early stopping: ${mae_no_early:,.0f}")
    
    # Test XGBoost with early stopping
    print("\n" + "=" * 40)
    print("Testing XGBoost with early stopping")
    print("=" * 40)
    
    # Reset models to test early stopping
    predictor.models = {}
    
    xgb_model_early = predictor.train_xgboost(
        X_train, y_train, 
        X_valid, y_valid,
        n_estimators=1000,
        learning_rate=0.1,
        early_stopping_rounds=10
    )
    
    mae_early = predictor.evaluate(X_valid, y_valid, 'xgboost')
    print(f"MAE with early stopping: ${mae_early:,.0f}")
    
    # Compare models
    print("\n" + "=" * 40)
    print("Comparing Random Forest vs XGBoost")
    print("=" * 40)
    
    # Reset models for comparison
    predictor.models = {}
    
    results = predictor.compare_models(X_train, y_train, X_valid, y_valid)
    
    print(f"\nModel Comparison Results:")
    for model_name, mae in results.items():
        print(f"  {model_name}: ${mae:,.0f}")
    
    # Test predictions
    print("\n" + "=" * 40)
    print("Testing predictions")
    print("=" * 40)
    
    # Make predictions on a few samples
    sample_X = X_valid.head(5)
    sample_y = y_valid[:5]  # Fix for numpy array
    
    rf_preds = predictor.predict(sample_X, 'random_forest')
    xgb_preds = predictor.predict(sample_X, 'xgboost')
    
    print("Sample predictions:")
    print(f"{'Actual':>12} {'Random Forest':>15} {'XGBoost':>12} {'RF Error':>12} {'XGB Error':>12}")
    print("-" * 75)
    
    for i in range(len(sample_y)):
        actual = sample_y[i]  # Fix for numpy array
        rf_pred = rf_preds[i]
        xgb_pred = xgb_preds[i]
        rf_error = abs(actual - rf_pred)
        xgb_error = abs(actual - xgb_pred)
        
        print(f"${actual:>11,.0f} ${rf_pred:>14,.0f} ${xgb_pred:>11,.0f} ${rf_error:>11,.0f} ${xgb_error:>11,.0f}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_xgb_predictor()