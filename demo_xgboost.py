"""
XGBoost Housing Price Prediction Demo

This script demonstrates the XGBoost regressor implementation for housing price prediction.
It shows the key components of the model including preprocessing, training, and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

def create_demo_data():
    """Create demo housing data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic housing features
    data = {
        'OverallQual': np.random.randint(1, 11, n_samples),
        'GrLivArea': np.random.randint(800, 3000, n_samples),
        'TotalBsmtSF': np.random.randint(0, 2000, n_samples),
        'GarageCars': np.random.randint(0, 4, n_samples),
        '1stFlrSF': np.random.randint(500, 2500, n_samples),
        'FullBath': np.random.randint(1, 4, n_samples),
        'YearBuilt': np.random.randint(1950, 2021, n_samples),
        'TotRmsAbvGrd': np.random.randint(3, 12, n_samples),
        'Neighborhood': np.random.choice(['Downtown', 'Suburbs', 'Uptown', 'Riverside'], n_samples),
        'KitchenQual': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_samples),
        'ExterQual': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create target variable with some realistic relationship
    price = (
        df['OverallQual'] * 25000 +
        df['GrLivArea'] * 50 +
        df['TotalBsmtSF'] * 30 +
        df['GarageCars'] * 15000 +
        df['FullBath'] * 8000 +
        (df['YearBuilt'] - 1950) * 500 +
        np.random.normal(0, 20000, n_samples)
    )
    
    # Add neighborhood premium
    neighborhood_premium = {
        'Downtown': 50000,
        'Suburbs': 20000,
        'Uptown': 80000,
        'Riverside': 30000
    }
    
    for neighborhood, premium in neighborhood_premium.items():
        price[df['Neighborhood'] == neighborhood] += premium
    
    # Ensure positive prices
    price = np.maximum(price, 50000)
    df['SalePrice'] = price
    
    return df

def preprocess_data(df):
    """Preprocess the data for XGBoost"""
    
    # Separate features and target
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    
    # Separate numerical and categorical features
    numerical_cols = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 
                     '1stFlrSF', 'FullBath', 'YearBuilt', 'TotRmsAbvGrd']
    categorical_cols = ['Neighborhood', 'KitchenQual', 'ExterQual']
    
    # Handle missing values (fill with median for numerical, mode for categorical)
    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])
    
    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols)
    
    return X_encoded, y

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model with demonstration of parameters"""
    
    print("🚀 Training XGBoost Regressor")
    print("=" * 50)
    
    # Initialize XGBoost model with commonly used parameters
    xgb_model = XGBRegressor(
        n_estimators=100,           # Number of boosting rounds
        max_depth=6,                # Maximum depth of trees
        learning_rate=0.1,          # Learning rate (eta)
        subsample=0.8,              # Subsample ratio
        colsample_bytree=0.8,       # Column subsample ratio
        random_state=42,            # For reproducibility
        eval_metric='mae'           # Evaluation metric
    )
    
    # Train the model
    print(f"Training with {len(X_train)} samples...")
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print(f"✅ Training completed!")
    
    # Only show best iteration if early stopping was used
    try:
        print(f"Best iteration: {xgb_model.best_iteration}")
        print(f"Best score: {xgb_model.best_score:.2f}")
    except AttributeError:
        print("Early stopping not used - all iterations completed.")
    
    return xgb_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the XGBoost model"""
    
    print("\n📊 Model Evaluation")
    print("=" * 50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Square Error (RMSE): ${rmse:,.2f}")
    print(f"R² Score: {model.score(X_test, y_test):.4f}")
    
    return y_pred, mae, rmse

def show_feature_importance(model, X_train):
    """Display feature importance from XGBoost model"""
    
    print("\n🔍 Feature Importance")
    print("=" * 50)
    
    # Get feature importance
    importance = model.feature_importances_
    feature_names = X_train.columns
    
    # Create DataFrame for easy sorting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Display top 10 features
    print("Top 10 most important features:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<20}: {row['importance']:.4f}")
    
    return importance_df

def demonstrate_predictions(model, X_test, y_test):
    """Show sample predictions"""
    
    print("\n🎯 Sample Predictions")
    print("=" * 50)
    
    # Make predictions on first 5 samples
    sample_X = X_test.head(5)
    sample_y = y_test.head(5)
    predictions = model.predict(sample_X)
    
    print(f"{'Actual Price':>15} {'Predicted Price':>17} {'Error':>12}")
    print("-" * 50)
    
    for i in range(len(sample_y)):
        actual = sample_y.iloc[i]
        predicted = predictions[i]
        error = abs(actual - predicted)
        print(f"${actual:>14,.0f} ${predicted:>16,.0f} ${error:>11,.0f}")

def main():
    """Main demonstration function"""
    
    print("🏠 XGBoost Housing Price Prediction Demo")
    print("=" * 60)
    
    # Create demo data
    print("📊 Creating demo housing data...")
    df = create_demo_data()
    print(f"Created dataset with {len(df)} samples and {len(df.columns)} features")
    print(f"Price range: ${df['SalePrice'].min():,.0f} - ${df['SalePrice'].max():,.0f}")
    
    # Preprocess data
    print("\n🔧 Preprocessing data...")
    X, y = preprocess_data(df)
    print(f"Processed features: {X.shape[1]} columns after encoding")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train XGBoost model
    model = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    y_pred, mae, rmse = evaluate_model(model, X_test, y_test)
    
    # Show feature importance
    importance_df = show_feature_importance(model, X_train)
    
    # Show sample predictions
    demonstrate_predictions(model, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("✅ XGBoost demonstration completed successfully!")
    print(f"Final model performance: MAE = ${mae:,.0f}")
    print("=" * 60)

if __name__ == "__main__":
    main()