#!/usr/bin/env python3
"""
Simple XGBoost Example - Housing Price Prediction

This script shows the core XGBoost regressor implementation in a clean, easy-to-understand format.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

print("🏠 XGBoost Housing Price Prediction Example")
print("=" * 50)

# Create sample housing data
np.random.seed(42)
n_samples = 1000

# Generate realistic housing features
data = {
    'OverallQual': np.random.randint(1, 11, n_samples),      # Quality rating 1-10
    'GrLivArea': np.random.randint(800, 3000, n_samples),    # Living area sqft
    'TotalBsmtSF': np.random.randint(0, 2000, n_samples),    # Basement sqft
    'GarageCars': np.random.randint(0, 4, n_samples),        # Garage capacity
    'YearBuilt': np.random.randint(1950, 2021, n_samples),   # Year built
    'FullBath': np.random.randint(1, 4, n_samples),          # Number of bathrooms
    'Neighborhood': np.random.choice(['Downtown', 'Suburbs', 'Uptown', 'Riverside'], n_samples)
}

# Create target variable (SalePrice) with realistic relationships
price = (
    data['OverallQual'] * 25000 +           # Quality has major impact
    data['GrLivArea'] * 50 +                # Living area per sqft
    data['TotalBsmtSF'] * 30 +              # Basement per sqft
    data['GarageCars'] * 15000 +            # Garage adds value
    (data['YearBuilt'] - 1950) * 500 +      # Newer homes worth more
    data['FullBath'] * 8000 +               # More bathrooms = more value
    np.random.normal(0, 20000, n_samples)   # Random noise
)

# Add neighborhood premium
neighborhood_premium = {'Downtown': 50000, 'Suburbs': 20000, 'Uptown': 80000, 'Riverside': 30000}
for i, neighborhood in enumerate(data['Neighborhood']):
    price[i] += neighborhood_premium[neighborhood]

# Ensure positive prices
price = np.maximum(price, 50000)

# Create DataFrame
df = pd.DataFrame(data)
df['SalePrice'] = price

print(f"📊 Dataset created: {len(df)} samples")
print(f"💰 Price range: ${df['SalePrice'].min():,.0f} - ${df['SalePrice'].max():,.0f}")
print("\n🔍 Sample data:")
print(df.head())

print("\n" + "=" * 50)
print("🔧 Data Preprocessing")
print("=" * 50)

# Encode categorical variables
label_encoder = LabelEncoder()
df['Neighborhood_encoded'] = label_encoder.fit_transform(df['Neighborhood'])

# Select features for modeling
feature_columns = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 
                  'YearBuilt', 'FullBath', 'Neighborhood_encoded']

X = df[feature_columns]
y = df['SalePrice']

print(f"✅ Features selected: {len(feature_columns)}")
print(f"📈 Feature names: {feature_columns}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"🎯 Training set: {len(X_train)} samples")
print(f"🎯 Test set: {len(X_test)} samples")

print("\n" + "=" * 50)
print("🚀 XGBoost Model Training")
print("=" * 50)

# Initialize XGBoost Regressor
xgb_regressor = XGBRegressor(
    n_estimators=100,        # Number of boosting rounds
    max_depth=6,             # Maximum depth of trees
    learning_rate=0.1,       # Learning rate (eta)
    subsample=0.8,           # Subsample ratio of the training instances
    colsample_bytree=0.8,    # Subsample ratio of columns
    random_state=42,         # For reproducibility
    n_jobs=-1                # Use all available cores
)

print("📋 XGBoost Parameters:")
print(f"  • n_estimators: {xgb_regressor.n_estimators}")
print(f"  • max_depth: {xgb_regressor.max_depth}")
print(f"  • learning_rate: {xgb_regressor.learning_rate}")
print(f"  • subsample: {xgb_regressor.subsample}")
print(f"  • colsample_bytree: {xgb_regressor.colsample_bytree}")

print("\n⏳ Training model...")

# Train the model
xgb_regressor.fit(X_train, y_train)

print("✅ Training completed!")

print("\n" + "=" * 50)
print("📊 Model Evaluation")
print("=" * 50)

# Make predictions
y_pred = xgb_regressor.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📈 Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"📈 R² Score: {r2:.4f}")

print("\n🔍 Feature Importance:")
feature_importance = xgb_regressor.feature_importances_
for i, (feature, importance) in enumerate(zip(feature_columns, feature_importance)):
    print(f"  {i+1}. {feature:<20}: {importance:.4f}")

print("\n" + "=" * 50)
print("🎯 Sample Predictions")
print("=" * 50)

# Show some sample predictions
print(f"{'Actual Price':>15} {'Predicted Price':>17} {'Error':>12}")
print("-" * 50)

for i in range(5):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    error = abs(actual - predicted)
    print(f"${actual:>14,.0f} ${predicted:>16,.0f} ${error:>11,.0f}")

print("\n" + "=" * 50)
print("🎉 XGBoost Example Complete!")
print("=" * 50)
print(f"🏆 Final Model Performance: MAE = ${mae:,.0f}, R² = {r2:.3f}")
print("✨ XGBoost successfully trained and evaluated!")