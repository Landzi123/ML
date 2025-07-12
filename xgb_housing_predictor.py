"""
XGBoost Housing Price Predictor

This module provides a clean implementation of XGBoost for housing price prediction,
extracted and fixed from the original Jupyter notebook.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


class HousingPricePredictor:
    """
    A class for predicting housing prices using XGBoost and Random Forest models.
    """
    
    def __init__(self, random_state=0):
        self.random_state = random_state
        self.models = {}
        self.preprocessor = None
        self.numerical_features = [
            'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars',
            '1stFlrSF', 'FullBath', 'YearBuilt', 'TotRmsAbvGrd'
        ]
        self.categorical_features = [
            'Neighborhood', 'KitchenQual', 'GarageFinish', 'BsmtQual',
            'ExterQual', 'HeatingQC'
        ]
    
    def load_data(self, train_path, test_path=None):
        """
        Load training and test data from CSV files.
        
        Args:
            train_path (str): Path to training data CSV file
            test_path (str, optional): Path to test data CSV file
        
        Returns:
            tuple: (X_train, y_train, X_test) or (X_train, y_train) if no test data
        """
        try:
            # Load training data
            train_data = pd.read_csv(train_path, index_col='Id')
            
            # Separate features and target
            X = train_data.drop('SalePrice', axis=1)
            y = train_data['SalePrice']
            
            # Load test data if provided
            X_test = None
            if test_path:
                X_test = pd.read_csv(test_path, index_col='Id')
            
            return (X, y, X_test) if X_test is not None else (X, y)
            
        except FileNotFoundError as e:
            print(f"Error: Could not find data file. {e}")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, X):
        """
        Preprocess the data by handling missing values and feature selection.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        # Create a copy to avoid modifying the original data
        X_processed = X.copy()
        
        # Handle missing values
        # For numerical features: fill with 0
        num_cols = X_processed.select_dtypes(include=['int64', 'float64']).columns
        X_processed[num_cols] = X_processed[num_cols].fillna(0)
        
        # For categorical features: fill with a default value
        cat_cols = X_processed.select_dtypes(include=['object']).columns
        X_processed[cat_cols] = X_processed[cat_cols].fillna("Do_not_have_this_feature")
        
        return X_processed
    
    def create_preprocessor(self):
        """
        Create preprocessing pipeline for numerical and categorical features.
        
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        # Numerical transformer
        numerical_transformer = SimpleImputer(strategy='constant', fill_value=0)
        
        # Categorical transformer
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Do_not_have_this_feature')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, self.numerical_features),
            ('cat', categorical_transformer, self.categorical_features)
        ])
        
        return preprocessor
    
    def train_random_forest(self, X_train, y_train, n_estimators=100):
        """
        Train a Random Forest model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training targets
            n_estimators (int): Number of trees in the forest
            
        Returns:
            Pipeline: Trained Random Forest pipeline
        """
        preprocessor = self.create_preprocessor()
        
        rf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=self.random_state))
        ])
        
        rf_pipeline.fit(X_train, y_train)
        self.models['random_forest'] = rf_pipeline
        
        return rf_pipeline
    
    def train_xgboost(self, X_train, y_train, X_valid=None, y_valid=None, 
                     n_estimators=1000, learning_rate=0.1, early_stopping_rounds=10):
        """
        Train an XGBoost model with optional early stopping.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training targets
            X_valid (pd.DataFrame, optional): Validation features for early stopping
            y_valid (pd.Series, optional): Validation targets for early stopping
            n_estimators (int): Maximum number of boosting rounds
            learning_rate (float): Learning rate
            early_stopping_rounds (int): Early stopping rounds
            
        Returns:
            Pipeline or XGBRegressor: Trained XGBoost model
        """
        preprocessor = self.create_preprocessor()
        
        if X_valid is not None and y_valid is not None:
            # Use early stopping - we need to preprocess data manually
            # because Pipeline doesn't support early_stopping_rounds parameter
            
            # Fit preprocessor and transform data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_valid_processed = preprocessor.transform(X_valid)
            
            # Train XGBoost with early stopping
            xgb_model = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=self.random_state,
                early_stopping_rounds=early_stopping_rounds,
                eval_metric='mae'
            )
            
            xgb_model.fit(
                X_train_processed, y_train,
                eval_set=[(X_valid_processed, y_valid)],
                verbose=False
            )
            
            # Store both preprocessor and model for prediction
            self.models['xgboost'] = {
                'preprocessor': preprocessor,
                'model': xgb_model
            }
            
            return xgb_model
        else:
            # Use Pipeline without early stopping
            xgb_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    random_state=self.random_state
                ))
            ])
            
            xgb_pipeline.fit(X_train, y_train)
            self.models['xgboost'] = xgb_pipeline
            
            return xgb_pipeline
    
    def predict(self, X, model_name='xgboost'):
        """
        Make predictions using the specified model.
        
        Args:
            X (pd.DataFrame): Features for prediction
            model_name (str): Name of the model to use ('xgboost' or 'random_forest')
            
        Returns:
            np.ndarray: Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Handle XGBoost with early stopping (stored as dict)
        if isinstance(model, dict) and 'preprocessor' in model:
            X_processed = model['preprocessor'].transform(X)
            return model['model'].predict(X_processed)
        else:
            # Handle Pipeline models
            return model.predict(X)
    
    def evaluate(self, X_test, y_test, model_name='xgboost'):
        """
        Evaluate model performance using Mean Absolute Error.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test targets
            model_name (str): Name of the model to evaluate
            
        Returns:
            float: Mean Absolute Error
        """
        predictions = self.predict(X_test, model_name)
        return mean_absolute_error(y_test, predictions)
    
    def compare_models(self, X_train, y_train, X_valid, y_valid):
        """
        Train and compare Random Forest and XGBoost models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training targets
            X_valid (pd.DataFrame): Validation features
            y_valid (pd.Series): Validation targets
            
        Returns:
            dict: Model comparison results
        """
        results = {}
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_model = self.train_random_forest(X_train, y_train)
        rf_mae = self.evaluate(X_valid, y_valid, 'random_forest')
        results['Random Forest'] = rf_mae
        print(f"📊 MAE (Random Forest): {rf_mae:.0f}")
        
        # Train XGBoost with early stopping
        print("Training XGBoost...")
        xgb_model = self.train_xgboost(X_train, y_train, X_valid, y_valid)
        xgb_mae = self.evaluate(X_valid, y_valid, 'xgboost')
        results['XGBoost'] = xgb_mae
        print(f"📊 MAE (XGBoost): {xgb_mae:.0f}")
        
        # Determine best model
        best_model = min(results, key=results.get)
        print(f"🏆 Best model: {best_model} (MAE: {results[best_model]:.0f})")
        
        return results


def main():
    """
    Example usage of the HousingPricePredictor class.
    """
    # Initialize predictor
    predictor = HousingPricePredictor(random_state=0)
    
    # Example with dummy data (replace with actual data loading)
    print("This is a demonstration of the XGBoost Housing Price Predictor.")
    print("To use with real data, call:")
    print("predictor.load_data('path/to/train.csv', 'path/to/test.csv')")
    print("\nExample usage:")
    print("X, y = predictor.load_data('train.csv')")
    print("X_processed = predictor.preprocess_data(X)")
    print("X_train, X_valid, y_train, y_valid = train_test_split(X_processed, y, test_size=0.2)")
    print("results = predictor.compare_models(X_train, y_train, X_valid, y_valid)")


if __name__ == "__main__":
    main()