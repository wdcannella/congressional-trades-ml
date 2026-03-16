"""
Machine Learning Models for Congressional Trading Prediction
==========================================================

This script implements baseline machine learning models to predict whether
congressional trades will outperform the market (S&P 500).

Implemented Models:
- Logistic Regression

Usage:
    python src/models.py --model logistic_regression
"""

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.impute import SimpleImputer


def load_data(file_path='data/trades_with_returns.csv'):
    """Load the dataset with returns."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} trades with returns data.")
    return df


def prepare_features(df, target_col='outperformed_90d'):
    """
    Select features and target, and handle missing values in target.
    
    Args:
        df: Input DataFrame
        target_col: The target column to predict
        
    Returns:
        X, y: Features and target
    """
    # Filter rows where target is NaN
    df_clean = df.dropna(subset=[target_col]).copy()
    print(f"Dataset size after dropping NaN in {target_col}: {len(df_clean)}")
    
    # Define features
    # Numerical features
    num_features = [
        'trade_amount', 'age', 'num_committees', 
        'committee_power_score', 'filed_after_days'
    ]
    
    # Categorical features
    cat_features = [
        'party', 'chamber', 'gender', 'transaction_type', 'is_leadership', 'is_committee_leader'
    ]
    
    # Ensure columns exist
    num_features = [f for f in num_features if f in df_clean.columns]
    cat_features = [f for f in cat_features if f in df_clean.columns]
    
    # Force categorical features to be strings to avoid mixed type issues in OneHotEncoder
    for col in cat_features:
        df_clean[col] = df_clean[col].astype(str)
    
    X = df_clean[num_features + cat_features]
    y = df_clean[target_col]
    
    return X, y, num_features, cat_features


def build_pipeline(num_features, cat_features, model_type='logistic_regression'):
    """Build a preprocessing and modeling pipeline."""
    
    # Preprocessing for numerical data
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical data
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])
    
    # Select classifier
    if model_type == 'logistic_regression':
        classifier = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'random_forest':
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline


def train_and_evaluate(X, y, pipeline):
    """Split data, train model, and print evaluation metrics."""
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Class distribution (y=1): {y.mean():.2%}")
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    print("\n" + "="*40)
    print("MODEL PERFORMANCE")
    print("="*40)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature Importance
    try:
        classifier = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Get feature names after one-hot encoding
        onehot_cols = (preprocessor
                       .named_transformers_['cat']
                       .named_steps['onehot']
                       .get_feature_names_out(cat_features))
        
        feature_names = num_features + list(onehot_cols)
        
        if hasattr(classifier, 'coef_'):
            importance = classifier.coef_[0]
            importance_col = 'Coefficient'
        elif hasattr(classifier, 'feature_importances_'):
            importance = classifier.feature_importances_
            importance_col = 'Importance'
        else:
            importance = None

        if importance is not None:
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                importance_col: importance,
                'Abs_Importance': np.abs(importance)
            }).sort_values(by='Abs_Importance', ascending=False)
            
            print(f"\nTop 10 Most Influential Features ({importance_col}):")
            print(importance_df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"\nCould not extract feature importance: {e}")
        
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ML models for congressional trades.')
    parser.add_argument('--model', type=str, default='logistic_regression',
                       choices=['logistic_regression', 'random_forest'],
                       help='Model type to train')
    parser.add_argument('--target', type=str, default='outperformed_90d',
                       help='Target column (e.g., outperformed_30d, outperformed_90d)')
    
    args = parser.parse_args()
    
    print(f"Starting modeling pipeline for target: {args.target}")
    
    # 1. Load data
    df = load_data()
    
    if df is not None:
        # 2. Prepare features and target
        X, y, num_features, cat_features = prepare_features(df, target_col=args.target)
        
        # 3. Build pipeline
        pipeline = build_pipeline(num_features, cat_features, model_type=args.model)
        
        # 4. Train and evaluate
        print(f"\nTraining {args.model}...")
        trained_pipeline = train_and_evaluate(X, y, pipeline)
        
        # Save results summary if needed
        print(f"\n{args.model} training complete.")
