"""
ML Models for Congressional Trading Prediction

This script implements machine learning models to predict whether
congressional trades will outperform the market (S&P 500).

Implemented Models:
- Logistic Regression
- Random Forest
- XGBoost

Usage:
    python src/models.py --model logistic_regression
    python src/models.py --model random_forest
    python src/models.py --model xgboost
"""

import pandas as pd
import numpy as np
import argparse
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score,
)
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor


def load_data(file_path='data/trades_with_returns.csv'):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} trades with returns data.")
    return df


def prepare_features(df, target_col='outperformed_90d'):
    """
    Select features and target
    Args:
        df: Input DataFrame
        target_col: The target column to predict
        
    Returns:
        X, y: Features and target
    """

    # Filter rows where target is NaN, then sort chronologically
    df_clean = df.dropna(subset=[target_col]).copy()
    df_clean['traded_date'] = pd.to_datetime(df_clean['traded_date'])
    df_clean = df_clean.sort_values('traded_date').reset_index(drop=True)
    print(f"Dataset size after dropping NaN in {target_col}: {len(df_clean)}")
    print(f"Date range: {df_clean['traded_date'].min().date()} to {df_clean['traded_date'].max().date()}")

    num_features =[
        'age', 'num_committees', 'committee_power_score',
        'committee_sector_alignment',
        'spy_momentum_30d', 'spy_momentum_90d',
        'stock_momentum_30d', 'stock_momentum_60d',
        'sector_etf_momentum_30d',
    ]

    cat_features = [
        'party', 'chamber', 'gender', 'transaction_type',
        'is_leadership', 'is_committee_leader',
        'sector', 'trade_size_bucket',
    ]

    num_features =[f for f in num_features if f in df_clean.columns]
    cat_features =[f for f in cat_features if f in df_clean.columns]

    for col in cat_features:
        df_clean[col] = df_clean[col].astype(str)

    X =df_clean[num_features + cat_features]
    y= df_clean[target_col]

    return X, y, num_features, cat_features


def build_pipeline(num_features, cat_features, model_type='logistic_regression', task='classification'):
    """Build a modeling pipeline for classification or regression."""

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer',  SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features),
    ])

    if task == 'classification':
        if model_type == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

        elif model_type == 'xgboost':
            #scale_pos_weight is set dynamically in train_and_evaluate from y_train class counts
            model = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                random_state=42, eval_metric='logloss', verbosity=0,
            )

        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    elif task == 'regression':
        if model_type == 'logistic_regression':
            model = Ridge(alpha=1.0)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,  max_depth=8, min_samples_split=5, random_state=42,
            )

        elif model_type == 'xgboost':
            model = XGBRegressor(
                n_estimators= 300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                random_state=42, verbosity =0,
            )

        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    else:
        raise ValueError(f"Unknown task: {task}")

    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])


def cross_validate_temporal(X, y, pipeline, num_features, cat_features, task='classification', n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metric_label  = 'ROC-AUC' if task == 'classification' else 'R2'
    scores = []

    print(f"\nTimeSeriesSplit CV ({n_splits} folds):")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx],X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        fold_pipe = clone(pipeline)
        clf = fold_pipe.named_steps['classifier']

        if isinstance(clf, XGBClassifier):
            neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
            if pos > 0:
                clf.set_params(scale_pos_weight=neg / pos)

        fold_pipe.fit(X_tr, y_tr)

        if task == 'classification':
            score = roc_auc_score(y_te, fold_pipe.predict_proba(X_te)[:, 1])
        else:
            score = r2_score(y_te, fold_pipe.predict(X_te))

        scores.append(score)
        print(f"  Fold {fold + 1}: {metric_label}={score:.4f}  (train={len(X_tr)}, test={len(X_te)})")

    print(f"  Mean {metric_label}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    return scores


def train_and_evaluate(X, y, pipeline, num_features, cat_features, task='classification'):
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size:      {len(X_test)}")
    if task == 'classification':
        print(f"Class distribution — train (y=1): {y_train.mean():.2%} | test (y=1): {y_test.mean():.2%}")
    else:
        print(f"Target — train mean: {y_train.mean():.2f}%  test mean: {y_test.mean():.2f}%")

    classifier = pipeline.named_steps['classifier']
    if isinstance(classifier, XGBClassifier):
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        spw = neg / pos
        classifier.set_params(scale_pos_weight=spw)
        print(f"scale_pos_weight set to {spw:.4f} ({neg} neg / {pos} pos)")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n" + "=" * 40)
    print("MODEL PERFORMANCE  (80/20 temporal split)")
    print("=" * 40)

    if task == 'classification':
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            'Accuracy':  accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall':    recall_score(y_test, y_pred, zero_division=0),
            'F1-Score':  f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC':   roc_auc_score(y_test, y_prob),
        }
        for name, val in metrics.items():
            print(f"{name}: {val:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    else:
        directional = ((y_pred > 0) == (y_test > 0)).mean()
        metrics = {
            'R2':              r2_score(y_test, y_pred),
            'MAE (%)':         mean_absolute_error(y_test, y_pred),
            'Directional Acc': directional,
        }
        for name, val in metrics.items():
            print(f"{name}: {val:.4f}")

    #feature importance
    try:
        clf = pipeline.named_steps['classifier']
        pre = pipeline.named_steps['preprocessor']
        onehot_cols = pre.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features)
        feature_names = num_features + list(onehot_cols)

        if hasattr(clf, 'coef_'):
            imp = clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_
            imp_label = 'Coefficient'
        elif hasattr(clf, 'feature_importances_'):
            imp = clf.feature_importances_
            imp_label = 'Importance'
        else:
            imp = None

        if imp is not None:
            imp_df = pd.DataFrame({
                'Feature': feature_names,
                imp_label: imp,
                'Abs': np.abs(imp),
            }).sort_values('Abs', ascending=False)
            print(f"\nTop 10 Features ({imp_label}):")
            print(imp_df.head(10).to_string(index=False))
    except Exception as e:
        print(f"\nCould not extract feature importance: {e}")

    # TimeSeriesSplit cross-validation
    cross_validate_temporal(X, y, pipeline, num_features, cat_features, task=task)

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ML models for congressional trades.')
    parser.add_argument('--model', type=str, default='logistic_regression',
                        choices=['logistic_regression', 'random_forest', 'xgboost'],
                        help='Model type to train')
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help='classification (default) or regression on excess_return_90d')
    parser.add_argument('--target', type=str, default=None,
                        help='Override target column. Defaults: classification=outperformed_90d, regression=excess_return_90d')

    args = parser.parse_args()

    default_targets = {'classification': 'outperformed_90d', 'regression': 'excess_return_90d'}
    target_col = args.target or default_targets[args.task]
    print(f"Task: {args.task} | Model: {args.model} | Target: {target_col}")

    df = load_data()

    if df is not None:
        X, y, num_features, cat_features = prepare_features(df, target_col=target_col)
        pipeline = build_pipeline(num_features, cat_features, model_type=args.model, task=args.task)

        print(f"\nTraining {args.model} ({args.task})...")
        train_and_evaluate(X, y, pipeline, num_features, cat_features, task=args.task)

        print(f"\n{args.model} ({args.task}) complete.")