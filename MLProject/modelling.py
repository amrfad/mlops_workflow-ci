# modelling.py (for MLProject)
import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

def train_model(n_estimators, max_depth):
    # Load preprocessed training data (relative to MLProject directory)
    train_data_path = "FD001_preprocessing/FD001_preprocessing_train.csv"
    print(f"Loading preprocessed training data from {train_data_path}...")
    train = pd.read_csv(train_data_path)

    # Split by unit_nr to prevent data leakage
    train_units = train['unit_nr'].unique()
    split_idx = int(len(train_units) * 0.8)
    train_set = train[train['unit_nr'].isin(train_units[:split_idx])]
    val_set = train[train['unit_nr'].isin(train_units[split_idx:])]

    X_train = train_set.drop(['unit_nr', 'time_cycles', 'RUL'], axis=1)
    y_train = train_set['RUL']
    X_val = val_set.drop(['unit_nr', 'time_cycles', 'RUL'], axis=1)
    y_val = val_set['RUL']

    print(f"Train features shape: {X_train.shape}, Validation features shape: {X_val.shape}")
    print(f"Parameters - n_estimators: {n_estimators}, max_depth: {max_depth}")

    with mlflow.start_run():
        # Enable autologging
        mlflow.sklearn.autolog()
        
        print("Training RandomForestRegressor...")
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Predict on validation set
        y_pred = model.predict(X_val)
        
        # Calculate validation metrics
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        print(f"Validation Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        # Log additional validation metrics
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_rmse", rmse)
        mlflow.log_metric("val_r2", r2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=str, default="20")
    args = parser.parse_args()
    
    # Handle max_depth which can be "None" or integer string
    try:
        max_depth_val = int(args.max_depth)
    except ValueError:
        max_depth_val = None
        
    train_model(n_estimators=args.n_estimators, max_depth=max_depth_val)
