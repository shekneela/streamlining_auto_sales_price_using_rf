import argparse
import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn


def load_data(folder, filename):
    path = os.path.join(folder, filename)
    return pd.read_csv(path)


def main(args):
    mlflow.start_run()

    # Load train and test data
    train_df = load_data(args.train_data, "train.csv")
    test_df = load_data(args.test_data, "test.csv")

    target_col = "price"
    feature_cols = [c for c in train_df.columns if c != target_col]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Feature engineering: define categorical and numeric columns
    categorical_features = ["Segment"]
    numeric_features = [c for c in feature_cols if c != "Segment"]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Define model
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    )

    # Build full pipeline
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", rf),
    ])

    # Train
    model_pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

    # Log parameters and metrics
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Save model
    os.makedirs(args.model_output, exist_ok=True)
    model_path = os.path.join(args.model_output, "model.joblib")
    joblib.dump(model_pipeline, model_path)

    # Log model to MLflow
    mlflow.sklearn.log_model(model_pipeline, "model")

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args()
    main(args)
