import argparse
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

def load_data(folder, filename):
    path = os.path.join(folder, filename)
    return pd.read_csv(path)

def main(args):
    mlflow.start_run()

    # Load train/test data
    train_df = load_data(args.train_data, "train.csv")
    test_df = load_data(args.test_data, "test.csv")

    print("Training data sample:")
    print(train_df.head())
    print("Data types:")
    print(train_df.dtypes)

    target_col = "price"
    feature_cols = [c for c in train_df.columns if c != target_col]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    categorical_features = ["Segment"]
    numeric_features = [c for c in feature_cols if c != "Segment"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    criterion = args.criterion

    rf = RandomForestRegressor(
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth) if args.max_depth else None,
        criterion=criterion,
        random_state=42,
        n_jobs=-1,
    )

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", rf)
    ])

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    mlflow.log_params({
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "criterion": args.criterion
    })
    mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})

    # ✅ Save as MLflow model in the pipeline output folder
    os.makedirs(args.model_output, exist_ok=True)
    mlflow.sklearn.save_model(model_pipeline, args.model_output)

    print(f"✅ MLflow model saved to {args.model_output}")
    mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--criterion", type=str, default="mse")
    args = parser.parse_args()
    main(args)
