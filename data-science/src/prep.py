import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow


def main(args):
    # Start an MLflow run for tracking
    mlflow.start_run()

    # Load raw data
    df = pd.read_csv(args.input_data)

    # Basic cleaning: drop rows with any missing values
    df = df.dropna()

    # Log basic dataset info
    mlflow.log_metric("rows_after_cleaning", len(df))
    mlflow.log_metric("columns", df.shape[1])

    # Train-test split (80/20)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Ensure output directories exist
    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)

    train_path = os.path.join(args.train_output, "train.csv")
    test_path = os.path.join(args.test_output, "test.csv")

    # Save split datasets
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Log split sizes
    mlflow.log_metric("train_rows", len(train_df))
    mlflow.log_metric("test_rows", len(test_df))

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--train_output", type=str, required=True)
    parser.add_argument("--test_output", type=str, required=True)
    args = parser.parse_args()
    main(args)

