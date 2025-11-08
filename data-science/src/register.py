import argparse
import os
import mlflow


def main(args):
    mlflow.start_run()

    # Local path to the trained model produced by the previous step
    model_dir = args.model_path
    model_file = os.path.join(model_dir, "model.joblib")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"❌ Trained model not found at {model_file}")

    print(f"📦 Registering model from: {model_file}")

    # Log model to MLflow (creates version under the current experiment)
    mlflow.log_artifact(model_file, artifact_path="model")
    mlflow.sklearn.log_model(
        sk_model=None,
        artifact_path="model_ref",
        registered_model_name=args.registered_model_name
    )

    # Alternatively register directly to AzureML Model Registry
    print(f"✅ Registered model via MLflow: {args.registered_model_name}")
    mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register model in Azure ML")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model folder from previous step")
    parser.add_argument("--registered_model_name", type=str, required=True,
                        help="Name to register the model under in Azure ML")
    args = parser.parse_args()
    main(args)
