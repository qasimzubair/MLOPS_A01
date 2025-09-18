
import mlflow
import pandas as pd

client = mlflow.tracking.MlflowClient()

print("=== Model Registry Status ===")
registered_models = client.search_registered_models()
print(f"Registered models: {len(registered_models)}")

for model in registered_models:
    print(f"- {model.name}")
    for version in model.latest_versions:
        print(f"  Version {version.version}: {version.current_stage}")

# Performance summary
runs = mlflow.search_runs(experiment_ids=["0"])
best_run = runs.loc[runs['metrics.accuracy'].idxmax()]
print(f"\nBest model: {best_run['tags.mlflow.runName']}")
print(f"Accuracy: {best_run['metrics.accuracy']:.4f}")

print("Registry check complete.")