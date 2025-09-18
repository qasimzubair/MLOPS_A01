
import mlflow
import pandas as pd

# Search for all runs
runs = mlflow.search_runs(experiment_ids=["0"])

# Find best performing model
best_run = runs.loc[runs['metrics.accuracy'].idxmax()]
best_model_name = best_run['tags.mlflow.runName']
best_accuracy = best_run['metrics.accuracy']

print(f"Best model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")

# Tag best model for production
client = mlflow.tracking.MlflowClient()
model_name = best_model_name.replace(" ", "_").lower()

try:
    registered_models = client.search_registered_models(f"name='{model_name}'")
    if registered_models:
        model = registered_models[0]
        latest_version = model.latest_versions[0] if model.latest_versions else None
        
        if latest_version:
            # Tag as production model
            client.set_model_version_tag(
                name=model_name,
                version=latest_version.version,
                key="deployment_status",
                value="production"
            )
            client.set_model_version_tag(
                name=model_name,
                version=latest_version.version,
                key="model_performance", 
                value="best_model"
            )
            client.set_model_version_tag(
                name=model_name,
                version=latest_version.version,
                key="accuracy",
                value=str(best_accuracy)
            )
            print(f"Tagged {model_name} as production model")
            
except Exception as e:
    print(f"Error updating model tags: {e}")

print("Model monitoring complete.")