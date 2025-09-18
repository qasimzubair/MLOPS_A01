

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib


data_path = '../data/iris.csv'
if not os.path.exists(data_path):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = y
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)
else:
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load models
models = {}
model_files = {
    'Logistic Regression': '../models/logistic_regression.joblib',
    'Random Forest': '../models/random_forest.joblib',
    'SVM': '../models/svm.joblib'
}

for name, file_path in model_files.items():
    if os.path.exists(file_path):
        models[name] = joblib.load(file_path)
    else:
        if name == 'Logistic Regression':
            model = LogisticRegression(max_iter=200)
        elif name == 'Random Forest':
            model = RandomForestClassifier()
        elif name == 'SVM':
            model = SVC()
        
        model.fit(X_train, y_train)
        models[name] = model
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(model, file_path)

# MLflow tracking
for name, model in models.items():
    with mlflow.start_run(run_name=f"{name}_pretrained"):
        y_pred = model.predict(X_test)
        
        # Log parameters
        if hasattr(model, 'get_params'):
            mlflow.log_params(model.get_params())
        
        # Log metrics
        mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
        mlflow.log_metric('precision', precision_score(y_test, y_pred, average='macro'))
        mlflow.log_metric('recall', recall_score(y_test, y_pred, average='macro'))
        mlflow.log_metric('f1_score', f1_score(y_test, y_pred, average='macro'))
        
        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} - Confusion Matrix')
        plot_path = f'cm_{name.replace(" ", "_").lower()}.png'
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path, "plots")
        plt.close()
        os.remove(plot_path)
        
        # Register model
        model_name = f"{name.replace(' ', '_').lower()}_pretrained"
        mlflow.sklearn.log_model(
            model, 
            model_name, 
            registered_model_name=model_name
        )

print("MLflow tracking complete.")
