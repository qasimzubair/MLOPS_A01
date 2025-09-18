# MLOps Assignment 1: Machine Learning Pipeline with MLflow

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-orange)](https://mlflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green)](https://scikit-learn.org/)

A comprehensive MLOps pipeline implementing model training, experiment tracking, monitoring, and model registry using MLflow.

## 🚀 Project Overview

This project demonstrates a complete MLOps workflow for machine learning model development and deployment, featuring:

- **Model Training**: Multiple ML algorithms with performance comparison
- **Experiment Tracking**: Comprehensive MLflow integration for reproducibility
- **Model Registry**: Centralized model versioning and lifecycle management
- **Performance Monitoring**: Automated best model selection and production tagging

## 📂 Project Structure

```
mlops-assignment-1/
├── 📄 README.md                    # Project documentation
├── 📊 data/                        
│   └── iris.csv                   
├── 🤖 models/                      # Trained model artifacts
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── svm.joblib
├── 📓 notebooks/                   # Jupyter notebooks
│   └── model_training_comparison.ipynb
├── 🐍 src/                         # Source code
│   ├── mlflow_tracking.py         # MLflow experiment tracking
│   ├── model_monitoring.py        # Model performance monitoring
│   └── model_registry_status.py   # Registry status utilities
└── 📈 mlruns/                      # MLflow backend storage
```

## 🔧 Prerequisites

- Python 3.8 or higher
- pip package manager

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/qasimzubair/MLOPS_A01.git
   cd mlops-assignment-1
   ```

2. **Install dependencies**:
   ```bash
   pip install mlflow scikit-learn pandas matplotlib seaborn jupyter
   ```

## 🏃‍♂️ Quick Start

### 1. Train Models and Track Experiments
```bash
python src/mlflow_tracking.py
```
This will:
- Load/prepare the Iris dataset
- Train three ML models (Logistic Regression, Random Forest, SVM)
- Log experiments to MLflow with metrics and artifacts
- Register models in the MLflow Model Registry

### 2. Monitor Model Performance
```bash
python src/model_monitoring.py
```
This will:
- Analyze all experiment runs
- Identify the best performing model
- Tag the best model for production deployment

### 3. Check Registry Status
```bash
python src/model_registry_status.py
```
View registered models and their current status.

### 4. Launch MLflow UI
```bash
mlflow ui
```
Access the web interface at `http://localhost:5000` to:
- Compare experiment runs
- View model performance metrics
- Manage model registry
- Analyze artifacts and visualizations

## 🤖 Machine Learning Models

| Model | Algorithm | Accuracy | Status |
|-------|-----------|----------|--------|
| **Logistic Regression** | Linear Classification | 100% | Registered |
| **Random Forest** | Ensemble Method | 100% | Registered |
| **SVM** | Support Vector Machine | 100% | 🏆 **Production** |

> **Note**: All models achieve perfect accuracy on the Iris dataset due to its simplicity and clean separation of classes.

## 📊 MLflow Features Implemented

### Experiment Tracking
- ✅ Parameter logging (hyperparameters)
- ✅ Metric logging (accuracy, precision, recall, F1-score)
- ✅ Artifact logging (confusion matrices, model files)
- ✅ Custom tags and metadata

### Model Registry
- ✅ Model versioning and lifecycle management
- ✅ Production model tagging
- ✅ Model comparison and selection
- ✅ Centralized model storage

### Monitoring & Governance
- ✅ Performance tracking across experiments
- ✅ Automated best model identification
- ✅ Production readiness assessment

## 📈 Performance Metrics

All models are evaluated using standard classification metrics:

- **Accuracy**: Overall classification correctness
- **Precision**: True positive rate (macro-averaged)
- **Recall**: Sensitivity (macro-averaged)  
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed prediction breakdown

## 🔍 MLflow UI Navigation

1. **Experiments Tab**: View all training runs with sortable metrics
2. **Models Tab**: Browse registered models and versions
3. **Compare Runs**: Select multiple experiments for side-by-side comparison
4. **Artifacts**: Download confusion matrices and model files

## 🚀 Production Deployment

The best performing model (SVM) is automatically tagged for production with:
- `deployment_status: production`
- `model_performance: best_model`
- `accuracy: 1.0`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Qasim Zubair**
- GitHub: [@qasimzubair](https://github.com/qasimzubair)

## 🙏 Acknowledgments

- [MLflow](https://mlflow.org/) for experiment tracking and model management
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) for the classic ML dataset

