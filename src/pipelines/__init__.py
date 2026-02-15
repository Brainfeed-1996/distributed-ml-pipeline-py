from celery import Celery
from celery.schedules import crontab
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery app configuration
app = Celery(
    'training_pipeline',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1',
    include=['src.pipelines.training', 'src.pipelines.preprocessing', 'src.pipelines.evaluation']
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    worker_prefetch_multiplier=1,
    worker_concurrency=4,
)


class MLflowTracker:
    """MLflow experiment tracking wrapper."""
    
    def __init__(self, experiment_name: str, tracking_uri: str = "http://localhost:5000"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
    
    def start_run(self, run_name: Optional[str] = None):
        """Start an MLflow run."""
        self.run = mlflow.start_run(run_name=run_name, experiment_id=self.experiment_name)
        return self.run
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model, artifact_path: str = "model"):
        """Log model based on type."""
        model_type = type(model).__name__
        if hasattr(mlflow, model_type.lower()):
            getattr(mlflow, model_type.lower()).log_model(model, artifact_path)
        else:
            mlflow.sklearn.log_model(model, artifact_path)
    
    def end_run(self):
        """End the MLflow run."""
        mlflow.end_run()


@app.task(bind=True, max_retries=3, default_retry_delay=60)
def train_model(self, model_type: str, hyperparameters: Dict[str, Any], 
                experiment_name: str, data_path: Optional[str] = None):
    """
    Distributed model training task.
    
    Args:
        model_type: Type of model (xgboost, lightgbm, sklearn)
        hyperparameters: Model hyperparameters
        experiment_name: MLflow experiment name
        data_path: Path to training data
    
    Returns:
        dict: Training results
    """
    logger.info(f"Starting training for {model_type}")
    
    tracker = MLflowTracker(experiment_name)
    
    try:
        # Load data (mock for now)
        X, y = self.load_data(data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Track parameters
        tracker.start_run(run_name=f"{model_type}_training")
        tracker.log_params(hyperparameters)
        
        # Train model
        model = self._get_model(model_type, hyperparameters)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }
        
        tracker.log_metrics(metrics)
        tracker.log_model(model)
        tracker.end_run()
        
        logger.info(f"Training completed: {metrics}")
        
        return {
            "status": "success",
            "metrics": metrics,
            "model_type": model_type,
            "experiment_name": experiment_name,
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise self.retry(exc=e)

    def _get_model(self, model_type: str, hyperparameters: Dict[str, Any]):
        """Get model based on type."""
        if model_type == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(**hyperparameters)
        elif model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**hyperparameters)
        elif model_type == "sklearn":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**hyperparameters)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def load_data(self, data_path: Optional[str] = None):
        """Load training data (mock implementation)."""
        # In production, load from S3/HDFS/DB
        np.random.seed(42)
        n_samples = 10000
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        return X, y


@app.task
def preprocess_data(data_path: str, output_path: str, config: Dict[str, Any]):
    """
    Data preprocessing task.
    
    Args:
        data_path: Input data path
        output_path: Output path for processed data
        config: Preprocessing configuration
    
    Returns:
        dict: Processing results
    """
    logger.info(f"Preprocessing data from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Apply preprocessing
    if config.get("normalize"):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    if config.get("handle_missing"):
        df = df.fillna(df.median())
    
    # Save processed data
    df.to_parquet(output_path, index=False)
    
    return {
        "status": "success",
        "input_rows": len(df),
        "output_path": output_path,
    }


@app.task
def evaluate_model(experiment_name: str, model_stage: str = "Production"):
    """
    Model evaluation task.
    
    Args:
        experiment_name: MLflow experiment name
        model_stage: Model stage to evaluate
    
    Returns:
        dict: Evaluation results
    """
    logger.info(f"Evaluating model from {experiment_name}")
    
    # Load model from MLflow
    client = mlflow.tracking.MlflowClient()
    
    # Get latest model version
    versions = client.get_latest_versions(experiment_name, stages=[model_stage])
    if not versions:
        return {"status": "error", "message": f"No model in {model_stage} stage"}
    
    model_version = versions[0]
    model_uri = f"runs:/{model_version.run_id}/model"
    
    # Load model
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Evaluate (mock data)
    X_test, y_test = self.load_data(None)
    
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    
    logger.info(f"Evaluation results: {metrics}")
    
    return {
        "status": "success",
        "metrics": metrics,
        "model_version": model_version.version,
    }


# Celery Beat schedule for periodic tasks
app.conf.beat_schedule = {
    'retrain-model-daily': {
        'task': 'src.pipelines.training.train_model',
        'schedule': crontab(hour=2, minute=0),
        'kwargs': {
            'model_type': 'xgboost',
            'hyperparameters': {'n_estimators': 100, 'max_depth': 6},
            'experiment_name': 'production-v1',
        }
    },
}
