import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.sklearn
from typing import Dict, Any, Optional

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
        model_name = type(model).__name__
        if "XGB" in model_name:
            mlflow.xgboost.log_model(model, artifact_path)
        elif "LGBM" in model_name:
            mlflow.lightgbm.log_model(model, artifact_path)
        else:
            mlflow.sklearn.log_model(model, artifact_path)
    
    def end_run(self):
        """End the MLflow run."""
        mlflow.end_run()
