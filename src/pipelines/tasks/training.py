from src.pipelines.celery_app import app
from src.pipelines.tracker import MLflowTracker
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_mock_data():
    """Load training data (mock implementation)."""
    np.random.seed(42)
    n_samples = 10000
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def get_model(model_type: str, hyperparameters: Dict[str, Any]):
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

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def train_model(self, model_type: str, hyperparameters: Dict[str, Any], 
                experiment_name: str, data_path: Optional[str] = None):
    """
    Distributed model training task.
    """
    logger.info(f"Starting training for {model_type}")
    tracker = MLflowTracker(experiment_name)
    
    try:
        # Load data
        X, y = load_mock_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Track parameters
        tracker.start_run(run_name=f"{model_type}_training")
        tracker.log_params(hyperparameters)
        
        # Train model
        model = get_model(model_type, hyperparameters)
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
