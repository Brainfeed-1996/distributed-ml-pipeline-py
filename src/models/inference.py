from typing import Dict, Any, Optional, List
import numpy as np
import mlflow.pyfunc
import pickle
import hashlib
import uuid
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PredictionResult:
    prediction: np.ndarray
    probability: np.ndarray
    model_version: str
    prediction_id: str
    timestamp: float


class ModelRegistry:
    """Registry for managing ML models."""
    
    def __init__(self, mlflow_uri: str = "http://localhost:5000"):
        self.mlflow_uri = mlflow_uri
        self._loaded_models: Dict[str, mlflow.pyfunc.PyFuncModel] = {}
    
    def load_model(self, model_version: str) -> mlflow.pyfunc.PyFuncModel:
        """Load a model from MLflow."""
        if model_version in self._loaded_models:
            return self._loaded_models[model_version]
        
        model = mlflow.pyfunc.load_model(
            f"models:/{model_version}/Production"
        )
        self._loaded_models[model_version] = model
        return model
    
    def is_loaded(self, model_version: str = "latest") -> bool:
        """Check if a model is loaded."""
        return model_version in self._loaded_models
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        models = []
        for rm in client.search_registered_models():
            for version in rm.latest_versions:
                models.append({
                    "name": rm.name,
                    "version": version.version,
                    "stage": version.current_stage,
                    "creation_timestamp": version.creation_timestamp,
                })
        return models
    
    def get_model_info(self, model_version: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        try:
            versions = client.get_latest_versions(
                "production-pipeline", stages=["Production"]
            )
            for v in versions:
                if v.version == model_version:
                    return {
                        "version": v.version,
                        "stage": v.current_stage,
                        "run_id": v.run_id,
                        "creation_timestamp": v.creation_timestamp,
                    }
        except Exception as e:
            print(f"Error getting model info: {e}")
        return None


class InferenceEngine:
    """High-performance inference engine."""
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
        self.default_model = "latest"
    
    def predict(
        self,
        features: np.ndarray,
        model_version: Optional[str] = None,
    ) -> PredictionResult:
        """
        Make a single prediction.
        
        Args:
            features: Input features (shape: [1, n_features])
            model_version: Specific model version to use
        
        Returns:
            PredictionResult with prediction, probability, etc.
        """
        version = model_version or self.default_model
        
        model = self.registry.load_model(version)
        
        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        # Generate prediction ID
        prediction_id = self._generate_prediction_id(features)
        
        return PredictionResult(
            prediction=prediction,
            probability=probability[0] if len(probability.shape) > 1 else probability,
            model_version=version,
            prediction_id=prediction_id,
            timestamp=__import__('time').time(),
        )
    
    def predict_batch(
        self,
        features: np.ndarray,
        model_version: Optional[str] = None,
    ) -> PredictionResult:
        """
        Make batch predictions.
        
        Args:
            features: Input features (shape: [batch_size, n_features])
            model_version: Specific model version to use
        
        Returns:
            PredictionResult with batch predictions
        """
        version = model_version or self.default_model
        
        model = self.registry.load_model(version)
        
        # Make batch prediction
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        return PredictionResult(
            prediction=predictions,
            probability=probabilities,
            model_version=version,
            prediction_id=self._generate_prediction_id(features),
            timestamp=__import__('time').time(),
        )
    
    def _generate_prediction_id(self, features: np.ndarray) -> str:
        """Generate unique prediction ID."""
        content = features.tobytes() + str(__import__('time').time())
        return hashlib.md5(content.encode()).hexdigest()[:8]


# Streaming inference for high-throughput scenarios
class StreamingInferenceEngine(InferenceEngine):
    """Engine for streaming predictions."""
    
    def __init__(self, registry: Optional[ModelRegistry] = None, batch_size: int = 100):
        super().__init__(registry)
        self.batch_size = batch_size
        self._buffer: list = []
    
    def stream_predict(self, feature: List[float]) -> Optional[PredictionResult]:
        """
        Add feature to buffer and return prediction when batch is full.
        
        Args:
            feature: Single feature vector
        
        Returns:
            PredictionResult if batch is complete, None otherwise
        """
        self._buffer.append(np.array(feature))
        
        if len(self._buffer) >= self.batch_size:
            batch = np.array(self._buffer)
            self._buffer = []  # Clear buffer
            return self.predict_batch(batch)
        return None
