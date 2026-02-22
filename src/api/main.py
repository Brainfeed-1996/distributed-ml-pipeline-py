from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
from src.models.inference import ModelRegistry, InferenceEngine
from src.features.store import FeatureStore
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="ML Prediction API",
    description="Production ML model serving API",
    version="1.0.0",
)


# Request/Response models
class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="Input features")
    model_version: Optional[str] = None


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    prediction_id: str


class BatchPredictRequest(BaseModel):
    features: List[List[float]]


class BatchPredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_store_connected: bool


# Dependencies
def get_model_registry():
    return ModelRegistry()


def get_inference_engine():
    return InferenceEngine()


@app.get("/health", response_model=HealthResponse)
async def health_check(
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if registry.is_loaded() else "degraded",
        model_loaded=registry.is_loaded(),
        feature_store_connected=True,  # Check actual connection in production
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    engine: InferenceEngine = Depends(get_inference_engine),
):
    """
    Make a single prediction.
    
    Example:
    ```bash
    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"features": [1.2, 3.4, 5.6]}'
    ```
    """
    try:
        result = engine.predict(
            features=np.array(request.features).reshape(1, -1),
            model_version=request.model_version,
        )
        
        return PredictResponse(
            prediction=int(result.prediction),
            probability=float(result.probability),
            model_version=result.model_version,
            prediction_id=result.prediction_id,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictResponse)
async def batch_predict(
    request: BatchPredictRequest,
    engine: InferenceEngine = Depends(get_inference_engine),
):
    """
    Make batch predictions.
    
    Example:
    ```bash
    curl -X POST http://localhost:8000/batch_predict \\
         -H "Content-Type: application/json" \\
         -d '{"features": [[1.2, 3.4], [5.6, 7.8]]}'
    ```
    """
    try:
        features = np.array(request.features)
        results = engine.predict_batch(features)
        
        return BatchPredictResponse(
            predictions=[int(p) for p in results.predictions],
            probabilities=[float(p) for p in results.probabilities],
            model_version=results.model_version,
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models(
    registry: ModelRegistry = Depends(get_model_registry),
):
    """List available models."""
    return registry.list_models()


@app.get("/models/{model_version}")
async def get_model_info(
    model_version: str,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Get model information."""
    info = registry.get_model_info(model_version)
    if not info:
        raise HTTPException(status_code=404, detail="Model not found")
    return info


@app.get("/features/{entity_id}")
async def get_features(
    entity_id: str,
    feature_set: str = "default",
):
    """Get online features for an entity."""
    features = FeatureStore.get_online(
        entity_ids=[entity_id],
        feature_set=feature_set,
    )
    return features.get(entity_id, {})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
