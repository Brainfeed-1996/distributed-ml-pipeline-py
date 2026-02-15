# Distributed ML Pipeline

A production-ready distributed machine learning pipeline built with Python, Celery, Redis, and MLflow.

## 🚀 Features

- **Distributed Task Queue**: Celery workers for parallel ML tasks
- **MLflow Integration**: Experiment tracking and model registry
- **Feature Store**: Offline and online feature serving
- **Model Serving**: REST API with FastAPI
- **Data Pipeline**: Apache Airflow DAGs for ETL
- **Monitoring**: Prometheus metrics and Grafana dashboards

## 📦 Installation

```bash
git clone https://github.com/Brainfeed-1996/distributed-ml-pipeline-py.git
cd distributed-ml-pipeline-py

# Install dependencies
pip install -r requirements.txt

# Start services
docker-compose up -d
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Distributed ML Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │   Airflow   │───▶│  Celery      │───▶│   MLflow          │  │
│  │   DAGs      │    │  Workers      │    │   Server          │  │
│  └─────────────┘    └──────────────┘    └────────────────────┘  │
│         │                   │                    │               │
│         ▼                   ▼                    ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Redis Broker                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Feature Store                          │   │
│  │  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │   │
│  │  │   Offline   │    │   Online     │    │  Redis     │  │   │
│  │  │   (S3)      │    │   (API)      │    │  Cache     │  │   │
│  │  └─────────────┘    └──────────────┘    └────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   FastAPI Server                        │   │
│  │  - Model inference endpoint                             │   │
│  │  - Feature serving                                      │   │
│  │  - Health checks                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
distributed-ml-pipeline-py/
├── src/
│   ├── pipelines/           # Celery tasks
│   │   ├── training.py      # Model training
│   │   ├── preprocessing.py # Data preprocessing
│   │   └── evaluation.py    # Model evaluation
│   ├── features/           # Feature engineering
│   │   ├── definitions.py   # Feature definitions
│   │   └── store.py         # Feature store client
│   ├── models/              # Model registry
│   │   ├── training.py       # Training logic
│   │   └── inference.py     # Inference utilities
│   └── api/                  # FastAPI endpoints
├── dags/                    # Airflow DAGs
├── notebooks/               # Jupyter notebooks
├── tests/                   # Unit tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🔧 Usage

### Start the Pipeline

```bash
# Start Redis and MLflow
docker-compose up -d

# Start Celery workers
celery -A src.pipelines worker --loglevel=info

# Start FastAPI server
uvicorn src.api.main:app --reload
```

### Run Training Pipeline

```python
from src.pipelines.training import train_model

# Trigger distributed training
result = train_model.delay(
    model_type="xgboost",
    hyperparameters={"n_estimators": 100, "max_depth": 6},
    experiment_name="production-v1"
)
```

### Track Experiments

```python
import mlflow
import mlflow.xgboost

with mlflow.start_run(experiment_id="production-v1"):
    mlflow.log_params(hyperparameters)
    mlflow.log_metrics(metrics)
    mlflow.xgboost.log_model(model, "model")
```

## 📊 Features

### Feature Store

```python
from src.features.store import FeatureStore

# Get features for inference
features = FeatureStore.get_online(
    entity_ids=["user_123", "user_456"],
    feature_set="user_features"
)
```

### Model Serving

```bash
# Make predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.2, 3.4, 5.6]}'
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run integration tests
pytest tests/integration/ -v
```

## 📝 License

MIT License
