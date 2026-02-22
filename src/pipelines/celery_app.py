from celery import Celery
from celery.schedules import crontab
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery app configuration
app = Celery(
    'training_pipeline',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1',
    include=['src.pipelines.tasks.training', 'src.pipelines.tasks.preprocessing', 'src.pipelines.tasks.evaluation']
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

# Celery Beat schedule for periodic tasks
app.conf.beat_schedule = {
    'retrain-model-daily': {
        'task': 'src.pipelines.tasks.training.train_model',
        'schedule': crontab(hour=2, minute=0),
        'kwargs': {
            'model_type': 'xgboost',
            'hyperparameters': {'n_estimators': 100, 'max_depth': 6},
            'experiment_name': 'production-v1',
        }
    },
}
