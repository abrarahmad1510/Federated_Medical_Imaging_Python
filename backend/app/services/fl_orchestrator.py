"""
Federated Learning Orchestrator Service
"""
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from threading import Thread
import uuid
from celery import Celery
import redis
from flask import current_app
from ..models.experiment import Experiment, ExperimentStatus
from ..models.model_registry import Model, ModelStatus
from .. import db
class FLOrchestrator:
"""Orchestrates federated learning experiments"""
def __init__(self, app=None):
self.app = app or current_app
self.logger = logging.getLogger(__name__)
# Initialize Redis client
self.redis_client = redis.Redis(
host=self.app.config['REDIS_HOST'],
port=self.app.config['REDIS_PORT'],
password=self.app.config.get('REDIS_PASSWORD'),
db=self.app.config.get('REDIS_DB', 0),
decode_responses=True
)
# Initialize Celery
self.celery_app = Celery(
'medical_fl',
broker=self.app.config['CELERY_BROKER_URL'],
backend=self.app.config['CELERY_RESULT_BACKEND']
)
# Configure Celery
self.celery_app.conf.update(
task_serializer='json',
result_serializer='json',
accept_content=['json'],
task_track_started=True,
task_time_limit=30 * 60, # 30 minutes
task_soft_time_limit=25 * 60, # 25 minutes
worker_prefetch_multiplier=1,
worker_max_tasks_per_child=100
)
def start_experiment(self, experiment_id: uuid.UUID) -> None:
"""Start a federated learning experiment"""
try:
# Get experiment from database
with self.app.app_context():
experiment = Experiment.query.get(experiment_id)
if not experiment:
self.logger.error(f'Experiment {experiment_id} not found')
return
# Update experiment status
experiment.update_status(ExperimentStatus.RUNNING)
db.session.commit()
# Create model if not exists
if not experiment.model_id:
model = self._create_model_for_experiment(experiment)
experiment.model_id = model.id
db.session.commit()
# Start FL training task asynchronously
self._start_fl_training.delay(str(experiment_id))
self.logger.info(f'Started FL experiment {experiment_id}')
except Exception as e:
self.logger.error(f'Failed to start experiment {experiment_id}: {str(e)}')
# Update status to failed
with self.app.app_context():
experiment = Experiment.query.get(experiment_id)
if experiment:
experiment.update_status(ExperimentStatus.FAILED)
db.session.commit()
def stop_experiment(self, experiment_id: uuid.UUID) -> None:
"""Stop a running experiment"""
try:
with self.app.app_context():
experiment = Experiment.query.get(experiment_id)
if not experiment:
self.logger.error(f'Experiment {experiment_id} not found')
return
# Update status
experiment.update_status(ExperimentStatus.CANCELLED)
db.session.commit()
# Send stop signal via Redis
self.redis_client.publish(
f'experiment:{experiment_id}:control',
json.dumps({'action': 'stop'})
)
self.logger.info(f'Stopped FL experiment {experiment_id}')
except Exception as e:
self.logger.error(f'Failed to stop experiment {experiment_id}: {str(e)}')
def _create_model_for_experiment(self, experiment: Experiment) -> Model:
"""Create a model for the experiment"""
from ..models.model_registry import Model, ModelType
# Extract model configuration
model_config = experiment.model_config
# Create model
model = Model(
name=f"{experiment.name}_model",
user_id=experiment.user_id,
model_type=ModelType(model_config.get('type', 'unet_3d')),
description=f"Model for experiment: {experiment.name}",
architecture=model_config.get('architecture', {}),
hyperparameters=model_config.get('hyperparameters', {}),
tags=experiment.tags,
metadata={
'experiment_id': str(experiment.id),
'created_from_experiment': True
}
)
db.session.add(model)
db.session.commit()
return model
def get_experiment_status(self, experiment_id: uuid.UUID) -> Dict[str, Any]:
"""Get current status of an experiment"""
try:
with self.app.app_context():
experiment = Experiment.query.get(experiment_id)
if not experiment:
return {'error': 'Experiment not found'}
status = {
'experiment_id': str(experiment_id),
'status': experiment.status.value,
'metrics': experiment.metrics or {},
'started_at': experiment.started_at.isoformat() if experiment.star
ted_at else None,
'updated_at': experiment.updated_at.isoformat() if experiment.upda
ted_at else None
}
# Get active clients from Redis
active_clients = self.redis_client.smembers(f'experiment:{experiment_i
d}:clients')
status['active_clients'] = list(active_clients)
status['active_clients_count'] = len(active_clients)
# Get training progress
progress = self.redis_client.get(f'experiment:{experiment_id}:progress
')
if progress:
status['progress'] = json.loads(progress)
return status
except Exception as e:
self.logger.error(f'Failed to get experiment status {experiment_id}: {str(
e)}')
return {'error': str(e)}
def get_client_status(self, client_id: str) -> Dict[str, Any]:
"""Get status of a specific client"""
try:
# Get client info from Redis
client_key = f'client:{client_id}:info'
client_info = self.redis_client.get(client_key)
if not client_info:
return {'error': 'Client not found'}
info = json.loads(client_info)
# Get current experiment if any
experiment_id = self.redis_client.get(f'client:{client_id}:experiment')
if experiment_id:
info['current_experiment'] = experiment_id
# Get performance metrics
metrics_key = f'client:{client_id}:metrics'
metrics = self.redis_client.hgetall(metrics_key)
info['metrics'] = metrics
return info
except Exception as e:
self.logger.error(f'Failed to get client status {client_id}: {str(e)}')
return {'error': str(e)}
def broadcast_to_clients(self, experiment_id: uuid.UUID, message: Dict[str, Any])
-> None:
"""Broadcast message to all clients in an experiment"""
try:
# Get all clients for this experiment
client_ids = self.redis_client.smembers(f'experiment:{experiment_id}:clien
ts')
for client_id in client_ids:
channel = f'client:{client_id}:commands'
self.redis_client.publish(channel, json.dumps(message))
self.logger.info(f'Broadcasted message to {len(client_ids)} clients for ex
periment {experiment_id}')
except Exception as e:
self.logger.error(f'Failed to broadcast to clients: {str(e)}')
@property
def _start_fl_training(self):
"""Celery task to start FL training"""
@self.celery_app.task(bind=True, name='fl_orchestrator.start_fl_training')
def start_fl_training_task(self_task, experiment_id_str: str):
"""Start FL training for an experiment"""
experiment_id = uuid.UUID(experiment_id_str)
try:
with self.app.app_context():
# Get experiment
experiment = Experiment.query.get(experiment_id)
if not experiment:
self.logger.error(f'Experiment {experiment_id} not found')
return
# Update task ID
experiment.metadata['celery_task_id'] = self_task.request.id
db.session.commit()
# Start FL training
self._run_fl_training(experiment)
except Exception as e:
self.logger.error(f'FL training task failed for {experiment_id}: {str(
e)}')
# Update experiment status
with self.app.app_context():
experiment = Experiment.query.get(experiment_id)
if experiment:
experiment.update_status(ExperimentStatus.FAILED)
experiment.metrics['error'] = str(e)
db.session.commit()
return start_fl_training_task
def _run_fl_training(self, experiment: Experiment) -> None:
"""Run FL training (to be implemented with Flower)"""
# This is where Flower server would be started
# For now, we'll simulate the process
self.logger.info(f'Starting FL training for experiment {experiment.id}')
# Simulate training rounds
for round_num in range(1, experiment.fl_config.get('rounds', 10) + 1):
# Check if experiment was stopped
stop_signal = self.redis_client.get(f'experiment:{experiment.id}:stop')
if stop_signal:
self.logger.info(f'Experiment {experiment.id} stopped by user')
break
# Simulate round
self._simulate_training_round(experiment, round_num)
# Update progress in Redis
progress = {
'round': round_num,
'total_rounds': experiment.fl_config.get('rounds', 10),
'progress': (round_num / experiment.fl_config.get('rounds', 10)) * 100
,
'timestamp': datetime.utcnow().isoformat()
}
self.redis_client.setex(
f'experiment:{experiment.id}:progress',
300, # 5 minutes TTL
json.dumps(progress)
)
# Mark experiment as completed
with self.app.app_context():
experiment.update_status(ExperimentStatus.COMPLETED)
experiment.results = {
'completed_at': datetime.utcnow().isoformat(),
'total_rounds': experiment.fl_config.get('rounds', 10),
'final_metrics': experiment.metrics.get('rounds', {}).get('final', {})
}
db.session.commit()
self.logger.info(f'Completed FL training for experiment {experiment.id}')
def _simulate_training_round(self, experiment: Experiment, round_num: int) -> None
:
"""Simulate a training round (to be replaced with actual Flower)"""
import time
import random
# Simulate training time
time.sleep(2)
# Simulate metrics
accuracy = 0.5 + (round_num * 0.05) + random.uniform(-0.02, 0.02)
loss = 0.8 - (round_num * 0.07) + random.uniform(-0.03, 0.03)
# Update metrics in database
with self.app.app_context():
experiment.add_metric(round_num, 'accuracy', min(accuracy, 0.95))
experiment.add_metric(round_num, 'loss', max(loss, 0.1))
db.session.commit()
# Update metrics in Redis for real-time monitoring
metrics = {
'round': round_num,
'accuracy': accuracy,
'loss': loss,
'timestamp': datetime.utcnow().isoformat()
}
self.redis_client.publish(
f'experiment:{experiment.id}:metrics',
json.dumps(metrics)
)
self.logger.debug(f'Round {round_num} completed for experiment {experiment.id}
')
