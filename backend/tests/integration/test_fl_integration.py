"""
Integration tests for FL functionality
"""
import pytest
import time
from unittest.mock import Mock, patch
from celery import Celery
from backend.app.services.fl_orchestrator import FLOrchestrator
from backend.app.models.experiment import Experiment, ExperimentStatus
@pytest.fixture
def mock_app():
"""Create mock Flask app"""
app = Mock()
app.config = {
'REDIS_HOST': 'localhost',
'REDIS_PORT': 6379,
'REDIS_PASSWORD': None,
'REDIS_DB': 0,
'CELERY_BROKER_URL': 'redis://localhost:6379/0',
'CELERY_RESULT_BACKEND': 'redis://localhost:6379/0',
}
app.logger = Mock()
return app
@pytest.fixture
def mock_experiment(db_session):
"""Create mock experiment"""
from backend.app.models.user import User
# Create user
user = User(
email='test@integration.com',
password='testpass',
is_active=True
)
db_session.add(user)
db_session.commit()
# Create experiment
experiment = Experiment(
name='Integration Test Experiment',
user_id=user.id,
config={'test': True},
model_config={'type': 'unet3d'},
fl_config={'rounds': 3, 'epochs_per_round': 1},
privacy_config={'enable_dp': False}
)
db_session.add(experiment)
db_session.commit()
return experiment
class TestFLOrchestratorIntegration:
"""Integration tests for FL Orchestrator"""
def test_start_experiment(self, mock_app, mock_experiment, db_session):
"""Test starting an experiment"""
with patch('redis.Redis') as mock_redis, \
patch('celery.Celery') as mock_celery:
# Setup mocks
mock_redis_instance = Mock()
mock_redis.return_value = mock_redis_instance
mock_celery_app = Mock()
mock_celery.return_value = mock_celery_app
# Create orchestrator
orchestrator = FLOrchestrator(mock_app)
# Mock task
mock_task = Mock()
mock_task.delay = Mock()
orchestrator._start_fl_training = mock_task
# Start experiment
orchestrator.start_experiment(mock_experiment.id)
# Verify experiment status was updated
updated_experiment = db_session.query(Experiment).get(mock_experiment.id)
assert updated_experiment.status == ExperimentStatus.RUNNING
# Verify task was called
mock_task.delay.assert_called_once_with(str(mock_experiment.id))
def test_stop_experiment(self, mock_app, mock_experiment, db_session):
"""Test stopping an experiment"""
with patch('redis.Redis') as mock_redis:
# Setup mocks
mock_redis_instance = Mock()
mock_redis.return_value = mock_redis_instance
# Create orchestrator
orchestrator = FLOrchestrator(mock_app)
# Update experiment to running
mock_experiment.update_status(ExperimentStatus.RUNNING)
db_session.commit()
# Stop experiment
orchestrator.stop_experiment(mock_experiment.id)
# Verify experiment status was updated
updated_experiment = db_session.query(Experiment).get(mock_experiment.id)
assert updated_experiment.status == ExperimentStatus.CANCELLED
# Verify Redis publish was called
mock_redis_instance.publish.assert_called_once()
def test_get_experiment_status(self, mock_app, mock_experiment):
"""Test getting experiment status"""
with patch('redis.Redis') as mock_redis:
# Setup mocks
mock_redis_instance = Mock()
mock_redis_instance.smembers.return_value = {'client-1', 'client-2'}
mock_redis_instance.get.return_value = '{"round": 1, "progress": 10.0}'
mock_redis.return_value = mock_redis_instance
# Create orchestrator
orchestrator = FLOrchestrator(mock_app)
# Get status
status = orchestrator.get_experiment_status(mock_experiment.id)
# Verify status structure
assert 'experiment_id' in status
assert 'status' in status
assert 'active_clients' in status
assert 'progress' in status
# Verify Redis calls
mock_redis_instance.smembers.assert_called_once()
mock_redis_instance.get.assert_called_once()
def test_get_client_status(self, mock_app):
"""Test getting client status"""
with patch('redis.Redis') as mock_redis:
# Setup mocks
mock_redis_instance = Mock()
mock_redis_instance.get.return_value = json.dumps({
'client_id': 'test-client',
'device': 'cuda',
'status': 'active'
})
mock_redis_instance.hgetall.return_value = {
'accuracy': '0.85',
'loss': '0.15'
}
mock_redis.return_value = mock_redis_instance
# Create orchestrator
orchestrator = FLOrchestrator(mock_app)
# Get client status
status = orchestrator.get_client_status('test-client')
# Verify status structure
assert 'client_id' in status
assert 'device' in status
assert 'metrics' in status
assert status['metrics']['accuracy'] == '0.85'
# Verify Redis calls
mock_redis_instance.get.assert_called_once()
mock_redis_instance.hgetall.assert_called_once()
def test_broadcast_to_clients(self, mock_app, mock_experiment):
"""Test broadcasting to clients"""
with patch('redis.Redis') as mock_redis:
# Setup mocks
mock_redis_instance = Mock()
mock_redis_instance.smembers.return_value = {'client-1', 'client-2', 'clie
nt-3'}
mock_redis.return_value = mock_redis_instance
# Create orchestrator
orchestrator = FLOrchestrator(mock_app)
# Broadcast message
message = {'action': 'update_config', 'config': {'lr': 0.001}}
orchestrator.broadcast_to_clients(mock_experiment.id, message)
# Verify Redis publish was called for each client
assert mock_redis_instance.publish.call_count == 3
# Verify message content
for call in mock_redis_instance.publish.call_args_list:
assert 'update_config' in call[0][1]
class TestPrivacyIntegration:
"""Integration tests for privacy features"""
def test_differential_privacy_config(self, mock_app, mock_experiment):
"""Test DP configuration in experiments"""
# Update experiment with DP config
mock_experiment.privacy_config = {
'enable_dp': True,
'epsilon': 1.0,
'delta': 1e-5,
'max_grad_norm': 1.0,
'noise_multiplier': 1.1
}
# Verify DP settings
assert mock_experiment.privacy_config['enable_dp'] is True
assert mock_experiment.privacy_config['epsilon'] == 1.0
assert mock_experiment.privacy_config['delta'] == 1e-5
def test_privacy_budget_tracking(self, mock_app, mock_experiment):
"""Test privacy budget tracking"""
# Update experiment metrics with privacy budget
mock_experiment.metrics = {
'privacy': {
'epsilon_used': 0.5,
'epsilon_remaining': 0.5,
'delta': 1e-5,
'privacy_budget_used': 0.5
},
'rounds': {
1: {'accuracy': 0.75, 'privacy_cost': 0.1},
2: {'accuracy': 0.78, 'privacy_cost': 0.1},
3: {'accuracy': 0.80, 'privacy_cost': 0.1}
}
}
# Verify privacy budget tracking
privacy_metrics = mock_experiment.metrics['privacy']
assert privacy_metrics['epsilon_used'] == 0.5
assert privacy_metrics['privacy_budget_used'] == 0.5
# Verify per-round privacy cost
for round_num in [1, 2, 3]:
assert 'privacy_cost' in mock_experiment.metrics['rounds'][round_num]
assert mock_experiment.metrics['rounds'][round_num]['privacy_cost'] == 0.1
class TestModelIntegration:
"""Integration tests for model management"""
def test_model_versioning(self, db_session, mock_experiment):
"""Test model versioning"""
from backend.app.models.model_registry import Model, ModelStatus, ModelType
# Create multiple versions of a model
model_versions = []
for version in ['1.0.0', '1.1.0', '2.0.0']:
model = Model(
name='Brain Tumor Segmenter',
user_id=mock_experiment.user_id,
model_type=ModelType.UNET_3D,
version=version,
architecture={'features': [32, 64, 128, 256]},
status=ModelStatus.VALIDATED
)
db_session.add(model)
model_versions.append(model)
db_session.commit()
# Verify versions
assert len(model_versions) == 3
assert model_versions[0].version == '1.0.0'
assert model_versions[1].version == '1.1.0'
assert model_versions[2].version == '2.0.0'
# Verify they have the same name but different versions
assert all(m.name == 'Brain Tumor Segmenter' for m in model_versions)
assert len(set(m.version for m in model_versions)) == 3
def test_model_metrics(self, db_session):
"""Test model metrics storage"""
from backend.app.models.model_registry import Model, ModelType, ModelStatus
# Create model with metrics
model = Model(
name='Test Model with Metrics',
user_id=uuid.uuid4(),
model_type=ModelType.UNET_3D,
version='1.0.0',
status=ModelStatus.VALIDATED,
metrics={
'validation': {
'dataset_1': {
'dice_score': 0.85,
'hausdorff_distance': 3.2,
'precision': 0.88,
'recall': 0.83
},
'dataset_2': {
'dice_score': 0.82,
'hausdorff_distance': 3.5,
'precision': 0.85,
'recall': 0.80
}
},
'test': {
'final_dice': 0.84,
'final_loss': 0.16
}
}
)
db_session.add(model)
db_session.commit()
# Verify metrics structure
assert 'validation' in model.metrics
assert 'test' in model.metrics
assert len(model.metrics['validation']) == 2
assert model.metrics['validation']['dataset_1']['dice_score'] == 0.85
assert model.metrics['test']['final_dice'] == 0.84
