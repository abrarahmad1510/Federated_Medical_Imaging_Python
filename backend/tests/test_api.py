"""
API tests for Medical FL Platform
"""
import json
import uuid
import pytest
from datetime import datetime
from flask import url_for
from backend.app import create_app, db
from backend.app.models.user import User
from backend.app.models.experiment import Experiment, ExperimentStatus
@pytest.fixture
def app():
"""Create test application"""
app = create_app('testing')
with app.app_context():
db.create_all()
yield app
db.session.remove()
db.drop_all()
@pytest.fixture
def client(app):
"""Create test client"""
return app.test_client()
@pytest.fixture
def auth_headers(client):
"""Create authenticated user and return headers"""
# Create test user
user = User(
email='test@example.com',
password='testpassword123',
is_active=True
)
db.session.add(user)
db.session.commit()
# Login
response = client.post('/api/v1/auth/login', json={
'email': 'test@example.com',
'password': 'testpassword123'
})
assert response.status_code == 200
token = response.json['access_token']
return {
'Authorization': f'Bearer {token}',
'Content-Type': 'application/json'
}
class TestAuthAPI:
"""Test authentication endpoints"""
def test_register(self, client):
"""Test user registration"""
response = client.post('/api/v1/auth/register', json={
'email': 'newuser@example.com',
'password': 'Password123!',
'first_name': 'Test',
'last_name': 'User'
})
assert response.status_code == 201
assert 'email' in response.json
assert response.json['email'] == 'newuser@example.com'
def test_login(self, client):
"""Test user login"""
# First register
client.post('/api/v1/auth/register', json={
'email': 'login@example.com',
'password': 'Password123!'
})
# Then login
response = client.post('/api/v1/auth/login', json={
'email': 'login@example.com',
'password': 'Password123!'
})
assert response.status_code == 200
assert 'access_token' in response.json
assert 'refresh_token' in response.json
def test_invalid_login(self, client):
"""Test invalid login"""
response = client.post('/api/v1/auth/login', json={
'email': 'nonexistent@example.com',
'password': 'wrongpassword'
})
assert response.status_code == 401
assert 'message' in response.json
class TestExperimentAPI:
"""Test experiment endpoints"""
def test_create_experiment(self, client, auth_headers):
"""Test creating an experiment"""
response = client.post('/api/v1/experiments', json={
'name': 'Test Experiment',
'description': 'A test experiment',
'config': {
'dataset': 'brain_tumor',
'modality': 'MRI'
},
'model_config': {
'type': 'unet3d',
'features': [32, 64, 128]
},
'fl_config': {
'rounds': 10,
'epochs_per_round': 2
}
}, headers=auth_headers)
assert response.status_code == 201
assert response.json['name'] == 'Test Experiment'
assert response.json['status'] == 'draft'
def test_get_experiments(self, client, auth_headers):
"""Test getting experiments list"""
# Create test experiments
for i in range(3):
client.post('/api/v1/experiments', json={
'name': f'Test Experiment {i}',
'config': {}
}, headers=auth_headers)
response = client.get('/api/v1/experiments', headers=auth_headers)
assert response.status_code == 200
assert 'experiments' in response.json
assert len(response.json['experiments']) == 3
def test_get_experiment_detail(self, client, auth_headers):
"""Test getting experiment details"""
# Create experiment
create_resp = client.post('/api/v1/experiments', json={
'name': 'Detail Test',
'config': {}
}, headers=auth_headers)
exp_id = create_resp.json['id']
# Get details
response = client.get(f'/api/v1/experiments/{exp_id}', headers=auth_headers)
assert response.status_code == 200
assert response.json['id'] == exp_id
assert response.json['name'] == 'Detail Test'
def test_update_experiment(self, client, auth_headers):
"""Test updating an experiment"""
# Create experiment
create_resp = client.post('/api/v1/experiments', json={
'name': 'Update Test',
'description': 'Original description',
'config': {}
}, headers=auth_headers)
exp_id = create_resp.json['id']
# Update experiment
response = client.put(f'/api/v1/experiments/{exp_id}', json={
'name': 'Updated Name',
'description': 'Updated description',
'tags': ['test', 'updated']
}, headers=auth_headers)
assert response.status_code == 200
assert response.json['name'] == 'Updated Name'
assert response.json['description'] == 'Updated description'
assert 'test' in response.json['tags']
def test_delete_experiment(self, client, auth_headers):
"""Test deleting an experiment"""
# Create experiment
create_resp = client.post('/api/v1/experiments', json={
'name': 'Delete Test',
'config': {}
}, headers=auth_headers)
exp_id = create_resp.json['id']
# Delete experiment
response = client.delete(f'/api/v1/experiments/{exp_id}', headers=auth_headers
)
assert response.status_code == 200
# Verify it's deleted
get_resp = client.get(f'/api/v1/experiments/{exp_id}', headers=auth_headers)
assert get_resp.status_code == 404
class TestModelAPI:
"""Test model endpoints"""
def test_create_model(self, client, auth_headers):
"""Test creating a model"""
response = client.post('/api/v1/models', json={
'name': 'Test Model',
'model_type': 'unet_3d',
'architecture': {
'in_channels': 1,
'out_channels': 2,
'features': [32, 64, 128]
},
'description': 'A test model for brain tumor segmentation'
}, headers=auth_headers)
assert response.status_code == 201
assert response.json['name'] == 'Test Model'
assert response.json['model_type'] == 'unet_3d'
assert response.json['status'] == 'draft'
def test_get_models(self, client, auth_headers):
"""Test getting models list"""
# Create test models
for i in range(2):
client.post('/api/v1/models', json={
'name': f'Test Model {i}',
'model_type': 'unet_3d',
'architecture': {}
}, headers=auth_headers)
response = client.get('/api/v1/models', headers=auth_headers)
assert response.status_code == 200
assert 'models' in response.json
assert len(response.json['models']) == 2
class TestHealthCheck:
"""Test health check endpoints"""
def test_health_check(self, client):
"""Test health check endpoint"""
response = client.get('/health')
assert response.status_code == 200
assert response.json['status'] == 'healthy'
assert response.json['service'] == 'medical-fl-backend'
def test_api_root(self, client):
"""Test API root endpoint"""
response = client.get('/')
assert response.status_code == 200
assert 'name' in response.json
assert 'version' in response.json
assert 'endpoints' in response.json
class TestErrorHandling:
"""Test error handling"""
def test_404(self, client):
"""Test 404 error"""
response = client.get('/nonexistent')
assert response.status_code == 404
assert 'error' in response.json
assert response.json['error'] == 'Not Found'
def test_unauthorized(self, client):
"""Test unauthorized access"""
response = client.get('/api/v1/experiments')
assert response.status_code == 401
assert 'msg' in response.json
