"""
Custom decorators for the application
"""
from functools import wraps
from typing import Callable, Any
from flask import request, jsonify, current_app
from flask_jwt_extended import get_jwt_identity, verify_jwt_in_request
import jwt
import uuid
from ..models.user import User
def validate_json(f: Callable) -> Callable:
"""Decorator to validate JSON request body"""
@wraps(f)
def decorated_function(*args, **kwargs):
if not request.is_json:
return jsonify({'message': 'Request must be JSON'}), 400
return f(*args, **kwargs)
return decorated_function
def admin_required(f: Callable) -> Callable:
"""Decorator to require admin privileges"""
@wraps(f)
def decorated_function(*args, **kwargs):
verify_jwt_in_request()
user_id = get_jwt_identity()
try:
user = User.query.get(uuid.UUID(user_id))
if not user or not user.is_admin:
return jsonify({'message': 'Admin access required'}), 403
except (ValueError, AttributeError):
return jsonify({'message': 'Invalid user'}), 401
return f(*args, **kwargs)
return decorated_function
def rate_limit(max_requests: int, time_window: int) -> Callable:
"""Decorator for rate limiting"""
def decorator(f: Callable) -> Callable:
@wraps(f)
def decorated_function(*args, **kwargs):
# This is a simplified version
# In production, use Redis for distributed rate limiting
client_ip = request.remote_addr
endpoint = request.endpoint
# Check rate limit (simplified - would use Redis in production)
# For now, just pass through
return f(*args, **kwargs)
return decorated_function
return decorator
def validate_experiment_ownership(f: Callable) -> Callable:
"""Decorator to validate experiment ownership"""
@wraps(f)
def decorated_function(*args, **kwargs):
verify_jwt_in_request()
user_id = get_jwt_identity()
experiment_id = kwargs.get('experiment_id')
if not experiment_id:
return jsonify({'message': 'Experiment ID required'}), 400
from ..models.experiment import Experiment
try:
experiment = Experiment.query.get(uuid.UUID(experiment_id))
if not experiment or str(experiment.user_id) != user_id:
return jsonify({'message': 'Experiment not found or access denied'}),
404
except ValueError:
return jsonify({'message': 'Invalid experiment ID'}), 400
return f(*args, **kwargs)
return decorated_function
def handle_exceptions(f: Callable) -> Callable:
"""Decorator to handle exceptions gracefully"""
@wraps(f)
def decorated_function(*args, **kwargs):
try:
return f(*args, **kwargs)
except jwt.ExpiredSignatureError:
return jsonify({'message': 'Token has expired'}), 401
except jwt.InvalidTokenError:
return jsonify({'message': 'Invalid token'}), 401
except ValueError as e:
return jsonify({'message': str(e)}), 400
except Exception as e:
current_app.logger.error(f'Unhandled exception: {str(e)}')
return jsonify({'message': 'Internal server error'}), 500
return decorated_function
def validate_model_type(f: Callable) -> Callable:
"""Decorator to validate model type"""
@wraps(f)
def decorated_function(*args, **kwargs):
from ..models.model_registry import ModelType
model_type = kwargs.get('model_type')
if model_type and model_type not in [t.value for t in ModelType]:
valid_types = [t.value for t in ModelType]
return jsonify({
'message': f'Invalid model type. Valid types: {valid_types}'
}), 400
return f(*args, **kwargs)
return decorated_function
