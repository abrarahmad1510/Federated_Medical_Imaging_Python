"""
Medical FL Platform - Backend Application
Main application factory and configuration
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Dict, Any
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis
from .config import config
# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()
limiter = Limiter(key_func=get_remote_address)
redis_client = None
def create_app(config_name: str = None) -> Flask:
"""
Application factory function
Args:
config_name: Configuration name (development, testing, production)
Returns:
Flask application instance
"""
if config_name is None:
config_name = os.getenv('FLASK_ENV', 'development')
app = Flask(__name__)
# Load configuration
app.config.from_object(config[config_name])
# Initialize extensions
initialize_extensions(app)
# Register blueprints
register_blueprints(app)
# Register error handlers
register_error_handlers(app)
# Register CLI commands
register_commands(app)
# Setup logging
setup_logging(app)
# Health check endpoint
@app.route('/health')
def health_check():
"""Health check endpoint for Kubernetes"""
return {'status': 'healthy', 'service': 'medical-fl-backend'}, 200
# Root endpoint
@app.route('/')
def index():
"""API root endpoint"""
return {
'name': 'Medical FL Platform API',
'version': '1.0.0',
'documentation': '/api/v1/docs',
'endpoints': {
'auth': '/api/v1/auth',
'experiments': '/api/v1/experiments',
'models': '/api/v1/models',
'clients': '/api/v1/clients'
}
}, 200
return app
def initialize_extensions(app: Flask) -> None:
"""Initialize all Flask extensions"""
# Enable CORS
CORS(app, resources={
r"/api/*": {
"origins": app.config.get('CORS_ORIGINS', ["http://localhost:3000"]),
"methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
"allow_headers": ["Content-Type", "Authorization"],
"supports_credentials": True
}
})
# Initialize SQLAlchemy
db.init_app(app)
# Initialize Flask-Migrate
migrate.init_app(app, db)
# Initialize JWT
jwt.init_app(app)
# Initialize rate limiter
limiter.init_app(app)
# Initialize Redis
global redis_client
redis_client = redis.Redis(
host=app.config['REDIS_HOST'],
port=app.config['REDIS_PORT'],
password=app.config.get('REDIS_PASSWORD'),
db=app.config.get('REDIS_DB', 0),
decode_responses=True,
socket_connect_timeout=5,
socket_timeout=5,
retry_on_timeout=True
)
# Test Redis connection
try:
redis_client.ping()
app.logger.info("Redis connection established")
except redis.ConnectionError:
app.logger.error("Failed to connect to Redis")
raise
def register_blueprints(app: Flask) -> None:
"""Register all API blueprints"""
from .api.endpoints import auth, experiments, models, clients, data
api_prefix = '/api/v1'
app.register_blueprint(auth.bp, url_prefix=f'{api_prefix}/auth')
app.register_blueprint(experiments.bp, url_prefix=f'{api_prefix}/experiments')
app.register_blueprint(models.bp, url_prefix=f'{api_prefix}/models')
app.register_blueprint(clients.bp, url_prefix=f'{api_prefix}/clients')
app.register_blueprint(data.bp, url_prefix=f'{api_prefix}/data')
def register_error_handlers(app: Flask) -> None:
"""Register global error handlers"""
@app.errorhandler(404)
def not_found_error(error):
return {
'error': 'Not Found',
'message': 'The requested resource was not found',
'status_code': 404
}, 404
@app.errorhandler(500)
def internal_error(error):
app.logger.error(f'Internal server error: {error}')
return {
'error': 'Internal Server Error',
'message': 'An unexpected error occurred',
'status_code': 500
}, 500
@app.errorhandler(429)
def ratelimit_handler(error):
return {
'error': 'Too Many Requests',
'message': 'Rate limit exceeded',
'status_code': 429
}, 429
def register_commands(app: Flask) -> None:
"""Register CLI commands"""
@app.cli.command('init-db')
def init_db_command():
"""Initialize the database"""
from .db import init_db
init_db()
print('Database initialized.')
@app.cli.command('create-admin')
def create_admin_command():
"""Create an admin user"""
from .models.user import User
from werkzeug.security import generate_password_hash
email = input("Enter admin email: ")
password = input("Enter admin password: ")
admin = User(
email=email,
password_hash=generate_password_hash(password),
is_admin=True,
is_active=True
)
db.session.add(admin)
db.session.commit()
print(f'Admin user {email} created.')
@app.cli.command('seed-data')
def seed_data_command():
"""Seed database with sample data"""
from .db import seed_database
seed_database()
print('Database seeded with sample data.')
def setup_logging(app: Flask) -> None:
"""Configure application logging"""
if not app.debug:
# Production logging
if not os.path.exists('logs'):
os.mkdir('logs')
file_handler = RotatingFileHandler(
'logs/backend.log',
maxBytes=10240,
backupCount=10
)
file_handler.setFormatter(logging.Formatter(
'%(asctime)s %(levelname)s: %(message)s '
'[in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Medical FL Platform startup')
else:
# Development logging
app.logger.setLevel(logging.DEBUG)
