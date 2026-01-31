"""
Configuration management for the Medical FL Platform
"""
import os
from datetime import timedelta
from typing import Dict, Any, List
from dotenv import load_dotenv
load_dotenv()
class Config:
"""Base configuration"""
# Application
APP_NAME = os.getenv('APP_NAME', 'Medical FL Platform')
SECRET_KEY = os.getenv('APP_SECRET_KEY', 'dev-secret-key-change-in-production')
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
# Security
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', SECRET_KEY)
JWT_ACCESS_TOKEN_EXPIRES = timedelta(
seconds=int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES', 900))
)
JWT_REFRESH_TOKEN_EXPIRES = timedelta(
seconds=int(os.getenv('JWT_REFRESH_TOKEN_EXPIRES', 86400))
)
JWT_TOKEN_LOCATION = ['headers']
JWT_HEADER_NAME = 'Authorization'
JWT_HEADER_TYPE = 'Bearer'
# CORS
CORS_ORIGINS = eval(os.getenv('CORS_ORIGINS', '["http://localhost:3000"]'))
# Rate limiting
RATELIMIT_ENABLED = True
RATELIMIT_STORAGE_URI = f"redis://{os.getenv('REDIS_HOST', 'redis')}:{os.getenv('R
EDIS_PORT', '6379')}"
RATELIMIT_STRATEGY = 'fixed-window'
RATELIMIT_DEFAULT = '200 per minute'
# SQLAlchemy
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ENGINE_OPTIONS = {
'pool_size': int(os.getenv('DB_POOL_SIZE', 20)),
'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', 40)),
'pool_pre_ping': True,
'pool_recycle': 300,
}
# Redis
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
REDIS_DB = int(os.getenv('REDIS_DB', 0))
# Celery
CELERY_BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
CELERY_RESULT_BACKEND = CELERY_BROKER_URL
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TIMEZONE = 'UTC'
# Federated Learning
FLOWER_SERVER_HOST = os.getenv('FLOWER_SERVER_HOST', '0.0.0.0')
FLOWER_SERVER_PORT = int(os.getenv('FLOWER_SERVER_PORT', 8080))
FLOWER_SSL_ENABLED = os.getenv('FLOWER_SSL_ENABLED', 'false').lower() == 'true'
# Privacy
DP_EPSILON = float(os.getenv('DP_EPSILON', 1.0))
DP_DELTA = float(os.getenv('DP_DELTA', 1e-5))
DP_MAX_GRAD_NORM = float(os.getenv('DP_MAX_GRAD_NORM', 1.0))
# MLflow
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
# File upload
MAX_CONTENT_LENGTH = 100 * 1024 * 1024 # 100MB
ALLOWED_EXTENSIONS = {'dcm', 'nii', 'nii.gz', 'png', 'jpg', 'jpeg'}
# Email
MAIL_SERVER = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
MAIL_PORT = int(os.getenv('MAIL_PORT', 587))
MAIL_USE_TLS = os.getenv('MAIL_USE_TLS', 'true').lower() == 'true'
MAIL_USERNAME = os.getenv('MAIL_USERNAME')
MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = os.getenv('LOG_FORMAT', 'json')
class DevelopmentConfig(Config):
"""Development configuration"""
DEBUG = True
TESTING = False
# Database
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'medical_fl_dev')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
SQLALCHEMY_DATABASE_URI = (
f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
# Development-specific settings
PROPAGATE_EXCEPTIONS = True
PRESERVE_CONTEXT_ON_EXCEPTION = False
# Disable rate limiting in development
RATELIMIT_ENABLED = False
class TestingConfig(Config):
"""Testing configuration"""
DEBUG = False
TESTING = True
# Use SQLite for testing
SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
# Disable rate limiting in tests
RATELIMIT_ENABLED = False
# Use fake email backend
MAIL_SUPPRESS_SEND = True
# Faster JWT tokens for tests
JWT_ACCESS_TOKEN_EXPIRES = timedelta(seconds=30)
JWT_REFRESH_TOKEN_EXPIRES = timedelta(seconds=60)
class ProductionConfig(Config):
"""Production configuration"""
DEBUG = False
TESTING = False
# Database
DB_HOST = os.getenv('DB_HOST', 'postgres')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'medical_fl')
DB_USER = os.getenv('DB_USER', 'fl_admin')
DB_PASSWORD = os.getenv('DB_PASSWORD')
SQLALCHEMY_DATABASE_URI = (
f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
# Security
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
# Rate limiting
RATELIMIT_DEFAULT = '100 per minute'
RATELIMIT_APPLICATION = '1000 per hour'
# Production logging
LOG_LEVEL = 'WARNING'
# Enable SSL
FLOWER_SSL_ENABLED = True
# Configuration dictionary
config: Dict[str, Any] = {
'development': DevelopmentConfig,
'testing': TestingConfig,
'production': ProductionConfig,
'default': DevelopmentConfig
}
