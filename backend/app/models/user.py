"""
User model for authentication and authorization
"""
from datetime import datetime
from typing import Optional
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.dialects.postgresql import UUID
import uuid
from .. import db
class User(UserMixin, db.Model):
"""User model for authentication"""
__tablename__ = 'users'
# Primary key
id = db.Column(
UUID(as_uuid=True),
primary_key=True,
default=uuid.uuid4,
nullable=False
)
# User information
email = db.Column(db.String(255), unique=True, nullable=False, index=True)
username = db.Column(db.String(100), unique=True, nullable=True)
first_name = db.Column(db.String(100))
last_name = db.Column(db.String(100))
# Authentication
password_hash = db.Column(db.String(255), nullable=False)
# Timestamps
created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
updated_at = db.Column(
db.DateTime,
default=datetime.utcnow,
onupdate=datetime.utcnow,
nullable=False
)
last_login = db.Column(db.DateTime)
# Status flags
is_active = db.Column(db.Boolean, default=True, nullable=False)
is_admin = db.Column(db.Boolean, default=False, nullable=False)
email_verified = db.Column(db.Boolean, default=False, nullable=False)
# Relationships
experiments = db.relationship('Experiment', back_populates='user', lazy='dynamic')
api_keys = db.relationship('ApiKey', back_populates='user', lazy='dynamic')
def __init__(self, email: str, password: str, **kwargs):
"""Initialize a new user"""
self.email = email
self.set_password(password)
for key, value in kwargs.items():
setattr(self, key, value)
def set_password(self, password: str) -> None:
"""Set password hash"""
self.password_hash = generate_password_hash(password)
def check_password(self, password: str) -> bool:
"""Check password against hash"""
return check_password_hash(self.password_hash, password)
def to_dict(self) -> dict:
"""Convert user to dictionary"""
return {
'id': str(self.id),
'email': self.email,
'username': self.username,
'first_name': self.first_name,
'last_name': self.last_name,
'is_active': self.is_active,
'is_admin': self.is_admin,
'email_verified': self.email_verified,
'created_at': self.created_at.isoformat() if self.created_at else None,
'last_login': self.last_login.isoformat() if self.last_login else None
}
def __repr__(self) -> str:
return f'<User {self.email}>'
class ApiKey(db.Model):
"""API Key model for programmatic access"""
__tablename__ = 'api_keys'
# Primary key
id = db.Column(
UUID(as_uuid=True),
primary_key=True,
default=uuid.uuid4,
nullable=False
)
# Foreign key
user_id = db.Column(
UUID(as_uuid=True),
db.ForeignKey('users.id'),
nullable=False,
index=True
)
# API Key information
name = db.Column(db.String(100), nullable=False)
key_hash = db.Column(db.String(255), nullable=False, unique=True, index=True)
last_used = db.Column(db.DateTime)
# Scope and permissions
scopes = db.Column(db.JSON, default=list) # List of allowed scopes
rate_limit = db.Column(db.Integer, default=100) # Requests per minute
# Status
is_active = db.Column(db.Boolean, default=True, nullable=False)
expires_at = db.Column(db.DateTime)
# Timestamps
created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
# Relationships
user = db.relationship('User', back_populates='api_keys')
def __repr__(self) -> str:
return f'<ApiKey {self.name} for user {self.user_id}>'
