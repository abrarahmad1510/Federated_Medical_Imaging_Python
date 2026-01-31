"""
Model registry for versioning and tracking ML models
"""
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from .. import db
class ModelType(Enum):
"""Model type enumeration"""
UNET_2D = 'unet_2d'
UNET_3D = 'unet_3d'
DEEPLAB = 'deeplab'
NNU_NET = 'nnu_net'
CUSTOM = 'custom'
class ModelStatus(Enum):
"""Model status enumeration"""
DRAFT = 'draft'
TRAINING = 'training'
VALIDATED = 'validated'
DEPLOYED = 'deployed'
ARCHIVED = 'archived'
FAILED = 'failed'
class Model(db.Model):
"""Model registry for ML models"""
__tablename__ = 'models'
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
# Basic information
name = db.Column(db.String(255), nullable=False, index=True)
version = db.Column(db.String(50), nullable=False, default='1.0.0')
description = db.Column(db.Text)
# Model type and status
model_type = db.Column(db.Enum(ModelType), nullable=False)
status = db.Column(
db.Enum(ModelStatus),
default=ModelStatus.DRAFT,
nullable=False,
index=True
)
# Architecture
architecture = db.Column(JSONB, nullable=False, default=dict)
hyperparameters = db.Column(JSONB, default=dict)
# Storage
weights_path = db.Column(db.String(500)) # Path to model weights
model_path = db.Column(db.String(500)) # Path to full model
# Performance metrics
metrics = db.Column(JSONB, default=dict)
# Deployment
deployment_info = db.Column(JSONB, default=dict)
endpoint_url = db.Column(db.String(500))
# Privacy
privacy_budget_used = db.Column(db.Float, default=0.0)
epsilon_used = db.Column(db.Float, default=0.0)
# Timestamps
created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
updated_at = db.Column(
db.DateTime,
default=datetime.utcnow,
onupdate=datetime.utcnow,
nullable=False
)
trained_at = db.Column(db.DateTime)
deployed_at = db.Column(db.DateTime)
# Metadata
tags = db.Column(JSONB, default=list)
metadata = db.Column(JSONB, default=dict)
# Relationships
experiments = db.relationship('Experiment', back_populates='model')
:
def __init__(self, name: str, user_id: uuid.UUID, model_type: ModelType, **kwargs)
"""Initialize a new model"""
self.name = name
self.user_id = user_id
self.model_type = model_type
for key, value in kwargs.items():
setattr(self, key, value)
def to_dict(self) -> Dict[str, Any]:
"""Convert model to dictionary"""
return {
'id': str(self.id),
'name': self.name,
'version': self.version,
'description': self.description,
'model_type': self.model_type.value,
'status': self.status.value,
'user_id': str(self.user_id),
'architecture': self.architecture,
'hyperparameters': self.hyperparameters,
'weights_path': self.weights_path,
'model_path': self.model_path,
'metrics': self.metrics,
'deployment_info': self.deployment_info,
'endpoint_url': self.endpoint_url,
'privacy_budget_used': self.privacy_budget_used,
'epsilon_used': self.epsilon_used,
'tags': self.tags,
'metadata': self.metadata,
'created_at': self.created_at.isoformat() if self.created_at else None,
'updated_at': self.updated_at.isoformat() if self.updated_at else None,
'trained_at': self.trained_at.isoformat() if self.trained_at else None,
'deployed_at': self.deployed_at.isoformat() if self.deployed_at else None
}
def update_status(self, new_status: ModelStatus) -> None:
"""Update model status with timestamp"""
self.status = new_status
if new_status == ModelStatus.DEPLOYED:
self.deployed_at = datetime.utcnow()
elif new_status == ModelStatus.VALIDATED:
self.trained_at = datetime.utcnow()
def add_metric(self, dataset_name: str, metric_name: str, value: float) -> None:
"""Add performance metric"""
if 'validation' not in self.metrics:
self.metrics['validation'] = {}
if dataset_name not in self.metrics['validation']:
self.metrics['validation'][dataset_name] = {}
self.metrics['validation'][dataset_name][metric_name] = value
def get_version_string(self) -> str:
"""Get full version string"""
return f"{self.name}-v{self.version}"
def __repr__(self) -> str:
return f'<Model {self.name} v{self.version} ({self.status.value})>'
