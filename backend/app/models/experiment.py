"""
Experiment model for managing FL training sessions
"""
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from .. import db
class ExperimentStatus(Enum):
"""Experiment status enumeration"""
DRAFT = 'draft'
PENDING = 'pending'
RUNNING = 'running'
PAUSED = 'paused'
COMPLETED = 'completed'
FAILED = 'failed'
CANCELLED = 'cancelled'
class Experiment(db.Model):
"""Experiment model for FL training sessions"""
__tablename__ = 'experiments'
# Primary key
id = db.Column(
UUID(as_uuid=True),
primary_key=True,
default=uuid.uuid4,
nullable=False
)
# Foreign keys
user_id = db.Column(
UUID(as_uuid=True),
db.ForeignKey('users.id'),
nullable=False,
index=True
)
model_id = db.Column(
UUID(as_uuid=True),
db.ForeignKey('models.id'),
nullable=True,
index=True
)
# Basic information
name = db.Column(db.String(255), nullable=False)
description = db.Column(db.Text)
# Status
status = db.Column(
db.Enum(ExperimentStatus),
default=ExperimentStatus.DRAFT,
nullable=False,
index=True
)
# Configuration
config = db.Column(JSONB, nullable=False, default=dict)
# Model configuration
model_config = db.Column(JSONB, nullable=False, default=dict)
# FL configuration
fl_config = db.Column(JSONB, nullable=False, default=dict)
# Privacy configuration
privacy_config = db.Column(JSONB, nullable=False, default=dict)
# Training metrics
metrics = db.Column(JSONB, default=dict)
# Results
results = db.Column(JSONB, default=dict)
# Timestamps
created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
updated_at = db.Column(
db.DateTime,
default=datetime.utcnow,
onupdate=datetime.utcnow,
nullable=False
)
started_at = db.Column(db.DateTime)
completed_at = db.Column(db.DateTime)
# Metadata
tags = db.Column(JSONB, default=list)
metadata = db.Column(JSONB, default=dict)
# Relationships
user = db.relationship('User', back_populates='experiments')
model = db.relationship('Model', back_populates='experiments')
client_sessions = db.relationship('ClientSession', back_populates='experiment')
def __init__(self, name: str, user_id: uuid.UUID, **kwargs):
"""Initialize a new experiment"""
self.name = name
self.user_id = user_id
for key, value in kwargs.items():
setattr(self, key, value)
def to_dict(self) -> Dict[str, Any]:
"""Convert experiment to dictionary"""
return {
'id': str(self.id),
'name': self.name,
'description': self.description,
'status': self.status.value,
'user_id': str(self.user_id),
'model_id': str(self.model_id) if self.model_id else None,
'config': self.config,
'model_config': self.model_config,
'fl_config': self.fl_config,
'privacy_config': self.privacy_config,
'metrics': self.metrics,
'results': self.results,
'tags': self.tags,
'metadata': self.metadata,
'created_at': self.created_at.isoformat() if self.created_at else None,
'updated_at': self.updated_at.isoformat() if self.updated_at else None,
'started_at': self.started_at.isoformat() if self.started_at else None,
'completed_at': self.completed_at.isoformat() if self.completed_at else No
ne
}
def update_status(self, new_status: ExperimentStatus) -> None:
"""Update experiment status with timestamp"""
old_status = self.status
self.status = new_status
if new_status == ExperimentStatus.RUNNING and old_status != ExperimentStatus.R
UNNING:
self.started_at = datetime.utcnow()
elif new_status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, Exper
imentStatus.CANCELLED]:
self.completed_at = datetime.utcnow()
def add_metric(self, round_num: int, metric_name: str, value: float) -> None:
"""Add training metric"""
if 'rounds' not in self.metrics:
self.metrics['rounds'] = {}
if round_num not in self.metrics['rounds']:
self.metrics['rounds'][round_num] = {}
self.metrics['rounds'][round_num][metric_name] = value
self.metrics['last_updated'] = datetime.utcnow().isoformat()
def __repr__(self) -> str:
return f'<Experiment {self.name} ({self.status.value})>'
