"""
Experiment management API endpoints
"""
from datetime import datetime
from typing import Dict, Any, List
from flask import request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_restx import Namespace, Resource, fields, reqparse
import uuid
from .... import db
from ...models.experiment import Experiment, ExperimentStatus
from ...models.user import User
from ...utils.decorators import validate_json
from ...services.fl_orchestrator import FLOrchestrator
# Create namespace
bp = Namespace('experiments', description='Experiment operations')
# Request/Response models
experiment_model = bp.model('Experiment', {
'name': fields.String(required=True, description='Experiment name'),
'description': fields.String(description='Experiment description'),
'config': fields.Raw(description='Experiment configuration'),
'model_config': fields.Raw(description='Model configuration'),
'fl_config': fields.Raw(description='FL configuration'),
'privacy_config': fields.Raw(description='Privacy configuration'),
'tags': fields.List(fields.String, description='Experiment tags')
})
experiment_response = bp.model('ExperimentResponse', {
'id': fields.String(description='Experiment ID'),
'name': fields.String(description='Experiment name'),
'description': fields.String(description='Experiment description'),
'status': fields.String(description='Experiment status'),
'user_id': fields.String(description='User ID'),
'model_id': fields.String(description='Model ID'),
'config': fields.Raw(description='Experiment configuration'),
'model_config': fields.Raw(description='Model configuration'),
'fl_config': fields.Raw(description='FL configuration'),
'privacy_config': fields.Raw(description='Privacy configuration'),
'metrics': fields.Raw(description='Training metrics'),
'results': fields.Raw(description='Experiment results'),
'tags': fields.Raw(description='Experiment tags'),
'metadata': fields.Raw(description='Experiment metadata'),
'created_at': fields.String(description='Creation timestamp'),
'updated_at': fields.String(description='Update timestamp'),
'started_at': fields.String(description='Start timestamp'),
'completed_at': fields.String(description='Completion timestamp')
})
experiment_list_response = bp.model('ExperimentListResponse', {
'experiments': fields.List(fields.Nested(experiment_response)),
'total': fields.Integer(description='Total experiments'),
'page': fields.Integer(description='Current page'),
'per_page': fields.Integer(description='Items per page'),
'pages': fields.Integer(description='Total pages')
})
# Query parser
parser = reqparse.RequestParser()
parser.add_argument('page', type=int, default=1, help='Page number')
parser.add_argument('per_page', type=int, default=20, help='Items per page')
parser.add_argument('status', type=str, help='Filter by status')
parser.add_argument('search', type=str, help='Search in name/description')
parser.add_argument('sort_by', type=str, default='created_at', help='Sort field')
parser.add_argument('sort_order', type=str, default='desc', help='Sort order')
@bp.route('')
class ExperimentList(Resource):
@jwt_required()
@bp.expect(parser)
@bp.response(200, 'Success', experiment_list_response)
def get(self):
"""Get all experiments for current user"""
args = parser.parse_args()
user_id = get_jwt_identity()
# Build query
query = Experiment.query.filter_by(user_id=uuid.UUID(user_id))
# Apply filters
if args['status']:
query = query.filter_by(status=ExperimentStatus(args['status']))
if args['search']:
search_term = f"%{args['search']}%"
query = query.filter(
db.or_(
Experiment.name.ilike(search_term),
Experiment.description.ilike(search_term)
)
)
# Apply sorting
sort_field = getattr(Experiment, args['sort_by'], Experiment.created_at)
if args['sort_order'] == 'desc':
query = query.order_by(db.desc(sort_field))
else:
query = query.order_by(sort_field)
# Pagination
total = query.count()
per_page = min(args['per_page'], 100) # Limit to 100 per page
page = args['page']
experiments = query.paginate(
page=page,
per_page=per_page,
error_out=False
)
response = {
'experiments': [exp.to_dict() for exp in experiments.items],
'total': total,
'page': page,
'per_page': per_page,
'pages': experiments.pages
}
return response, 200
@jwt_required()
@bp.expect(experiment_model)
@bp.response(201, 'Experiment created', experiment_response)
@bp.response(400, 'Invalid data')
@validate_json
def post(self):
"""Create a new experiment"""
user_id = get_jwt_identity()
data = request.get_json()
# Validate required fields
if not data or 'name' not in data:
return {'message': 'Experiment name is required'}, 400
# Create experiment
try:
experiment = Experiment(
name=data['name'],
user_id=uuid.UUID(user_id),
description=data.get('description', ''),
config=data.get('config', {}),
model_config=data.get('model_config', {}),
fl_config=data.get('fl_config', {}),
privacy_config=data.get('privacy_config', {}),
tags=data.get('tags', []),
metadata=data.get('metadata', {})
)
db.session.add(experiment)
db.session.commit()
# Log experiment creation
current_app.logger.info(f'Experiment created: {experiment.id} by user {use
r_id}')
return experiment.to_dict(), 201
except Exception as e:
db.session.rollback()
current_app.logger.error(f'Experiment creation error: {str(e)}')
return {'message': 'Failed to create experiment'}, 500
@bp.route('/<string:experiment_id>')
class ExperimentDetail(Resource):
@jwt_required()
@bp.response(200, 'Success', experiment_response)
@bp.response(404, 'Experiment not found')
def get(self, experiment_id: str):
"""Get experiment details"""
user_id = get_jwt_identity()
try:
experiment = Experiment.query.filter_by(
id=uuid.UUID(experiment_id),
user_id=uuid.UUID(user_id)
).first()
if not experiment:
return {'message': 'Experiment not found'}, 404
return experiment.to_dict(), 200
except ValueError:
return {'message': 'Invalid experiment ID'}, 400
@jwt_required()
@bp.expect(experiment_model)
@bp.response(200, 'Experiment updated', experiment_response)
@bp.response(404, 'Experiment not found')
@validate_json
def put(self, experiment_id: str):
"""Update experiment"""
user_id = get_jwt_identity()
data = request.get_json()
try:
experiment = Experiment.query.filter_by(
id=uuid.UUID(experiment_id),
user_id=uuid.UUID(user_id)
).first()
if not experiment:
return {'message': 'Experiment not found'}, 404
# Only allow updates if not running
if experiment.status == ExperimentStatus.RUNNING:
return {'message': 'Cannot update running experiment'}, 400
# Update fields
if 'name' in data:
experiment.name = data['name']
if 'description' in data:
experiment.description = data.get('description')
if 'config' in data:
experiment.config = data['config']
if 'model_config' in data:
experiment.model_config = data['model_config']
if 'fl_config' in data:
experiment.fl_config = data['fl_config']
if 'privacy_config' in data:
experiment.privacy_config = data['privacy_config']
if 'tags' in data:
experiment.tags = data['tags']
if 'metadata' in data:
experiment.metadata = data['metadata']
db.session.commit()
return experiment.to_dict(), 200
except ValueError:
return {'message': 'Invalid experiment ID'}, 400
except Exception as e:
db.session.rollback()
current_app.logger.error(f'Experiment update error: {str(e)}')
return {'message': 'Failed to update experiment'}, 500
@jwt_required()
@bp.response(200, 'Experiment deleted')
@bp.response(404, 'Experiment not found')
def delete(self, experiment_id: str):
"""Delete experiment"""
user_id = get_jwt_identity()
try:
experiment = Experiment.query.filter_by(
id=uuid.UUID(experiment_id),
user_id=uuid.UUID(user_id)
).first()
if not experiment:
return {'message': 'Experiment not found'}, 404
# Only allow deletion if not running
if experiment.status == ExperimentStatus.RUNNING:
return {'message': 'Cannot delete running experiment'}, 400
db.session.delete(experiment)
db.session.commit()
current_app.logger.info(f'Experiment deleted: {experiment_id} by user {use
r_id}')
return {'message': 'Experiment deleted successfully'}, 200
except ValueError:
return {'message': 'Invalid experiment ID'}, 400
except Exception as e:
db.session.rollback()
current_app.logger.error(f'Experiment deletion error: {str(e)}')
return {'message': 'Failed to delete experiment'}, 500
@bp.route('/<string:experiment_id>/start')
class StartExperiment(Resource):
@jwt_required()
@bp.response(200, 'Experiment started')
@bp.response(404, 'Experiment not found')
@bp.response(400, 'Cannot start experiment')
def post(self, experiment_id: str):
"""Start experiment training"""
user_id = get_jwt_identity()
try:
experiment = Experiment.query.filter_by(
id=uuid.UUID(experiment_id),
user_id=uuid.UUID(user_id)
).first()
if not experiment:
return {'message': 'Experiment not found'}, 404
# Check if experiment can be started
if experiment.status != ExperimentStatus.PENDING:
return {
'message': f'Cannot start experiment in {experiment.status.value}
state'
}, 400
# Update status
experiment.update_status(ExperimentStatus.RUNNING)
db.session.commit()
# Start FL training asynchronously
fl_orchestrator = FLOrchestrator(current_app)
fl_orchestrator.start_experiment(experiment.id)
current_app.logger.info(f'Experiment started: {experiment_id}')
return {'message': 'Experiment started successfully'}, 200
except ValueError:
return {'message': 'Invalid experiment ID'}, 400
except Exception as e:
db.session.rollback()
current_app.logger.error(f'Experiment start error: {str(e)}')
# Update status to failed
if experiment:
experiment.update_status(ExperimentStatus.FAILED)
db.session.commit()
return {'message': f'Failed to start experiment: {str(e)}'}, 500
@bp.route('/<string:experiment_id>/stop')
class StopExperiment(Resource):
@jwt_required()
@bp.response(200, 'Experiment stopped')
@bp.response(404, 'Experiment not found')
@bp.response(400, 'Cannot stop experiment')
def post(self, experiment_id: str):
"""Stop experiment training"""
user_id = get_jwt_identity()
try:
experiment = Experiment.query.filter_by(
id=uuid.UUID(experiment_id),
user_id=uuid.UUID(user_id)
).first()
if not experiment:
return {'message': 'Experiment not found'}, 404
# Check if experiment can be stopped
if experiment.status != ExperimentStatus.RUNNING:
return {
'message': f'Cannot stop experiment in {experiment.status.value} s
tate'
}, 400
# Update status
experiment.update_status(ExperimentStatus.CANCELLED)
db.session.commit()
# Stop FL training
fl_orchestrator = FLOrchestrator(current_app)
fl_orchestrator.stop_experiment(experiment.id)
current_app.logger.info(f'Experiment stopped: {experiment_id}')
return {'message': 'Experiment stopped successfully'}, 200
except ValueError:
return {'message': 'Invalid experiment ID'}, 400
except Exception as e:
db.session.rollback()
current_app.logger.error(f'Experiment stop error: {str(e)}')
return {'message': 'Failed to stop experiment'}, 500
@bp.route('/<string:experiment_id>/metrics')
class ExperimentMetrics(Resource):
@jwt_required()
@bp.response(200, 'Success')
@bp.response(404, 'Experiment not found')
def get(self, experiment_id: str):
"""Get experiment metrics"""
user_id = get_jwt_identity()
try:
experiment = Experiment.query.filter_by(
id=uuid.UUID(experiment_id),
user_id=uuid.UUID(user_id)
).first()
if not experiment:
return {'message': 'Experiment not found'}, 404
return {
'experiment_id': experiment_id,
'metrics': experiment.metrics or {}
}, 200
except ValueError:
return {'message': 'Invalid experiment ID'}, 400
@bp.route('/<string:experiment_id>/results')
class ExperimentResults(Resource):
@jwt_required()
@bp.response(200, 'Success')
@bp.response(404, 'Experiment not found')
def get(self, experiment_id: str):
"""Get experiment results"""
user_id = get_jwt_identity()
try:
experiment = Experiment.query.filter_by(
id=uuid.UUID(experiment_id),
user_id=uuid.UUID(user_id)
).first()
if not experiment:
return {'message': 'Experiment not found'}, 404
return {
'experiment_id': experiment_id,
'results': experiment.results or {},
'status': experiment.status.value
}, 200
except ValueError:
return {'message': 'Invalid experiment ID'}, 400
