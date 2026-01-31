"""
Authentication API endpoints
"""
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
from flask import request, jsonify, current_app
from flask_jwt_extended import (
create_access_token,
create_refresh_token,
jwt_required,
get_jwt_identity,
get_jwt
)
from flask_restx import Namespace, Resource, fields, reqparse
import uuid
from .... import db
from ...models.user import User
from ...utils.decorators import admin_required, rate_limit
# Create namespace
bp = Namespace('auth', description='Authentication operations')
# Request/Response models
login_model = bp.model('Login', {
'email': fields.String(required=True, description='User email'),
'password': fields.String(required=True, description='User password')
})
register_model = bp.model('Register', {
'email': fields.String(required=True, description='User email'),
'password': fields.String(required=True, description='User password'),
'first_name': fields.String(description='First name'),
'last_name': fields.String(description='Last name'),
'username': fields.String(description='Username')
})
token_response = bp.model('TokenResponse', {
'access_token': fields.String(description='JWT access token'),
'refresh_token': fields.String(description='JWT refresh token'),
'token_type': fields.String(description='Token type (Bearer)'),
'expires_in': fields.Integer(description='Token expiration in seconds'),
'user': fields.Raw(description='User information')
})
user_response = bp.model('UserResponse', {
'id': fields.String(description='User ID'),
'email': fields.String(description='User email'),
'username': fields.String(description='Username'),
'first_name': fields.String(description='First name'),
'last_name': fields.String(description='Last name'),
'is_active': fields.Boolean(description='Active status'),
'is_admin': fields.Boolean(description='Admin status'),
'email_verified': fields.Boolean(description='Email verified'),
'created_at': fields.String(description='Creation timestamp'),
'last_login': fields.String(description='Last login timestamp')
})
@bp.route('/login')
class Login(Resource):
@bp.expect(login_model)
@bp.response(200, 'Success', token_response)
@bp.response(401, 'Invalid credentials')
@rate_limit(5, 60) # 5 attempts per minute
def post(self):
"""User login"""
data = request.get_json()
# Validate required fields
if not data or 'email' not in data or 'password' not in data:
return {'message': 'Email and password are required'}, 400
# Find user
user = User.query.filter_by(email=data['email']).first()
# Check user exists and password is correct
if not user or not user.check_password(data['password']):
return {'message': 'Invalid email or password'}, 401
# Check if user is active
if not user.is_active:
return {'message': 'Account is disabled'}, 403
# Update last login
user.last_login = datetime.utcnow()
db.session.commit()
# Create tokens
access_token = create_access_token(
identity=str(user.id),
additional_claims={
'email': user.email,
'is_admin': user.is_admin
}
)
refresh_token = create_refresh_token(identity=str(user.id))
# Prepare response
response = {
'access_token': access_token,
'refresh_token': refresh_token,
'token_type': 'Bearer',
'expires_in': current_app.config['JWT_ACCESS_TOKEN_EXPIRES'].seconds,
'user': user.to_dict()
}
return response, 200
@bp.route('/register')
class Register(Resource):
@bp.expect(register_model)
@bp.response(201, 'User created', user_response)
@bp.response(400, 'Invalid data')
@bp.response(409, 'User already exists')
@rate_limit(2, 300) # 2 registrations per 5 minutes
def post(self):
"""Register new user"""
data = request.get_json()
# Validate required fields
if not data or 'email' not in data or 'password' not in data:
return {'message': 'Email and password are required'}, 400
# Check if user already exists
existing_user = User.query.filter_by(email=data['email']).first()
if existing_user:
return {'message': 'User with this email already exists'}, 409
# Check username uniqueness if provided
if 'username' in data and data['username']:
existing_username = User.query.filter_by(username=data['username']).first(
)
if existing_username:
return {'message': 'Username already taken'}, 409
# Create new user
try:
user = User(
email=data['email'],
password=data['password'],
first_name=data.get('first_name'),
last_name=data.get('last_name'),
username=data.get('username')
)
db.session.add(user)
db.session.commit()
# Send verification email (in production)
# send_verification_email(user)
return user.to_dict(), 201
except Exception as e:
db.session.rollback()
current_app.logger.error(f'Registration error: {str(e)}')
return {'message': 'Failed to create user'}, 500
@bp.route('/refresh')
class Refresh(Resource):
@jwt_required(refresh=True)
@bp.response(200, 'Success', token_response)
@bp.response(401, 'Invalid refresh token')
def post(self):
"""Refresh access token"""
user_id = get_jwt_identity()
# Find user
user = User.query.get(uuid.UUID(user_id))
if not user or not user.is_active:
return {'message': 'User not found or inactive'}, 401
# Create new access token
access_token = create_access_token(
identity=str(user.id),
additional_claims={
'email': user.email,
'is_admin': user.is_admin
}
)
response = {
'access_token': access_token,
'token_type': 'Bearer',
'expires_in': current_app.config['JWT_ACCESS_TOKEN_EXPIRES'].seconds
}
return response, 200
@bp.route('/logout')
class Logout(Resource):
@jwt_required()
@bp.response(200, 'Successfully logged out')
def post(self):
"""User logout"""
# In a production system, you might want to add the token to a blacklist
# For now, we'll just return success
return {'message': 'Successfully logged out'}, 200
@bp.route('/me')
class CurrentUser(Resource):
@jwt_required()
@bp.response(200, 'Success', user_response)
@bp.response(401, 'Unauthorized')
def get(self):
"""Get current user information"""
user_id = get_jwt_identity()
user = User.query.get(uuid.UUID(user_id))
if not user:
return {'message': 'User not found'}, 404
return user.to_dict(), 200
@bp.route('/change-password')
class ChangePassword(Resource):
@jwt_required()
@bp.expect(bp.model('ChangePassword', {
'current_password': fields.String(required=True),
'new_password': fields.String(required=True, min_length=8)
}))
@bp.response(200, 'Password changed')
@bp.response(400, 'Invalid request')
@bp.response(401, 'Invalid current password')
def post(self):
"""Change user password"""
user_id = get_jwt_identity()
data = request.get_json()
# Validate request
if not data or 'current_password' not in data or 'new_password' not in data:
return {'message': 'Current and new password are required'}, 400
if len(data['new_password']) < 8:
return {'message': 'New password must be at least 8 characters'}, 400
# Find user
user = User.query.get(uuid.UUID(user_id))
if not user:
return {'message': 'User not found'}, 404
# Verify current password
if not user.check_password(data['current_password']):
return {'message': 'Current password is incorrect'}, 401
# Update password
try:
user.set_password(data['new_password'])
db.session.commit()
return {'message': 'Password updated successfully'}, 200
except Exception as e:
db.session.rollback()
current_app.logger.error(f'Password change error: {str(e)}')
return {'message': 'Failed to update password'}, 500
@bp.route('/admin/users')
class UserList(Resource):
@jwt_required()
@admin_required
@bp.response(200, 'Success', [user_response])
@bp.response(403, 'Admin access required')
def get(self):
"""Get all users (admin only)"""
users = User.query.all()
return [user.to_dict() for user in users], 200
