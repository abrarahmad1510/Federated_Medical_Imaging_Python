#!/bin/bash
# Medical FL Platform Deployment Script
set -e
echo "=== Medical FL Platform Deployment ==="
echo "Date: $(date)"
echo "User: $(whoami)"
echo ""
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
# Logging function
log() {
echo -e "${GREEN}[INFO]${NC} $1"
}
warn() {
echo -e "${YELLOW}[WARN]${NC} $1"
}
error() {
echo -e "${RED}[ERROR]${NC} $1"
}
# Check prerequisites
check_prerequisites() {
log "Checking prerequisites..."
# Check Docker
if ! command -v docker &> /dev/null; then
error "Docker is not installed"
exit 1
fi
log "✓ Docker installed"
# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
error "Docker Compose is not installed"
exit 1
fi
log "✓ Docker Compose installed"
# Check kubectl if K8s deployment
if [ "$DEPLOY_MODE" = "kubernetes" ]; then
if ! command -v kubectl &> /dev/null; then
error "kubectl is not installed"
exit 1
fi
log "✓ kubectl installed"
fi
# Check environment file
if [ ! -f .env ]; then
warn ".env file not found, creating from template"
if [ -f .env.example ]; then
cp .env.example .env
warn "Please edit .env file with your configuration"
exit 1
else
error ".env.example not found"
exit 1
fi
fi
log "✓ Environment file found"
}
# Load environment
load_environment() {
log "Loading environment..."
set -a
source .env
set +a
}
# Build Docker images
build_images() {
log "Building Docker images..."
# Build backend
log "Building backend image..."
docker build -t medical-fl-backend:latest -f deploy/docker/backend.Dockerfile .
# Build frontend
log "Building frontend image..."
docker build -t medical-fl-frontend:latest -f deploy/docker/frontend.Dockerfile .
# Build FL server
log "Building FL server image..."
docker build -t medical-fl-fl-server:latest -f deploy/docker/fl-server.Dockerfile .
# Build client
log "Building client image..."
docker build -t medical-fl-client:latest -f deploy/docker/client.Dockerfile .
log "✓ All images built successfully"
}
# Start with Docker Compose
start_docker_compose() {
log "Starting with Docker Compose..."
# Stop existing containers
log "Stopping existing containers..."
docker-compose down || true
# Build and start
log "Building and starting containers..."
docker-compose up --build -d
# Wait for services to be ready
log "Waiting for services to be ready..."
sleep 30
# Check service health
check_service_health
}
# Deploy to Kubernetes
deploy_kubernetes() {
log "Deploying to Kubernetes..."
# Create namespace
log "Creating namespace..."
kubectl apply -f deploy/kubernetes/namespace.yaml
# Create secrets
log "Creating secrets..."
kubectl apply -f deploy/kubernetes/secrets/
# Create config maps
log "Creating config maps..."
kubectl apply -f deploy/kubernetes/configs/
# Deploy PostgreSQL
log "Deploying PostgreSQL..."
kubectl apply -f deploy/kubernetes/postgres/
# Wait for PostgreSQL
log "Waiting for PostgreSQL to be ready..."
kubectl wait --namespace medical-fl \
--for=condition=ready pod \
--selector=app=postgres \
--timeout=300s
# Deploy Redis
log "Deploying Redis..."
kubectl apply -f deploy/kubernetes/redis/
# Deploy backend
log "Deploying backend..."
kubectl apply -f deploy/kubernetes/backend/
# Deploy frontend
log "Deploying frontend..."
kubectl apply -f deploy/kubernetes/frontend/
# Deploy FL server
log "Deploying FL server..."
kubectl apply -f deploy/kubernetes/fl-server/
# Deploy monitoring
log "Deploying monitoring..."
kubectl apply -f deploy/kubernetes/monitoring/
# Deploy clients
log "Deploying clients..."
kubectl apply -f deploy/kubernetes/clients/
log "✓ Kubernetes deployment completed"
}
# Check service health
check_service_health() {
log "Checking service health..."
# Backend health
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
log "✓ Backend is healthy"
else
error "Backend health check failed"
return 1
fi
# Frontend health
if curl -f http://localhost:3000 > /dev/null 2>&1; then
log "✓ Frontend is healthy"
else
error "Frontend health check failed"
return 1
fi
# Flower server health
if curl -f http://localhost:8080 > /dev/null 2>&1; then
log "✓ Flower server is healthy"
else
warn "Flower server health check failed (might be starting)"
fi
# MLflow health
if curl -f http://localhost:5001 > /dev/null 2>&1; then
log "✓ MLflow is healthy"
else
warn "MLflow health check failed (might be starting)"
fi
return 0
}
# Run database migrations
run_migrations() {
log "Running database migrations..."
# Get backend pod
BACKEND_POD=$(kubectl get pods -n medical-fl -l app=backend -o jsonpath='{.items[0].metadata.name}')
# Run migrations
kubectl exec -n medical-fl $BACKEND_POD -- flask db upgrade
log "✓ Database migrations completed"
}
# Initialize the system
initialize_system() {
log "Initializing system..."
# Get backend pod
BACKEND_POD=$(kubectl get pods -n medical-fl -l app=backend -o jsonpath='{.items[0].metadata.name}')
# Create admin user
log "Creating admin user..."
kubectl exec -n medical-fl $BACKEND_POD -- flask create-admin << 'EOF'
admin@medicalfl.example.com
AdminPassword123!
EOF
# Seed sample data
log "Seeding sample data..."
kubectl exec -n medical-fl $BACKEND_POD -- flask seed-data
log "✓ System initialization completed"
}
# Display deployment information
show_deployment_info() {
echo ""
echo "=== Deployment Information ==="
echo ""
if [ "$DEPLOY_MODE" = "docker" ]; then
echo "Services are running at:"
echo " Frontend: http://localhost:3000"
echo " Backend API: http://localhost:5000"
echo " Flower Server: http://localhost:8080"
echo " MLflow: http://localhost:5001"
echo " Grafana: http://localhost:3001 (admin/admin)"
echo ""
echo "To view logs:"
echo " docker-compose logs -f [service]"
echo ""
echo "To stop:"
echo " docker-compose down"
else
echo "Kubernetes deployment completed in namespace: medical-fl"
echo ""
echo "To get service URLs:"
echo " kubectl get svc -n medical-fl"
echo ""
echo "To view pods:"
echo " kubectl get pods -n medical-fl"
echo ""
echo "To view logs:"
echo " kubectl logs -n medical-fl [pod-name]"
echo ""
echo "To access the frontend:"
echo " kubectl port-forward svc/frontend 3000:80 -n medical-fl"
echo " Then open: http://localhost:3000"
fi
echo ""
echo "Default credentials:"
echo " Email: admin@medicalfl.example.com"
echo " Password: AdminPassword123!"
echo ""
echo "=== Deployment Completed Successfully ==="
}
# Main deployment function
main() {
DEPLOY_MODE=${1:-"docker"}
log "Starting deployment in $DEPLOY_MODE mode"
# Check prerequisites
check_prerequisites
# Load environment
load_environment
# Build images
build_images
# Deploy based on mode
case $DEPLOY_MODE in
"docker")
start_docker_compose
;;
"kubernetes"|"k8s")
deploy_kubernetes
sleep 30 # Wait for pods to start
run_migrations
initialize_system
;;
*)
error "Invalid deployment mode: $DEPLOY_MODE"
echo "Usage: $0 [docker|kubernetes]"
exit 1
;;
esac
# Show deployment info
show_deployment_info
}
# Run main function
main "$@"
