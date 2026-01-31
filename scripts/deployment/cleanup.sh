#!/bin/bash
# Cleanup script for Medical FL Platform
echo "=== Cleaning up Medical FL Platform ==="
# Stop and remove Docker containers
echo "Stopping Docker containers..."
docker-compose down -v
# Remove Docker images
echo "Removing Docker images..."
docker rmi medical-fl-backend medical-fl-frontend medical-fl-fl-server medical-fl-clie
nt || true
# Remove Docker volumes
echo "Removing Docker volumes..."
docker volume prune -f
# Remove unused Docker networks
echo "Cleaning up Docker networks..."
docker network prune -f
# Remove Kubernetes resources if deployed
if command -v kubectl &> /dev/null; then
echo "Removing Kubernetes resources..."
kubectl delete namespace medical-fl --ignore-not-found=true
kubectl delete pvc -n medical-fl --all --ignore-not-found=true
kubectl delete pv --all --ignore-not-found=true
fi
# Clean up local directories
echo "Cleaning up local directories..."
rm -rf logs/* mlflow/* models/* uploads/* data-simulation/*
# Remove Python cache files
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
# Remove node modules
echo "Removing node modules..."
rm -rf frontend/node_modules frontend/build
echo "=== Cleanup completed ==="
