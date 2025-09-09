#!/bin/bash

# Production deployment script for Receipt OCR API

set -e

echo "🚀 Starting production deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Pull latest images (if using registry)
echo "📦 Pulling latest images..."
# docker-compose pull

# Build the application
echo "🔨 Building the application..."
docker-compose build --no-cache

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Start the application
echo "🎯 Starting the application..."
docker-compose up -d

# Wait for health check
echo "🔍 Waiting for application to be healthy..."
sleep 10

# Check if the application is running
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Application is running successfully!"
    echo "🌐 API is available at: http://localhost:8000"
    echo "📊 Health check: http://localhost:8000/health"
    echo "📖 API docs: http://localhost:8000/docs"
else
    echo "❌ Application failed to start properly"
    echo "📋 Checking logs..."
    docker-compose logs --tail=50
    exit 1
fi

echo "🎉 Deployment completed successfully!"
