#!/bin/bash

# Start ClipsAI Web Container
echo "🚀 Starting ClipsAI Web Container"

# Create necessary directories
mkdir -p uploads downloads temp

# Stop and remove existing container
echo "🛑 Stopping existing container..."
docker-compose down

# Build and start
echo "🔨 Building container..."
docker-compose build --no-cache

echo "🚀 Starting container..."
docker-compose up -d

# Wait for startup
echo "⏳ Waiting for container to start..."
sleep 30

# Check health
echo "🏥 Checking health..."
if curl -f http://localhost:5555/api/health > /dev/null 2>&1; then
    echo "✅ Container is healthy and running!"
    echo "🌐 Access at: http://localhost:5555"
    echo ""
    echo "📋 Useful commands:"
    echo "   View logs: docker-compose logs -f"
    echo "   Stop: docker-compose down"
    echo "   Restart: docker-compose restart"
else
    echo "❌ Container health check failed. Checking logs..."
    docker-compose logs
fi