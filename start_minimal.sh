#!/bin/bash

# Start ClipsAI Minimal Container (No TorchAudio/FFmpeg-Python conflicts)
echo "🚀 Starting ClipsAI Minimal Container"

# Create necessary directories
mkdir -p uploads downloads temp

# Stop and remove existing container
echo "🛑 Stopping existing containers..."
docker stop clipsai-minimal 2>/dev/null || true
docker rm clipsai-minimal 2>/dev/null || true
docker-compose -f docker-compose.minimal.yml down 2>/dev/null || true

# Build and start
echo "🔨 Building minimal container..."
docker-compose -f docker-compose.minimal.yml build --no-cache

echo "🚀 Starting container..."
docker-compose -f docker-compose.minimal.yml up -d

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
    echo "   View logs: docker-compose -f docker-compose.minimal.yml logs -f"
    echo "   Stop: docker-compose -f docker-compose.minimal.yml down"
    echo "   Restart: docker-compose -f docker-compose.minimal.yml restart"
    echo ""
    echo "🎯 Features:"
    echo "   ✅ Precise video trimming"
    echo "   ✅ Automatic clip generation"
    echo "   ✅ Broken pipe error handling"
    echo "   ✅ No TorchAudio/FFmpeg-Python conflicts"
else
    echo "❌ Container health check failed. Checking logs..."
    docker-compose -f docker-compose.minimal.yml logs
fi