# ClipsAI Web Container - Production Ready
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/downloads /app/temp && \
    chmod 777 /app/uploads /app/downloads /app/temp

# Create test video
RUN ffmpeg -f lavfi -i testsrc=duration=60:size=1280x720:rate=30 \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
    /app/test_vid.mp4

# Expose port
EXPOSE 5555

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5555
ENV UPLOAD_DIR=/app/uploads

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5555/api/health || exit 1

# Start command
CMD ["python", "server.py"]