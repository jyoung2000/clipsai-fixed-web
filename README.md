# ClipsAI Web Container

A fixed version of ClipsAI with:
- ✅ Precise video trimming (no more full videos)
- ✅ Intelligent aspect ratio conversion (16:9 ↔ 9:16)
- ✅ H.264 corruption fixes
- ✅ Broken pipe error handling
- ✅ Automatic clip generation
- ✅ Visual interest detection for cropping

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/jyoung2000/clipsai-fixed-web.git
cd clipsai-fixed-web

# Make start script executable
chmod +x start.sh

# Start the container
./start.sh
```

### Option 2: Docker Compose

```bash
# Build and start
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

### Option 3: Direct Docker

```bash
# Build
docker build -t clipsai-web .

# Run
docker run -d -p 5555:5555 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/downloads:/app/downloads \
  --name clipsai-web clipsai-web
```

## Access

- **Web Interface**: http://localhost:5555
- **Health Check**: http://localhost:5555/api/health
- **API Documentation**: Available in the web interface

## Features Fixed

### 1. Video Trimming
- **Before**: Generated full videos instead of clips
- **After**: Precise trimming to exact start/end times
- **Technology**: FFmpeg with exact `-ss` and `-t` parameters

### 2. Aspect Ratio Conversion
- **Before**: Simple cropping that cut off important content
- **After**: Intelligent cropping based on visual interest
- **Technology**: OpenCV face detection, motion analysis, edge detection

### 3. H.264 Corruption
- **Before**: Corrupted MP4 files with NAL unit errors
- **After**: Proper H.264 encoding with error tolerance
- **Technology**: FFmpeg with error-resilient parameters

### 4. Broken Pipe Errors
- **Before**: Server crashes when clients disconnect
- **After**: Graceful handling of connection drops
- **Technology**: Proper exception handling for `BrokenPipeError`

### 5. Automatic Clip Generation
- **Before**: Manual "Generate Clips" button that didn't work
- **After**: Automatic generation when clips are found
- **Technology**: Integrated workflow in the clip finding process

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Browser   │───▶│   Python Server │───▶│   FFmpeg/OpenCV│
│                 │    │                 │    │                 │
│ - Upload videos │    │ - AI transcription│    │ - Video processing│
│ - View clips    │    │ - Clip detection │    │ - Aspect conversion│
│ - Download      │    │ - Visual analysis│    │ - Precise trimming│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## API Endpoints

- `GET /` - Main web interface
- `GET /api/health` - Health check
- `POST /api/upload` - Upload video file
- `POST /api/transcribe` - Transcribe video using Whisper AI
- `POST /api/find_clips` - Find interesting clips (auto-generates)
- `POST /api/trim_clip` - Trim specific clip
- `GET /uploads/*` - Serve uploaded files
- `GET /downloads/*` - Serve generated clips

## Testing

```bash
# Test health
curl http://localhost:5555/api/health

# Test upload (replace with your video file)
curl -X POST -F "video=@your_video.mp4" http://localhost:5555/api/upload

# Test transcription
curl -X POST -H "Content-Type: application/json" \
  -d '{"video_id":"your_video_id"}' \
  http://localhost:5555/api/transcribe
```

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker-compose logs

# Check if port 5555 is available
lsof -i :5555

# Force recreate
docker-compose down
docker-compose up --force-recreate
```

### Dependencies Issues
```bash
# Rebuild without cache
docker-compose build --no-cache

# Check Python dependencies
docker-compose exec clipsai pip list
```

### Video Processing Issues
```bash
# Check FFmpeg version
docker-compose exec clipsai ffmpeg -version

# Check OpenCV installation
docker-compose exec clipsai python -c "import cv2; print(cv2.__version__)"
```

## Performance

- **CPU**: Optimized for CPU-only inference (no GPU required)
- **Memory**: ~2GB RAM recommended for video processing
- **Storage**: Temporary files are cleaned up automatically
- **Network**: Supports HTTP range requests for large files

## Security

- Files are served with proper MIME types
- Upload size limits enforced
- Temporary files cleaned up after processing
- No external network access required after setup

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python server.py
```

### Environment Variables
- `PORT`: Server port (default: 5555)
- `UPLOAD_DIR`: Upload directory (default: ./uploads)
- `PYTHONUNBUFFERED`: Enable unbuffered output

### File Structure
```
clipsai-fixed-web/
├── server.py              # Main server application
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Orchestration
├── requirements.txt       # Python dependencies
├── start.sh              # Startup script
├── README.md             # This file
├── uploads/              # Uploaded videos
├── downloads/            # Generated clips
└── temp/                 # Temporary files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues, questions, or feature requests, please open an issue on GitHub.

---

**🎬 ClipsAI Fixed Web Container** - Precise video processing made simple!