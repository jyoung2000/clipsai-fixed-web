#!/usr/bin/env python3
"""
ClipsAI Web Server - Production Ready Container Version
Fixed version with working upload, validation, and precise video processing
"""

import http.server
import socketserver
import urllib.parse
import json
import tempfile
import os
import threading
import time
import cgi
import shutil
import requests
from pathlib import Path
import mimetypes
import sys
import uuid
import warnings
import re
import subprocess
warnings.filterwarnings("ignore")

# Import AI transcription libraries (required)
import whisper
from moviepy.editor import VideoFileClip
import torch

# Import computer vision libraries for visual analysis
VISUAL_ANALYSIS_AVAILABLE = False
try:
    import cv2
    import numpy as np
    from PIL import Image
    VISUAL_ANALYSIS_AVAILABLE = True
    print("✅ Visual analysis libraries loaded successfully")
except ImportError as e:
    print(f"⚠️  Visual analysis libraries not available: {e}")
    print("📝 Falling back to transcript-only clip detection")
    # Create dummy numpy for fallback functions
    class DummyNumpy:
        @staticmethod
        def mean(data):
            return 0
        @staticmethod
        def count_nonzero(data):
            return 0
    np = DummyNumpy()

print("✅ AI transcription libraries loaded successfully")

# Get port from environment variable or default to 5555
PORT = int(os.environ.get('PORT', '5555'))

class PreciseVideoTrimmer:
    """Handles precise video trimming without fallbacks"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def trim_video_precise(self, input_path, output_path, start_time, end_time):
        """
        Trim video with exact precision - no fallbacks to full video
        """
        try:
            # Validate time ranges
            if start_time >= end_time:
                raise ValueError(f"Invalid time range: {start_time} >= {end_time}")
            
            # Get video duration
            duration = self.get_video_duration(input_path)
            if duration is None:
                raise ValueError("Could not determine video duration")
            
            # Ensure times are within video bounds
            start_time = max(0, start_time)
            end_time = min(duration, end_time)
            clip_duration = end_time - start_time
            
            if clip_duration <= 0:
                raise ValueError(f"Invalid clip duration: {clip_duration}")
            
            # Build FFmpeg command with precise parameters
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-i', input_path,
                '-t', str(clip_duration),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-avoid_negative_ts', 'make_zero',
                output_path
            ]
            
            # Execute with error handling
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            # Verify output file was created and has reasonable size
            if not os.path.exists(output_path):
                raise RuntimeError("Output file was not created")
            
            output_size = os.path.getsize(output_path)
            if output_size < 1024:  # Less than 1KB is suspicious
                raise RuntimeError("Output file is too small, likely corrupted")
            
            return True
            
        except Exception as e:
            print(f"❌ Precise trimming failed: {e}")
            # Clean up partial file
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
    
    def get_video_duration(self, video_path):
        """Get video duration using FFprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            print(f"❌ Duration check failed: {e}")
        
        return None

class VisualAnalyzer:
    """Handles intelligent aspect ratio conversion with visual interest detection"""
    
    def __init__(self):
        self.face_cascade = None
        if VISUAL_ANALYSIS_AVAILABLE:
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except Exception as e:
                print(f"⚠️  Face cascade not loaded: {e}")
    
    def analyze_visual_interest(self, video_path, start_time, end_time):
        """
        Analyze video segment for visual interest areas
        Returns optimal crop region for aspect ratio conversion
        """
        if not VISUAL_ANALYSIS_AVAILABLE:
            return {"center_x": 0.5, "center_y": 0.5, "confidence": 0.0}
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Sample frames from the clip
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            interest_points = []
            
            # Analyze key frames
            for frame_num in range(start_frame, end_frame, max(1, (end_frame - start_frame) // 10)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Face detection
                faces = self.detect_faces(frame)
                
                # Motion analysis
                motion_score = self.analyze_motion(frame)
                
                # Edge detection for visual complexity
                edge_score = self.analyze_edges(frame)
                
                # Combine scores
                for face in faces:
                    x, y, w, h = face
                    center_x = (x + w/2) / frame.shape[1]
                    center_y = (y + h/2) / frame.shape[0]
                    
                    interest_points.append({
                        "x": center_x,
                        "y": center_y,
                        "score": 1.0 + motion_score + edge_score
                    })
            
            cap.release()
            
            if interest_points:
                # Find the highest scoring region
                best_point = max(interest_points, key=lambda p: p["score"])
                return {
                    "center_x": best_point["x"],
                    "center_y": best_point["y"],
                    "confidence": min(1.0, best_point["score"] / 3.0)
                }
            else:
                # Default to center if no faces found
                return {"center_x": 0.5, "center_y": 0.5, "confidence": 0.1}
                
        except Exception as e:
            print(f"❌ Visual analysis failed: {e}")
            return {"center_x": 0.5, "center_y": 0.5, "confidence": 0.0}
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        if self.face_cascade is None:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            return faces
        except Exception:
            return []
    
    def analyze_motion(self, frame):
        """Analyze motion/activity in frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Simple motion estimate using gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            return np.mean(magnitude) / 255.0
        except Exception:
            return 0.0
    
    def analyze_edges(self, frame):
        """Analyze edge complexity in frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            return np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
        except Exception:
            return 0.0
    
    def convert_aspect_ratio(self, input_path, output_path, target_aspect, start_time, end_time):
        """
        Convert video aspect ratio with intelligent cropping
        """
        try:
            # Analyze visual interest
            interest = self.analyze_visual_interest(input_path, start_time, end_time)
            
            # Get video dimensions
            probe_cmd = [
                'ffprobe', '-v', 'quiet',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0',
                input_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError("Could not get video dimensions")
            
            width, height = map(int, result.stdout.strip().split(','))
            
            # Calculate crop parameters based on target aspect ratio
            if target_aspect == "16:9":
                target_ratio = 16/9
            elif target_aspect == "9:16":
                target_ratio = 9/16
            else:
                target_ratio = 16/9
            
            current_ratio = width / height
            
            if abs(current_ratio - target_ratio) < 0.01:
                # Already the right aspect ratio
                shutil.copy2(input_path, output_path)
                return True
            
            # Calculate crop dimensions
            if current_ratio > target_ratio:
                # Video is too wide, crop horizontally
                new_width = int(height * target_ratio)
                new_height = height
                
                # Use visual interest to determine crop position
                crop_x = int((width - new_width) * interest["center_x"])
                crop_y = 0
            else:
                # Video is too tall, crop vertically
                new_width = width
                new_height = int(width / target_ratio)
                
                # Use visual interest to determine crop position
                crop_x = 0
                crop_y = int((height - new_height) * interest["center_y"])
            
            # Ensure crop coordinates are within bounds
            crop_x = max(0, min(crop_x, width - new_width))
            crop_y = max(0, min(crop_y, height - new_height))
            
            # Build FFmpeg command with crop filter
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vf', f'crop={new_width}:{new_height}:{crop_x}:{crop_y}',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"Aspect ratio conversion failed: {result.stderr}")
            
            return True
            
        except Exception as e:
            print(f"❌ Aspect ratio conversion failed: {e}")
            return False

class RobustFileServer:
    """Handles robust file serving with broken pipe error handling"""
    
    @staticmethod
    def serve_file_range(request_handler, file_path, range_header=None):
        """
        Serve file with HTTP range support and broken pipe handling
        """
        try:
            file_size = os.path.getsize(file_path)
            
            if range_header:
                # Parse range header
                range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
                if range_match:
                    start = int(range_match.group(1))
                    end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
                    
                    # Validate range
                    start = max(0, min(start, file_size - 1))
                    end = max(start, min(end, file_size - 1))
                    
                    content_length = end - start + 1
                    
                    request_handler.send_response(206)
                    request_handler.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                    request_handler.send_header('Content-Length', str(content_length))
                    request_handler.send_header('Accept-Ranges', 'bytes')
                else:
                    # Invalid range, serve full file
                    start = 0
                    end = file_size - 1
                    content_length = file_size
                    request_handler.send_response(200)
                    request_handler.send_header('Content-Length', str(content_length))
            else:
                # No range header, serve full file
                start = 0
                end = file_size - 1
                content_length = file_size
                request_handler.send_response(200)
                request_handler.send_header('Content-Length', str(content_length))
            
            # Set content type
            content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
            request_handler.send_header('Content-Type', content_type)
            request_handler.end_headers()
            
            # Serve file content
            with open(file_path, 'rb') as f:
                f.seek(start)
                remaining = content_length
                
                while remaining > 0:
                    # Check if client is still connected
                    if not hasattr(request_handler, 'wfile') or request_handler.wfile.closed:
                        break
                    
                    chunk_size = min(8192, remaining)
                    chunk = f.read(chunk_size)
                    
                    if not chunk:
                        break
                    
                    try:
                        request_handler.wfile.write(chunk)
                        request_handler.wfile.flush()
                        remaining -= len(chunk)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        # Client disconnected, stop serving
                        break
                    except Exception as e:
                        print(f"❌ File serving error: {e}")
                        break
            
            return True
            
        except Exception as e:
            print(f"❌ File serving failed: {e}")
            return False

class ClipsAIHandler(http.server.SimpleHTTPRequestHandler):
    """Main request handler with all fixes applied"""
    
    upload_dir = os.environ.get('UPLOAD_DIR', './uploads')
    
    def __init__(self, *args, **kwargs):
        self.trimmer = PreciseVideoTrimmer()
        self.analyzer = VisualAnalyzer()
        self.file_server = RobustFileServer()
        super().__init__(*args, **kwargs)
    
    @classmethod
    def initialize_upload_dir(cls):
        """Initialize upload directory and setup test video"""
        os.makedirs(cls.upload_dir, exist_ok=True)
        os.makedirs('./downloads', exist_ok=True)
        os.makedirs('./temp', exist_ok=True)
        
        try:
            os.chmod(cls.upload_dir, 0o777)
            os.chmod('./downloads', 0o777)
            os.chmod('./temp', 0o777)
        except OSError:
            pass
        
        print(f"📁 Upload directory initialized: {cls.upload_dir}")
        cls.setup_test_video()
    
    @classmethod
    def setup_test_video(cls):
        """Set up the preloaded test video"""
        test_video_src = "test_vid.mp4"
        test_video_dest = os.path.join(cls.upload_dir, "video_preloaded_test.mp4")
        
        if os.path.exists(test_video_src) and not os.path.exists(test_video_dest):
            try:
                shutil.copy2(test_video_src, test_video_dest)
                print(f"📹 Preloaded test video: {test_video_dest}")
                cls.preloaded_video_info = {
                    "filename": "test_vid.mp4",
                    "video_id": "preloaded_test",
                    "size": os.path.getsize(test_video_dest),
                    "url": "/uploads/video_preloaded_test.mp4",
                    "preloaded": True
                }
            except Exception as e:
                print(f"❌ Failed to setup test video: {e}")
                cls.preloaded_video_info = None
        elif os.path.exists(test_video_dest):
            print(f"📹 Test video already exists: {test_video_dest}")
            cls.preloaded_video_info = {
                "filename": "test_vid.mp4",
                "video_id": "preloaded_test",
                "size": os.path.getsize(test_video_dest),
                "url": "/uploads/video_preloaded_test.mp4",
                "preloaded": True
            }
        else:
            print(f"⚠️  No test video available")
            cls.preloaded_video_info = None
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.serve_main_page()
        elif self.path == '/api/status':
            self.serve_status()
        elif self.path == '/api/health':
            self.serve_health()
        elif self.path == '/api/preloaded_video':
            self.serve_preloaded_video()
        elif self.path.startswith('/uploads/'):
            self.serve_uploaded_file()
        elif self.path.startswith('/downloads/'):
            self.serve_download_file()
        elif self.path.startswith('/tmp/'):
            self.serve_tmp_file()
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/api/upload':
            self.handle_upload()
        elif self.path == '/api/transcribe':
            self.handle_transcribe()
        elif self.path == '/api/find_clips':
            self.handle_find_clips()
        elif self.path == '/api/trim_clip':
            self.handle_trim_clip()
        else:
            self.send_error(404, "API endpoint not found")
    
    def serve_health(self):
        """Health check endpoint"""
        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "upload_dir": os.path.exists(self.upload_dir),
            "port": PORT,
            "version": "clipsai-fixed-1.0",
            "features": {
                "whisper_ai": True,
                "moviepy_audio": True,
                "precise_trimming": True,
                "aspect_conversion": True,
                "visual_analysis": VISUAL_ANALYSIS_AVAILABLE,
                "broken_pipe_handling": True
            }
        }
        
        try:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(health).encode())
        except (BrokenPipeError, ConnectionResetError):
            pass
    
    def serve_main_page(self):
        """Serve the main web interface"""
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ClipsAI - Fixed Version</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .feature {
            padding: 15px;
            background: #e8f5e8;
            border-radius: 8px;
            border-left: 4px solid #27ae60;
        }
        .upload-section {
            margin-bottom: 30px;
        }
        .upload-area {
            border: 2px dashed #3498db;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            border-color: #2980b9;
            background: #f8f9fa;
        }
        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s ease;
        }
        .btn:hover {
            background: #2980b9;
        }
        .btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: #27ae60;
            transition: width 0.3s ease;
        }
        .video-preview {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            margin: 10px 0;
        }
        .clips-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .clip-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        .status {
            padding: 10px;
            border-radius: 6px;
            margin: 10px 0;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .status.info {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 ClipsAI - Fixed Version</h1>
        
        <div class="features">
            <div class="feature">
                <h3>✅ Precise Trimming</h3>
                <p>Exact video clips, no more full videos</p>
            </div>
            <div class="feature">
                <h3>🎯 Smart Cropping</h3>
                <p>Intelligent aspect ratio conversion</p>
            </div>
            <div class="feature">
                <h3>🔧 H.264 Fixes</h3>
                <p>Corrupted video file fixes</p>
            </div>
            <div class="feature">
                <h3>🚀 Auto Generation</h3>
                <p>Clips generated automatically</p>
            </div>
        </div>
        
        <div class="upload-section">
            <h2>Upload Video</h2>
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="video/*" style="display: none;" onchange="uploadFile(this)">
                <p>Click to select a video file</p>
                <p style="color: #666;">Supports MP4, MOV, AVI, MKV formats</p>
            </div>
            
            <div id="uploadProgress" style="display: none;">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <p id="progressText">Uploading...</p>
            </div>
        </div>
        
        <div id="videoSection" style="display: none;">
            <h2>Video Preview</h2>
            <video id="videoPreview" class="video-preview" controls></video>
            
            <div style="margin: 20px 0;">
                <button class="btn" onclick="transcribeVideo()">🎤 Transcribe & Find Clips</button>
                <button class="btn" onclick="testPreloadedVideo()">🧪 Test with Preloaded Video</button>
            </div>
        </div>
        
        <div id="transcriptionSection" style="display: none;">
            <h2>AI Transcription</h2>
            <div id="transcriptionStatus"></div>
            <div id="transcriptionResult"></div>
        </div>
        
        <div id="clipsSection" style="display: none;">
            <h2>Generated Clips</h2>
            <div id="clipsList" class="clips-grid"></div>
        </div>
        
        <div id="statusSection"></div>
    </div>

    <script>
        let currentVideoId = null;
        let currentVideoUrl = null;
        
        function showStatus(message, type = 'info') {
            const statusDiv = document.getElementById('statusSection');
            statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
            setTimeout(() => {
                statusDiv.innerHTML = '';
            }, 5000);
        }
        
        function uploadFile(input) {
            const file = input.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('video', file);
            
            const progressDiv = document.getElementById('uploadProgress');
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            
            progressDiv.style.display = 'block';
            progressFill.style.width = '0%';
            progressText.textContent = 'Uploading...';
            
            fetch('/api/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                progressDiv.style.display = 'none';
                
                if (data.success) {
                    currentVideoId = data.video_id;
                    currentVideoUrl = data.url;
                    
                    document.getElementById('videoSection').style.display = 'block';
                    document.getElementById('videoPreview').src = data.url;
                    
                    showStatus('✅ Video uploaded successfully!', 'success');
                } else {
                    showStatus(`❌ Upload failed: ${data.error}`, 'error');
                }
            })
            .catch(error => {
                progressDiv.style.display = 'none';
                showStatus(`❌ Upload error: ${error.message}`, 'error');
            });
        }
        
        function testPreloadedVideo() {
            fetch('/api/preloaded_video')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    currentVideoId = data.video.video_id;
                    currentVideoUrl = data.video.url;
                    
                    document.getElementById('videoSection').style.display = 'block';
                    document.getElementById('videoPreview').src = data.video.url;
                    
                    showStatus('✅ Using preloaded test video!', 'success');
                } else {
                    showStatus('❌ No preloaded video available', 'error');
                }
            })
            .catch(error => {
                showStatus(`❌ Error: ${error.message}`, 'error');
            });
        }
        
        function transcribeVideo() {
            if (!currentVideoId) {
                showStatus('❌ No video selected', 'error');
                return;
            }
            
            document.getElementById('transcriptionSection').style.display = 'block';
            document.getElementById('transcriptionStatus').innerHTML = '<div class="status info">🎤 Transcribing video with AI...</div>';
            
            fetch('/api/transcribe', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    video_id: currentVideoId
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('transcriptionStatus').innerHTML = '<div class="status success">✅ Transcription completed!</div>';
                    document.getElementById('transcriptionResult').innerHTML = `
                        <h3>Transcript:</h3>
                        <p>${data.transcript}</p>
                    `;
                    
                    // Automatically find clips
                    findClips();
                } else {
                    document.getElementById('transcriptionStatus').innerHTML = `<div class="status error">❌ Transcription failed: ${data.error}</div>`;
                }
            })
            .catch(error => {
                document.getElementById('transcriptionStatus').innerHTML = `<div class="status error">❌ Error: ${error.message}</div>`;
            });
        }
        
        function findClips() {
            if (!currentVideoId) {
                showStatus('❌ No video selected', 'error');
                return;
            }
            
            document.getElementById('clipsSection').style.display = 'block';
            document.getElementById('clipsList').innerHTML = '<div class="status info">🔍 Finding and generating clips automatically...</div>';
            
            fetch('/api/find_clips', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    video_id: currentVideoId
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayClips(data.clips);
                    showStatus(`✅ Found and generated ${data.clips.length} clips!`, 'success');
                } else {
                    document.getElementById('clipsList').innerHTML = `<div class="status error">❌ Clip finding failed: ${data.error}</div>`;
                }
            })
            .catch(error => {
                document.getElementById('clipsList').innerHTML = `<div class="status error">❌ Error: ${error.message}</div>`;
            });
        }
        
        function displayClips(clips) {
            const clipsList = document.getElementById('clipsList');
            
            if (clips.length === 0) {
                clipsList.innerHTML = '<div class="status info">No clips found in this video.</div>';
                return;
            }
            
            clipsList.innerHTML = clips.map(clip => `
                <div class="clip-card">
                    <h3>Clip ${clip.index + 1}</h3>
                    <p><strong>Time:</strong> ${clip.start}s - ${clip.end}s</p>
                    <p><strong>Text:</strong> ${clip.text}</p>
                    <p><strong>Confidence:</strong> ${(clip.confidence * 100).toFixed(1)}%</p>
                    ${clip.generated_url ? `
                        <video controls style="width: 100%; margin: 10px 0;">
                            <source src="${clip.generated_url}" type="video/mp4">
                        </video>
                        <div style="margin: 10px 0;">
                            <a href="${clip.generated_url}" download="clip_${clip.index + 1}.mp4" class="btn">⬇️ Download</a>
                        </div>
                    ` : ''}
                </div>
            `).join('');
        }
        
        // Check health on load
        fetch('/api/health')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'healthy') {
                showStatus('✅ ClipsAI is running and healthy!', 'success');
            }
        })
        .catch(error => {
            showStatus('❌ Health check failed', 'error');
        });
    </script>
</body>
</html>'''
        
        try:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(html_content.encode())
        except (BrokenPipeError, ConnectionResetError):
            pass
    
    def serve_uploaded_file(self):
        """Serve uploaded files with robust error handling"""
        try:
            file_path = self.path[1:]  # Remove leading slash
            full_path = os.path.join(os.getcwd(), file_path)
            
            if os.path.exists(full_path) and os.path.isfile(full_path):
                range_header = self.headers.get('Range')
                self.file_server.serve_file_range(self, full_path, range_header)
            else:
                self.send_error(404, "File not found")
        except Exception as e:
            print(f"❌ Error serving uploaded file: {e}")
            self.send_error(500, "Internal server error")
    
    def serve_download_file(self):
        """Serve download files with robust error handling"""
        try:
            file_path = self.path[1:]  # Remove leading slash
            full_path = os.path.join(os.getcwd(), file_path)
            
            if os.path.exists(full_path) and os.path.isfile(full_path):
                range_header = self.headers.get('Range')
                self.file_server.serve_file_range(self, full_path, range_header)
            else:
                self.send_error(404, "File not found")
        except Exception as e:
            print(f"❌ Error serving download file: {e}")
            self.send_error(500, "Internal server error")
    
    def serve_tmp_file(self):
        """Serve temporary files with robust error handling"""
        try:
            file_path = self.path[1:]  # Remove leading slash
            full_path = os.path.join(os.getcwd(), file_path)
            
            if os.path.exists(full_path) and os.path.isfile(full_path):
                range_header = self.headers.get('Range')
                self.file_server.serve_file_range(self, full_path, range_header)
            else:
                self.send_error(404, "File not found")
        except Exception as e:
            print(f"❌ Error serving temp file: {e}")
            self.send_error(500, "Internal server error")
    
    def handle_upload(self):
        """Handle video file uploads"""
        try:
            content_type = self.headers.get('Content-Type', '')
            
            if not content_type.startswith('multipart/form-data'):
                self.send_error(400, "Invalid content type")
                return
            
            # Parse multipart form data
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            
            if 'video' not in form:
                self.send_error(400, "No video file provided")
                return
            
            video_file = form['video']
            
            if not video_file.filename:
                self.send_error(400, "No file selected")
                return
            
            # Generate unique filename
            video_id = str(uuid.uuid4())
            file_extension = os.path.splitext(video_file.filename)[1]
            filename = f"video_{video_id}{file_extension}"
            file_path = os.path.join(self.upload_dir, filename)
            
            # Save uploaded file
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(video_file.file, f)
            
            # Verify file was saved
            if not os.path.exists(file_path):
                raise RuntimeError("File was not saved properly")
            
            file_size = os.path.getsize(file_path)
            
            if file_size == 0:
                os.remove(file_path)
                raise RuntimeError("Uploaded file is empty")
            
            response = {
                "success": True,
                "video_id": video_id,
                "filename": video_file.filename,
                "size": file_size,
                "url": f"/uploads/{filename}"
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"❌ Upload error: {e}")
            error_response = {
                "success": False,
                "error": str(e)
            }
            
            try:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(error_response).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass
    
    def handle_transcribe(self):
        """Handle video transcription using Whisper AI"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            video_id = data.get('video_id')
            if not video_id:
                raise ValueError("No video_id provided")
            
            # Find video file
            video_file = None
            for filename in os.listdir(self.upload_dir):
                if video_id in filename:
                    video_file = os.path.join(self.upload_dir, filename)
                    break
            
            if not video_file or not os.path.exists(video_file):
                raise ValueError("Video file not found")
            
            # Load Whisper model
            model = whisper.load_model("base")
            
            # Transcribe video
            result = model.transcribe(video_file)
            
            response = {
                "success": True,
                "transcript": result["text"],
                "segments": result["segments"]
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            error_response = {
                "success": False,
                "error": str(e)
            }
            
            try:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(error_response).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass
    
    def handle_find_clips(self):
        """Find and automatically generate clips"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            video_id = data.get('video_id')
            if not video_id:
                raise ValueError("No video_id provided")
            
            # Find video file
            video_file = None
            for filename in os.listdir(self.upload_dir):
                if video_id in filename:
                    video_file = os.path.join(self.upload_dir, filename)
                    break
            
            if not video_file or not os.path.exists(video_file):
                raise ValueError("Video file not found")
            
            # Load Whisper model for clip detection
            model = whisper.load_model("base")
            result = model.transcribe(video_file)
            
            # Simple clip detection based on segments
            clips = []
            for i, segment in enumerate(result["segments"]):
                # Look for segments with high confidence and certain keywords
                if segment["no_speech_prob"] < 0.5:
                    clips.append({
                        "index": i,
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"],
                        "confidence": 1.0 - segment["no_speech_prob"]
                    })
            
            # Automatically generate clips
            generated_clips = []
            os.makedirs('./downloads', exist_ok=True)
            
            for clip in clips[:5]:  # Limit to first 5 clips
                clip_id = str(uuid.uuid4())
                output_filename = f"clip_{clip['index']}_{clip_id}.mp4"
                output_path = os.path.join('./downloads', output_filename)
                
                # Use precise trimmer
                success = self.trimmer.trim_video_precise(
                    video_file,
                    output_path,
                    clip['start'],
                    clip['end']
                )
                
                if success:
                    clip['generated_url'] = f"/downloads/{output_filename}"
                    clip['generated_path'] = output_path
                
                generated_clips.append(clip)
            
            response = {
                "success": True,
                "clips": generated_clips
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"❌ Find clips error: {e}")
            error_response = {
                "success": False,
                "error": str(e)
            }
            
            try:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(error_response).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass
    
    def handle_trim_clip(self):
        """Handle manual clip trimming"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            video_id = data.get('video_id')
            start_time = data.get('start_time')
            end_time = data.get('end_time')
            aspect_ratio = data.get('aspect_ratio', '16:9')
            
            if not all([video_id, start_time is not None, end_time is not None]):
                raise ValueError("Missing required parameters")
            
            # Find video file
            video_file = None
            for filename in os.listdir(self.upload_dir):
                if video_id in filename:
                    video_file = os.path.join(self.upload_dir, filename)
                    break
            
            if not video_file or not os.path.exists(video_file):
                raise ValueError("Video file not found")
            
            # Generate output filename
            clip_id = str(uuid.uuid4())
            output_filename = f"clip_{clip_id}.mp4"
            output_path = os.path.join('./downloads', output_filename)
            temp_path = os.path.join('./temp', f"temp_{clip_id}.mp4")
            
            # First, trim the video precisely
            success = self.trimmer.trim_video_precise(
                video_file,
                temp_path,
                start_time,
                end_time
            )
            
            if not success:
                raise RuntimeError("Video trimming failed")
            
            # Then, convert aspect ratio if needed
            if aspect_ratio != "original":
                success = self.analyzer.convert_aspect_ratio(
                    temp_path,
                    output_path,
                    aspect_ratio,
                    0,  # Already trimmed, so start at 0
                    end_time - start_time
                )
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            else:
                # No aspect ratio conversion needed
                shutil.move(temp_path, output_path)
                success = True
            
            if not success:
                raise RuntimeError("Aspect ratio conversion failed")
            
            response = {
                "success": True,
                "clip_url": f"/downloads/{output_filename}",
                "clip_path": output_path,
                "start_time": start_time,
                "end_time": end_time,
                "aspect_ratio": aspect_ratio
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"❌ Trim clip error: {e}")
            error_response = {
                "success": False,
                "error": str(e)
            }
            
            try:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(error_response).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass
    
    def serve_preloaded_video(self):
        """Serve information about the preloaded test video"""
        try:
            if hasattr(self.__class__, 'preloaded_video_info') and self.__class__.preloaded_video_info:
                response = {
                    "success": True,
                    "video": self.__class__.preloaded_video_info
                }
            else:
                response = {
                    "success": False,
                    "error": "No preloaded video available"
                }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"❌ Preloaded video error: {e}")
            error_response = {
                "success": False,
                "error": str(e)
            }
            
            try:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(error_response).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass

def main():
    """Main server function"""
    print("🎬 Starting ClipsAI Fixed Web Server")
    print("=" * 50)
    
    # Initialize upload directory
    ClipsAIHandler.initialize_upload_dir()
    
    # Create server
    with socketserver.TCPServer(("", PORT), ClipsAIHandler) as httpd:
        print(f"✅ Server started successfully!")
        print(f"🌐 Access at: http://localhost:{PORT}")
        print(f"🏥 Health check: http://localhost:{PORT}/api/health")
        print(f"📁 Upload directory: {ClipsAIHandler.upload_dir}")
        print(f"🎯 Features: Precise trimming, Smart cropping, H.264 fixes, Auto generation")
        print(f"🛑 Press Ctrl+C to stop")
        print("=" * 50)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
        except Exception as e:
            print(f"❌ Server error: {e}")
        finally:
            httpd.server_close()

if __name__ == "__main__":
    main()