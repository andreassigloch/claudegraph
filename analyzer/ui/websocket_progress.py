"""
WebSocket support for real-time analysis progress updates.
Uses Flask-SocketIO for WebSocket communication.
"""

import logging
from flask import request
from flask_socketio import SocketIO, emit, join_room, leave_room
from threading import Thread
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Global SocketIO instance
socketio = None

def init_socketio(app):
    """Initialize SocketIO with Flask app."""
    global socketio
    socketio = SocketIO(
        app, 
        cors_allowed_origins="*",
        logger=True,
        engineio_logger=False,
        ping_timeout=60,
        ping_interval=25
    )
    
    # Register event handlers
    register_socketio_events()
    
    return socketio

def register_socketio_events():
    """Register WebSocket event handlers."""
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        logger.info(f"Client connected: {request.sid}")
        emit('connected', {
            'status': 'connected',
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': request.sid
        })
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        logger.info(f"Client disconnected: {request.sid}")
    
    @socketio.on('join_analysis')
    def handle_join_analysis(data):
        """Join a room to receive updates for specific analysis job."""
        job_id = data.get('job_id')
        if job_id:
            join_room(f"analysis_{job_id}")
            logger.info(f"Client {request.sid} joined analysis room: {job_id}")
            emit('joined_analysis', {
                'job_id': job_id,
                'status': 'subscribed',
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            emit('error', {'message': 'job_id required'})
    
    @socketio.on('leave_analysis')
    def handle_leave_analysis(data):
        """Leave analysis room."""
        job_id = data.get('job_id')
        if job_id:
            leave_room(f"analysis_{job_id}")
            logger.info(f"Client {request.sid} left analysis room: {job_id}")
            emit('left_analysis', {
                'job_id': job_id,
                'status': 'unsubscribed'
            })
    
    @socketio.on('ping')
    def handle_ping():
        """Handle ping from client."""
        emit('pong', {'timestamp': datetime.utcnow().isoformat()})

def broadcast_progress(job_id: str, progress: int, message: str, extra_data: dict = None):
    """Broadcast progress update to all clients subscribed to job."""
    if not socketio:
        return
        
    data = {
        'job_id': job_id,
        'progress': progress,
        'message': message,
        'timestamp': datetime.utcnow().isoformat(),
        'type': 'progress'
    }
    
    if extra_data:
        data.update(extra_data)
    
    room = f"analysis_{job_id}"
    socketio.emit('analysis_progress', data, room=room)
    logger.debug(f"Broadcasted progress to room {room}: {progress}% - {message}")

def broadcast_completion(job_id: str, success: bool, result: dict = None, error: str = None):
    """Broadcast analysis completion to all subscribed clients."""
    if not socketio:
        return
        
    data = {
        'job_id': job_id,
        'success': success,
        'timestamp': datetime.utcnow().isoformat(),
        'type': 'completion'
    }
    
    if success and result:
        data['result_available'] = True
        data['stats'] = result.get('metadata', {}).get('analysis_stats', {})
    elif error:
        data['error'] = error
    
    room = f"analysis_{job_id}"
    socketio.emit('analysis_complete', data, room=room)
    logger.info(f"Broadcasted completion to room {room}: success={success}")

def broadcast_error(job_id: str, error: str, fatal: bool = True):
    """Broadcast error to subscribed clients."""
    if not socketio:
        return
        
    data = {
        'job_id': job_id,
        'error': error,
        'fatal': fatal,
        'timestamp': datetime.utcnow().isoformat(),
        'type': 'error'
    }
    
    room = f"analysis_{job_id}"
    socketio.emit('analysis_error', data, room=room)
    logger.warning(f"Broadcasted error to room {room}: {error}")

# Enhanced AnalysisJob class with WebSocket support
class WebSocketAnalysisJob:
    """Analysis job with WebSocket progress broadcasting."""
    
    def __init__(self, job_id: str, project_path: str, llm_provider: str = 'none', original_project_name: str = None):
        from pathlib import Path
        self.job_id = job_id
        self.project_path = project_path
        self.original_project_name = original_project_name or Path(project_path).name
        self.llm_provider = llm_provider
        self.status = 'initializing'
        self.progress = 0
        self.message = 'Initializing analysis...'
        self.result = None
        self.error = None
        self.created_at = datetime.utcnow()
        self.completed_at = None
        self.stats = {}
        self.analysis_file_path = None  # Path to saved analysis JSON file
        self.dead_code_analysis = None  # Dead code analysis data
        
    def update_progress(self, progress: int, message: str, extra_data: dict = None):
        """Update progress and broadcast via WebSocket."""
        self.progress = min(100, max(0, progress))
        self.message = message
        self.status = 'running' if progress < 100 else self.status
        
        # Broadcast via WebSocket
        broadcast_progress(self.job_id, self.progress, self.message, extra_data)
        
    def complete(self, result: dict):
        """Complete job and broadcast success."""
        self.status = 'completed'
        self.progress = 100
        self.message = 'Analysis completed successfully'
        self.result = result
        self.completed_at = datetime.utcnow()
        
        # Broadcast completion
        broadcast_completion(self.job_id, True, result)
        
    def fail(self, error: str):
        """Fail job and broadcast error."""
        self.status = 'failed'
        self.message = f'Analysis failed: {error}'
        self.error = error
        self.completed_at = datetime.utcnow()
        
        # Broadcast error
        broadcast_error(self.job_id, error, fatal=True)
        
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'job_id': self.job_id,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'stats': self.stats,
            'has_result': self.result is not None,
            'error': self.error
        }

# Utility function to test WebSocket functionality
def test_websocket_broadcast():
    """Test WebSocket broadcasting functionality."""
    if not socketio:
        logger.warning("SocketIO not initialized, cannot test broadcast")
        return
        
    def broadcast_test():
        """Broadcast test messages."""
        test_job_id = "test_job_123"
        
        # Simulate progress updates
        for i in range(0, 101, 10):
            broadcast_progress(test_job_id, i, f"Test progress: {i}%")
            time.sleep(0.5)
            
        # Simulate completion
        broadcast_completion(test_job_id, True, {"test": "result"})
    
    # Run in background thread
    thread = Thread(target=broadcast_test)
    thread.daemon = True
    thread.start()
    
    return "Test broadcast started"