"""
Flask application for the Code Architecture Analyzer Review Interface.

This module provides a web-based interface for reviewing and classifying
ambiguous code patterns identified during analysis.
"""

import os
import logging
from flask import Flask, session, request, jsonify, render_template
from datetime import datetime, timedelta
import uuid
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(config=None):
    """Create and configure the Flask application."""
    
    app = Flask(__name__)
    
    # Basic configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', str(uuid.uuid4()))
    app.config['SESSION_PERMANENT'] = False
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
    
    # Custom configuration
    if config:
        app.config.update(config)
    
    # Set up session configuration
    @app.before_request
    def before_request():
        """Initialize session data before each request."""
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
            session['created_at'] = datetime.utcnow().isoformat()
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        if request.is_json:
            return jsonify({'error': 'Not found', 'message': str(error)}), 404
        return render_template('error.html', 
                             error_code=404, 
                             error_message="Page not found"), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        logger.error(f"Internal server error: {error}")
        if request.is_json:
            return jsonify({'error': 'Internal server error', 'message': str(error)}), 500
        return render_template('error.html', 
                             error_code=500, 
                             error_message="Internal server error"), 500
    
    @app.errorhandler(413)
    def too_large(error):
        """Handle file too large errors."""
        if request.is_json:
            return jsonify({'error': 'File too large', 'message': 'Maximum file size is 16MB'}), 413
        return render_template('error.html', 
                             error_code=413, 
                             error_message="File too large (max 16MB)"), 413
    
    # Template filters
    @app.template_filter('datetime_format')
    def datetime_format(value, format='%Y-%m-%d %H:%M:%S'):
        """Format datetime for display."""
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                return value
        return value.strftime(format) if value else ''
    
    @app.template_filter('percentage')
    def percentage(value, total):
        """Calculate percentage for progress bars."""
        if total == 0:
            return 0
        return round((value / total) * 100, 1)
    
    @app.template_global()
    def moment():
        """Get current datetime for template use."""
        return datetime.now()
    
    @app.template_global()
    def format_datetime(dt, format_str='%Y-%m-%d %H:%M:%S'):
        """Format datetime with custom format."""
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
            except ValueError:
                return dt
        return dt.strftime(format_str) if dt else ''
    
    # Context processors
    @app.context_processor
    def inject_global_vars():
        """Inject global variables into templates."""
        return {
            'app_name': 'Code Architecture Analyzer',
            'version': '1.0.0',
            'current_year': datetime.now().year,
            'now': datetime.now()
        }
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        """Simple health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session.get('session_id'),
            'uptime': 'running'
        })
    
    # Session info endpoint
    @app.route('/api/session')
    def session_info():
        """Get current session information."""
        return jsonify({
            'session_id': session.get('session_id'),
            'created_at': session.get('created_at'),
            'progress': session.get('review_progress', {}),
            'active': True
        })
    
    # Register blueprints (will be created in routes.py)
    try:
        from .routes import main_bp, api_bp
        from .api_analyzer import api_analyzer_bp
        from .dead_code_routes import dead_code_bp
        app.register_blueprint(main_bp)
        app.register_blueprint(api_bp, url_prefix='/api')
        app.register_blueprint(api_analyzer_bp, url_prefix='/api')
        app.register_blueprint(dead_code_bp)
        logger.info("Blueprints registered successfully")
    except ImportError as e:
        logger.warning(f"Could not import routes: {e}")
        
        # Fallback route if routes module is not yet created
        @app.route('/')
        def index():
            return '''
            <html>
                <head><title>Code Architecture Analyzer</title></head>
                <body>
                    <h1>Code Architecture Analyzer - Review Interface</h1>
                    <p>Setting up the review interface...</p>
                    <p>Session ID: {}</p>
                </body>
            </html>
            '''.format(session.get('session_id'))
    
    return app

# Application factory
def run_development_server(host='127.0.0.1', port=5000, debug=True):
    """Run the development server."""
    app = create_app()
    logger.info(f"Starting development server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    # Development server
    run_development_server()