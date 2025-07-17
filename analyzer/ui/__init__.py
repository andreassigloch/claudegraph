"""
Code Architecture Analyzer - Web UI Package

This package provides a Flask-based web interface for reviewing and classifying
ambiguous code patterns identified during analysis.

The UI consists of:
- Dashboard: Analysis progress and statistics
- Review Interface: Manual classification of code patterns  
- Upload Interface: Project loading and configuration
- Results Display: Final analysis output and export options

Usage:
    from analyzer.ui import create_app
    
    app = create_app()
    app.run(debug=True)
"""

from .app import create_app, run_development_server

__version__ = "1.0.0"
__author__ = "Code Architecture Analyzer Team"

# Export main application factory
__all__ = ['create_app', 'run_development_server']

def create_ui_app(config=None, debug=False):
    """
    Convenience function to create and configure the Flask UI application.
    
    Args:
        config (dict, optional): Configuration dictionary
        debug (bool): Enable debug mode
        
    Returns:
        Flask: Configured Flask application instance
    """
    app_config = config or {}
    
    if debug:
        app_config.update({
            'DEBUG': True,
            'TESTING': False,
            'WTF_CSRF_ENABLED': False  # Disable CSRF for development
        })
    
    return create_app(app_config)

def run_ui_server(host='127.0.0.1', port=5000, debug=True, config=None):
    """
    Run the UI development server.
    
    Args:
        host (str): Host address to bind to
        port (int): Port number to listen on
        debug (bool): Enable debug mode
        config (dict, optional): Configuration dictionary
    """
    app = create_ui_app(config=config, debug=debug)
    
    print(f"Starting Code Architecture Analyzer UI server...")
    print(f"Server running at: http://{host}:{port}")
    print(f"Debug mode: {'ON' if debug else 'OFF'}")
    
    app.run(host=host, port=port, debug=debug)