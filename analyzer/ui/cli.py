#!/usr/bin/env python3
"""
CLI entry point for the Code Architecture Analyzer Web UI.

This module provides a command-line interface for starting the Flask web server
that hosts the review interface for manual classification of code patterns.

Usage:
    python -m analyzer.ui.cli [OPTIONS]
    
    # Or directly:
    python analyzer/ui/cli.py --host 0.0.0.0 --port 8080 --debug
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add the analyzer package to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from analyzer.ui import create_ui_app, run_ui_server
except ImportError as e:
    print(f"Error importing analyzer modules: {e}")
    print("Make sure you're running from the correct directory and all dependencies are installed.")
    sys.exit(1)

def setup_logging(level='INFO'):
    """Configure logging for the UI server."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Reduce Flask/Werkzeug logging in production
    if level.upper() != 'DEBUG':
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

def validate_args(args):
    """Validate command-line arguments."""
    errors = []
    
    # Validate port range
    if not (1 <= args.port <= 65535):
        errors.append(f"Port must be between 1 and 65535, got {args.port}")
    
    # Validate host format (basic check)
    if not args.host.replace('.', '').replace(':', '').isalnum() and args.host not in ['localhost', '0.0.0.0']:
        # Allow simple validation - more complex validation would require ipaddress module
        pass
    
    # Check if config file exists
    if args.config and not Path(args.config).exists():
        errors.append(f"Configuration file not found: {args.config}")
    
    return errors

def load_config(config_path):
    """Load configuration from file."""
    if not config_path:
        return {}
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        print("Warning: PyYAML not installed, cannot load YAML config files")
        return {}
    except Exception as e:
        print(f"Warning: Could not load config file {config_path}: {e}")
        return {}

def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Code Architecture Analyzer - Web UI Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Start with default settings
  %(prog)s --port 8080              # Use custom port
  %(prog)s --host 0.0.0.0 --debug   # Bind to all interfaces in debug mode
  %(prog)s --config config.yaml     # Use custom configuration file
  %(prog)s --no-debug --log-level WARNING  # Production mode with minimal logging

For more information, visit: https://github.com/your-repo/grphzer2
        """
    )
    
    # Server configuration
    server_group = parser.add_argument_group('Server Configuration')
    server_group.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host address to bind to (default: 127.0.0.1)'
    )
    server_group.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port number to listen on (default: 5000)'
    )
    
    # Debug and logging
    debug_group = parser.add_argument_group('Debug and Logging')
    debug_group.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (auto-reload, detailed errors)'
    )
    debug_group.add_argument(
        '--no-debug',
        action='store_true',
        help='Explicitly disable debug mode'
    )
    debug_group.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    # Configuration
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config',
        help='Path to YAML configuration file'
    )
    config_group.add_argument(
        '--secret-key',
        help='Flask secret key (overrides config file)'
    )
    
    # Application behavior
    app_group = parser.add_argument_group('Application Behavior')
    app_group.add_argument(
        '--auto-open',
        action='store_true',
        help='Automatically open browser after starting server'
    )
    app_group.add_argument(
        '--check-port',
        action='store_true',
        help='Check if port is available before starting'
    )
    
    # Information
    info_group = parser.add_argument_group('Information')
    info_group.add_argument(
        '--version',
        action='version',
        version='Code Architecture Analyzer UI v1.0.0'
    )
    
    return parser

def check_port_available(host, port):
    """Check if the specified port is available."""
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except OSError:
        return False

def open_browser(url):
    """Open the default web browser to the specified URL."""
    import webbrowser
    import threading
    import time
    
    def delayed_open():
        time.sleep(1.5)  # Give the server time to start
        try:
            webbrowser.open(url)
            print(f"Opened browser: {url}")
        except Exception as e:
            print(f"Could not open browser: {e}")
    
    threading.Thread(target=delayed_open, daemon=True).start()

def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    errors = validate_args(args)
    if errors:
        print("Error(s):")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Determine debug mode
    debug_mode = args.debug
    if args.no_debug:
        debug_mode = False
    
    # Check port availability
    if args.check_port and not check_port_available(args.host, args.port):
        print(f"Error: Port {args.port} is already in use on {args.host}")
        print("Try using a different port with --port or stop the service using that port")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.secret_key:
        config['SECRET_KEY'] = args.secret_key
    
    # Set debug mode in config
    config['DEBUG'] = debug_mode
    
    try:
        # Print startup information
        print("=" * 60)
        print("Code Architecture Analyzer - Web UI Server")
        print("=" * 60)
        print(f"Host: {args.host}")
        print(f"Port: {args.port}")
        print(f"Debug Mode: {'ON' if debug_mode else 'OFF'}")
        print(f"Log Level: {args.log_level}")
        if args.config:
            print(f"Config File: {args.config}")
        print("=" * 60)
        
        # Open browser if requested
        if args.auto_open:
            url = f"http://{args.host}:{args.port}"
            open_browser(url)
        
        # Start the server
        logger.info("Starting UI server...")
        run_ui_server(
            host=args.host,
            port=args.port,
            debug=debug_mode,
            config=config
        )
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
        logger.info("Server shutdown requested by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nError starting server: {e}")
        logger.error(f"Server startup failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()