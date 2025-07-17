"""
Flask routes for the Code Architecture Analyzer Review Interface.

This module defines all the web routes and API endpoints for the review interface,
including the dashboard, review workflow, and progress tracking.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
import tempfile
import os
import uuid
from werkzeug.datastructures import FileStorage

logger = logging.getLogger(__name__)

# Create blueprints
main_bp = Blueprint('main', __name__)
api_bp = Blueprint('api', __name__)


@main_bp.route('/')
def index():
    """Main index page - redirect to analyzer."""
    return redirect(url_for('main.analyzer_page'))


@main_bp.route('/viewer')
@main_bp.route('/viewer/<analysis_data>')
def graph_viewer(analysis_data=None):
    """Integrated graph visualization interface."""
    return render_template('viewer.html', analysis_data=analysis_data)

@main_bp.route('/logical-viewer')
@main_bp.route('/logical-viewer/<analysis_data>')
def logical_viewer(analysis_data=None):
    """Logical dependency visualization interface."""
    return render_template('logical_viewer.html', analysis_data=analysis_data)

@main_bp.route('/analyzer')
def analyzer_page():
    """Project analyzer with upload and real-time analysis."""
    return render_template('analyzer.html')

# Analysis API Routes
# Note: Main analysis endpoints moved to api_analyzer.py

# Status and result endpoints removed - now handled by api_analyzer.py

