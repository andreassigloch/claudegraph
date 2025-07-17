"""
API Wrapper for CLI Analyzer
Provides REST API endpoints that use the existing CLI analyzer without modification.
"""

import os
import json
import tempfile
import shutil
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import threading
import time
import uuid
from datetime import datetime

# Import the existing CLI analyzer (unchanged)
from analyzer.core.flow_based_analyzer import FlowBasedAnalyzer
from analyzer.cli.config import load_unified_config

# Import WebSocket support
try:
    from .websocket_progress import WebSocketAnalysisJob, broadcast_progress, broadcast_completion, broadcast_error
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    # Fallback to regular AnalysisJob
    WebSocketAnalysisJob = None

logger = logging.getLogger(__name__)

# Global storage for analysis jobs
analysis_jobs = {}
analysis_lock = threading.Lock()

api_analyzer_bp = Blueprint('api_analyzer', __name__)

def _improve_project_name_from_analysis(output_data):
    """Try to extract a better project name from analysis results."""
    try:
        # Look at module names in the analysis
        nodes = output_data.get('nodes', [])
        module_names = []
        
        for node in nodes:
            if node.get('type') == 'MOD':
                module_name = node.get('Name', '')
                if module_name:
                    module_names.append(module_name)
        
        # Look for common prefixes in module names
        if module_names:
            # Check for "mlx" prefix
            mlx_modules = [name for name in module_names if name.startswith('mlx')]
            if mlx_modules:
                return "mlx"
            
            # Look for common prefixes
            if len(module_names) > 1:
                common_prefix = ""
                first_module = module_names[0]
                for i, char in enumerate(first_module):
                    if all(len(name) > i and name[i] == char for name in module_names):
                        common_prefix += char
                    else:
                        break
                
                # Clean up the prefix
                common_prefix = common_prefix.rstrip('_-.')
                if len(common_prefix) >= 3:
                    return common_prefix
        
        return "uploaded_project"  # Fallback
        
    except Exception as e:
        logger.warning(f"Error improving project name: {e}")
        return "uploaded_project"

class AnalysisJob:
    """Tracks analysis job state and progress."""
    
    def __init__(self, job_id: str, project_path: str, llm_provider: str = 'none', original_project_name: str = None):
        self.job_id = job_id
        self.project_path = project_path
        self.original_project_name = original_project_name or Path(project_path).name
        self.llm_provider = llm_provider
        self.status = 'initializing'  # initializing, running, completed, failed
        self.progress = 0
        self.message = 'Initializing analysis...'
        self.result = None
        self.error = None
        self.created_at = datetime.utcnow()
        self.completed_at = None
        self.stats = {}
        self.analysis_file_path = None  # Path to saved analysis JSON file
        self.dead_code_analysis = None  # Dead code analysis data
        
    def update_progress(self, progress: int, message: str):
        """Update job progress."""
        self.progress = min(100, max(0, progress))
        self.message = message
        self.status = 'running' if progress < 100 else self.status
        
    def complete(self, result: dict):
        """Mark job as completed with result."""
        self.status = 'completed'
        self.progress = 100
        self.message = 'Analysis completed successfully'
        self.result = result
        self.completed_at = datetime.utcnow()
        
    def fail(self, error: str):
        """Mark job as failed with error."""
        self.status = 'failed'
        self.message = f'Analysis failed: {error}'
        self.error = error
        self.completed_at = datetime.utcnow()
        
    def to_dict(self) -> dict:
        """Convert job to dictionary for JSON response."""
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

def extract_directory_from_files(files: List[tuple]) -> Tuple[str, List[str]]:
    """Extract uploaded files to temporary directory."""
    temp_dir = tempfile.mkdtemp(prefix='analyzer_upload_')
    extracted_files = []
    
    try:
        for file_data, filename in files:
            # Secure the filename and preserve directory structure
            safe_path = secure_filename(filename)
            if '/' in filename:  # Preserve directory structure
                # Split path and secure each component
                path_parts = filename.split('/')
                safe_parts = [secure_filename(part) for part in path_parts if part]
                safe_path = '/'.join(safe_parts)
            
            full_path = os.path.join(temp_dir, safe_path)
            
            # Create directories if needed
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Write file
            with open(full_path, 'wb') as f:
                f.write(file_data)
            
            extracted_files.append(full_path)
            
        logger.info(f"Extracted {len(extracted_files)} files to {temp_dir}")
        return temp_dir, extracted_files
        
    except Exception as e:
        # Cleanup on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e

def run_analysis_job(job: AnalysisJob):
    """Run analysis in background thread using existing CLI analyzer."""
    try:
        job.update_progress(10, 'Loading configuration...')
        
        # Load the same config as CLI
        config_path = Path(__file__).parent.parent.parent / 'config' / 'llm_config.yaml'
        analyzer_config = load_unified_config(str(config_path), llm_provider=job.llm_provider)
        
        job.update_progress(20, 'Initializing analyzer...')
        
        # Create analyzer instance (same as CLI)
        analyzer = FlowBasedAnalyzer(analyzer_config)
        
        job.update_progress(30, 'Validating project...')
        
        # Validate project (same as CLI)
        is_valid, errors = analyzer.validate_project(job.project_path)
        if not is_valid:
            job.fail(f"Project validation failed: {'; '.join(errors[:3])}")
            return
            
        job.update_progress(50, 'Analyzing project structure...')
        
        # Run analysis (identical to CLI flow)
        result = analyzer.analyze(job.project_path)
        
        job.update_progress(80, 'Processing results...')
        
        if not result.is_successful():
            error_msg = '; '.join(result.errors[:3]) if result.errors else "Unknown analysis error"
            job.fail(f"Analysis failed: {error_msg}")
            return
            
        job.update_progress(90, 'Formatting output...')
        
        # Format output (same as CLI)
        output_data = analyzer.format_for_export(result)
        
        # Override the temp directory path with the original project name
        if 'metadata' in output_data and job.original_project_name:
            # If original project name is generic, try to improve it based on analysis
            final_project_name = job.original_project_name
            if final_project_name == "uploaded_project":
                # Try to extract a better name from the analysis
                final_project_name = _improve_project_name_from_analysis(output_data)
            
            output_data['metadata']['project_path'] = final_project_name
            output_data['metadata']['original_project_name'] = final_project_name
            logger.info(f"Updated project path to: {final_project_name}")
        
        # Save analysis result to temporary file for dead code analysis
        try:
            temp_analysis_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(output_data, temp_analysis_file, indent=2)
            temp_analysis_file.close()
            
            # Store file path in job for dead code analysis access
            job.analysis_file_path = temp_analysis_file.name
            
            logger.info(f"Analysis result saved to {temp_analysis_file.name}")
        except Exception as e:
            logger.warning(f"Failed to save analysis file: {e}")
            job.analysis_file_path = None
        
        # Store enhanced stats including dead code data
        job.stats = {
            'files_parsed': result.stats.files_parsed,
            'functions_found': result.stats.functions_found,
            'classes_found': result.stats.classes_found,
            'trigger_actors': result.stats.trigger_actors,
            'receiver_actors': result.stats.receiver_actors,
            'flow_chains': result.stats.flow_chains,
            'dead_functions': result.stats.dead_functions,
            'isolated_functions': result.stats.isolated_functions,
            'total_nodes': result.stats.sys_nodes + result.stats.mod_nodes + result.stats.func_nodes + result.stats.actor_nodes,
            'analysis_time_seconds': result.stats.analysis_time_seconds,
            'llm_calls_made': result.stats.llm_calls_made,
            'actors_enhanced': result.stats.actors_enhanced
        }
        
        # Store dead code analysis data if available
        if hasattr(result, 'dead_code_analysis') and result.dead_code_analysis:
            job.dead_code_analysis = result.dead_code_analysis
            logger.info(f"Dead code analysis stored: {len(result.dead_code_analysis.dead_functions)} dead functions")
        
        job.update_progress(100, 'Analysis completed!')
        job.complete(output_data)
        
    except Exception as e:
        logger.exception(f"Analysis job {job.job_id} failed")
        job.fail(str(e))
    finally:
        # Cleanup temporary directory
        if os.path.exists(job.project_path) and job.project_path.startswith(tempfile.gettempdir()):
            shutil.rmtree(job.project_path, ignore_errors=True)

@api_analyzer_bp.route('/validate-files', methods=['POST'])
def validate_files():
    """Validate uploaded files using backend filtering pipeline."""
    try:
        # Check if files were uploaded
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
            
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        # Sanity check: limit number of files to prevent overload
        if len(files) > 2000:
            return jsonify({'error': f'Too many files ({len(files)}). Maximum 2000 files allowed for validation.'}), 400
            
        # Read file data immediately while in request context
        file_data = []
        for file in files:
            if file.filename:
                # Read file content immediately
                content = file.read()
                # Use the original filename which contains the relative path from webkitdirectory
                relative_path = file.filename
                file_data.append((content, relative_path))
        
        # Extract files to temporary directory
        temp_dir, extracted_files = extract_directory_from_files(file_data)
        
        try:
            # Use CLI pipeline for consistent filtering - Load config first
            config_path = Path(__file__).parent.parent.parent / 'config' / 'llm_config.yaml'
            analyzer_config = load_unified_config(str(config_path), llm_provider='none')
            
            # Use ProjectDiscoverer for consistent file filtering (same as CLI)
            from analyzer.core.project_discoverer import ProjectDiscoverer
            discoverer = ProjectDiscoverer(analyzer_config)
            project_structure = discoverer.discover_project(temp_dir)
            
            # Get valid file paths relative to uploaded structure
            valid_files = []
            for project_file in project_structure.python_files:
                # Convert back to original file name format
                relative_to_temp = str(project_file.relative_path)
                # Find matching original file
                for _, original_path in file_data:
                    if original_path.endswith(relative_to_temp) or original_path == relative_to_temp:
                        valid_files.append(original_path)
                        break
            
            total_uploaded = len(file_data)
            excluded_count = total_uploaded - len(valid_files)
            
            return jsonify({
                'valid_files': valid_files,
                'total_uploaded': total_uploaded,
                'excluded_count': excluded_count,
                'total_lines': project_structure.total_lines,
                'message': f'Validated {len(valid_files)} files, excluded {excluded_count}'
            })
            
        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        logger.error(f"File validation error: {e}")
        return jsonify({'error': f'Validation failed: {str(e)}'}), 500

@api_analyzer_bp.route('/analyze-directory', methods=['POST'])
def analyze_directory():
    """Analyze a local directory path directly using CLI pipeline."""
    try:
        data = request.get_json()
        if not data or 'directory_path' not in data:
            return jsonify({'error': 'Directory path required'}), 400
            
        directory_path = data['directory_path']
        llm_provider = data.get('llm_provider', 'none')
        
        # Validate directory exists and is accessible
        from pathlib import Path
        dir_path = Path(directory_path)
        if not dir_path.exists():
            return jsonify({'error': f'Directory does not exist: {directory_path}'}), 400
        if not dir_path.is_dir():
            return jsonify({'error': f'Path is not a directory: {directory_path}'}), 400
            
        # Use CLI pipeline directly (no file upload needed)
        config_path = Path(__file__).parent.parent.parent / 'config' / 'llm_config.yaml'
        analyzer_config = load_unified_config(str(config_path), llm_provider=llm_provider)
        
        # Create analyzer instance (same as CLI)
        analyzer = FlowBasedAnalyzer(analyzer_config)
        
        # Validate project (same as CLI)
        is_valid, errors = analyzer.validate_project(directory_path)
        if not is_valid:
            return jsonify({'error': f"Project validation failed: {'; '.join(errors[:3])}"}), 400
            
        # Run analysis (identical to CLI flow)
        result = analyzer.analyze(directory_path)
        
        if not result.is_successful():
            error_msg = '; '.join(result.errors[:3]) if result.errors else "Unknown analysis error"
            return jsonify({'error': f"Analysis failed: {error_msg}"}), 500
            
        # Format output (same as CLI)
        output_data = analyzer.format_for_export(result)
        
        # Clean up project path for better display - do this AFTER format_for_export
        if 'metadata' in output_data:
            # Extract just the directory name for cleaner display
            project_name = Path(directory_path).name
            output_data['metadata']['project_path'] = project_name
            output_data['metadata']['original_project_name'] = project_name
            output_data['metadata']['full_directory_path'] = directory_path  # Keep original for reference
            logger.info(f"Set project name to: {project_name} (was: {directory_path})")
        
        # Return analysis result directly  
        return jsonify({
            'success': True,
            'result': output_data,
            'stats': {
                'files_parsed': result.stats.files_parsed,
                'functions_found': result.stats.functions_found,
                'total_lines': result.stats.files_discovered,  # Use available stat
                'analysis_time_seconds': result.stats.analysis_time_seconds
            }
        })
        
    except Exception as e:
        logger.error(f"Directory analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@api_analyzer_bp.route('/analyze', methods=['POST'])
def start_analysis():
    """Start a new analysis job."""
    try:
        # Check if files were uploaded
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
            
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
            
        # Get LLM provider option
        llm_provider = request.form.get('llm_provider', 'none')
        if llm_provider not in ['none', 'local', 'openai', 'anthropic']:
            llm_provider = 'none'
            
        # Process uploaded files and detect project name
        file_data = []
        potential_project_names = set()
        
        for file in files:
            if file.filename:
                content = file.read()
                # Get relative path from webkitRelativePath if available
                relative_path = request.form.get(f'path_{file.filename}', file.filename)
                file_data.append((content, relative_path))
                
                # Extract potential project name from file paths
                # Try multiple ways to get the project name
                path_parts = relative_path.split('/')
                
                if len(path_parts) >= 2:
                    # First part could be the project name
                    top_dir = path_parts[0]
                    if top_dir and not top_dir.startswith('.') and top_dir != 'src':
                        potential_project_names.add(top_dir)
                
                # Also try common patterns like project-name/*.py
                if relative_path.endswith('.py'):
                    # Look for meaningful directory names
                    for part in path_parts[:-1]:  # Exclude filename
                        if (part and not part.startswith('.') and 
                            part not in ['src', 'lib', 'python', 'app'] and
                            len(part) > 2):  # Meaningful name length
                            potential_project_names.add(part)
        
        # Determine the most likely project name
        if potential_project_names:
            # Use the most common top-level directory name
            project_name = max(potential_project_names, key=lambda x: sum(1 for _, path in file_data if path.startswith(x + '/')))
            logger.info(f"Detected project name: {project_name} from candidates: {potential_project_names}")
        else:
            # Try to guess from file names if no directory structure
            project_name = "uploaded_project"
            
            # Look for common patterns in filenames - improved detection
            for _, relative_path in file_data:
                filename = relative_path.split('/')[-1]  # Get just the filename
                if filename and filename.endswith('.py'):
                    # Remove .py extension
                    base_name = filename[:-3]
                    
                    # Look for patterns like "mlx_api_server" -> "mlx"
                    if '_' in base_name:
                        parts = base_name.split('_')
                        first_part = parts[0]
                        if len(first_part) >= 3 and first_part.isalpha():
                            project_name = first_part
                            logger.info(f"Detected project name '{project_name}' from filename: {filename}")
                            break
                    
                    # Look for patterns like "mlxtest" -> "mlx"  
                    if base_name.startswith('mlx'):
                        project_name = "mlx"
                        logger.info(f"Detected 'mlx' project from filename: {filename}")
                        break
                    
                    # Look for other common patterns
                    if len(base_name) >= 3 and base_name.isalpha():
                        # If it's a simple alphabetic name, use it
                        project_name = base_name
                        logger.info(f"Using base filename as project name: {project_name}")
                        break
            
            if project_name == "uploaded_project":
                logger.info("No project name detected, using default: uploaded_project")
        
        if not file_data:
            return jsonify({'error': 'No valid files found'}), 400
            
        # Extract files to temporary directory
        temp_dir, extracted_files = extract_directory_from_files(file_data)
        
        # Use CLI pipeline for consistent filtering - Load config first
        config_path = Path(__file__).parent.parent.parent / 'config' / 'llm_config.yaml'
        analyzer_config = load_unified_config(str(config_path), llm_provider=llm_provider)
        
        # Use ProjectDiscoverer for consistent file filtering (same as CLI)
        from analyzer.core.project_discoverer import ProjectDiscoverer
        discoverer = ProjectDiscoverer(analyzer_config)
        project_structure = discoverer.discover_project(temp_dir)
        
        # Check if any valid Python files found after filtering
        if project_structure.total_files == 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            error_msg = 'No Python files found after filtering'
            if project_structure.errors:
                error_msg += f': {"; ".join(project_structure.errors[:2])}'
            return jsonify({'error': error_msg}), 400
            
        # Create analysis job (with WebSocket support if available)
        job_id = str(uuid.uuid4())
        if WEBSOCKET_AVAILABLE and WebSocketAnalysisJob:
            job = WebSocketAnalysisJob(job_id, temp_dir, llm_provider, project_name)
        else:
            job = AnalysisJob(job_id, temp_dir, llm_provider, project_name)
        
        # Store job
        with analysis_lock:
            analysis_jobs[job_id] = job
            
        # Start analysis in background
        thread = threading.Thread(target=run_analysis_job, args=(job,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'started',
            'message': 'Analysis started',
            'python_files_found': project_structure.total_files,
            'total_lines': project_structure.total_lines,
            'llm_provider': llm_provider
        })
        
    except Exception as e:
        logger.exception("Failed to start analysis")
        return jsonify({'error': f'Failed to start analysis: {str(e)}'}), 500

@api_analyzer_bp.route('/status/<job_id>', methods=['GET'])
def get_analysis_status(job_id):
    """Get analysis job status."""
    with analysis_lock:
        job = analysis_jobs.get(job_id)
        
    if not job:
        return jsonify({'error': 'Job not found'}), 404
        
    return jsonify(job.to_dict())

@api_analyzer_bp.route('/result/<job_id>', methods=['GET'])
def get_analysis_result(job_id):
    """Get analysis result."""
    with analysis_lock:
        job = analysis_jobs.get(job_id)
        
    if not job:
        return jsonify({'error': 'Job not found'}), 404
        
    if job.status != 'completed':
        return jsonify({'error': 'Analysis not completed', 'status': job.status}), 400
        
    if not job.result:
        return jsonify({'error': 'No result available'}), 404
        
    return jsonify(job.result)

@api_analyzer_bp.route('/jobs', methods=['GET'])
def list_analysis_jobs():
    """List all analysis jobs."""
    with analysis_lock:
        jobs = [job.to_dict() for job in analysis_jobs.values()]
        
    # Sort by creation time, newest first
    jobs.sort(key=lambda x: x['created_at'], reverse=True)
    
    return jsonify({'jobs': jobs})

@api_analyzer_bp.route('/latest-job', methods=['GET'])
def get_latest_job():
    """Get the latest completed analysis job."""
    with analysis_lock:
        completed_jobs = [job for job in analysis_jobs.values() if job.status == 'completed']
        
    if not completed_jobs:
        return jsonify({'error': 'No completed analysis jobs found'}), 404
        
    # Sort by completion time, newest first
    latest_job = max(completed_jobs, key=lambda x: x.completed_at or x.created_at)
    
    return jsonify({
        'job_id': latest_job.job_id,
        'status': latest_job.status,
        'completed_at': latest_job.completed_at.isoformat() if latest_job.completed_at else None,
        'stats': latest_job.stats,
        'has_dead_code_analysis': hasattr(latest_job, 'dead_code_analysis') and latest_job.dead_code_analysis is not None,
        'analysis_file_path': getattr(latest_job, 'analysis_file_path', None)
    })

@api_analyzer_bp.route('/cleanup', methods=['POST'])
def cleanup_old_jobs():
    """Clean up old completed jobs."""
    cutoff_time = datetime.utcnow().timestamp() - (24 * 60 * 60)  # 24 hours ago
    
    with analysis_lock:
        to_remove = []
        for job_id, job in analysis_jobs.items():
            if (job.status in ['completed', 'failed'] and 
                job.created_at.timestamp() < cutoff_time):
                to_remove.append(job_id)
                
        for job_id in to_remove:
            del analysis_jobs[job_id]
            
    return jsonify({'cleaned_up': len(to_remove), 'remaining': len(analysis_jobs)})

# Health check for API
@api_analyzer_bp.route('/health', methods=['GET'])
def api_health():
    """API health check."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'active_jobs': len([j for j in analysis_jobs.values() if j.status == 'running']),
        'total_jobs': len(analysis_jobs)
    })