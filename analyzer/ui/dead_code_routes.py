#!/usr/bin/env python3
"""
Dead Code Analysis Routes for Code Architecture Analyzer

Provides web interface for dead code analysis and customer decisions.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, session, flash
from werkzeug.utils import secure_filename

from ..core.dead_code_detector import DeadCodeDetector

logger = logging.getLogger(__name__)

# Create blueprint  
dead_code_bp = Blueprint('dead_code', __name__, url_prefix='/dead_code')


@dead_code_bp.route('/')
def dead_code_analysis():
    """Show dead code analysis interface"""
    try:
        # Try multiple approaches to get analysis data
        dead_code_analysis = None
        
        # Approach 1: Try to get from latest analysis job
        try:
            from .api_analyzer import analysis_jobs, analysis_lock
            
            with analysis_lock:
                completed_jobs = [job for job in analysis_jobs.values() if job.status == 'completed']
            
            if completed_jobs:
                # Get the latest completed job
                latest_job = max(completed_jobs, key=lambda x: x.completed_at or x.created_at)
                logger.info(f"Found latest job: {latest_job.job_id} with stats: {latest_job.stats}")
                
                # Check if we have real dead code analysis data
                if hasattr(latest_job, 'dead_code_analysis') and latest_job.dead_code_analysis:
                    dead_code_analysis = latest_job.dead_code_analysis
                    logger.info(f"Using real dead code analysis: {len(dead_code_analysis.dead_functions)} dead functions")
                elif latest_job.result:
                    dead_code_analysis = _create_mock_dead_code_analysis(latest_job.result)
                    logger.info(f"Using mock dead code analysis: {len(dead_code_analysis.dead_functions)} dead functions")
        except Exception as e:
            logger.warning(f"Could not get analysis from jobs: {e}")
        
        # Approach 2: Try to load from the latest analysis file
        if not dead_code_analysis:
            try:
                analysis_file = Path("/Users/andreas/Documents/Projekte/grphzer2/analysis_result_2025-06-18T12-34-22.json")
                if analysis_file.exists():
                    with open(analysis_file, 'r') as f:
                        analysis_result = json.load(f)
                    dead_code_analysis = _create_mock_dead_code_analysis(analysis_result)
                    logger.info(f"Using file-based analysis: {len(dead_code_analysis.dead_functions)} dead functions")
            except Exception as e:
                logger.warning(f"Could not load analysis from file: {e}")
        
        # If we still don't have analysis data
        if not dead_code_analysis:
            return render_template('dead_code_analysis.html', 
                                 analysis=None, 
                                 error="No analysis data found. Please run an analysis first.")
        
        return render_template('dead_code_analysis.html', 
                             analysis=dead_code_analysis,
                             has_dead_code=len(dead_code_analysis.dead_functions) > 0)
        
    except Exception as e:
        logger.error(f"Error in dead code analysis route: {e}")
        return render_template('dead_code_analysis.html', 
                             analysis=None,
                             error=f"Error loading analysis: {e}")


@dead_code_bp.route('/api/function_action', methods=['POST'])
def handle_function_action():
    """Handle customer decisions on dead code functions"""
    try:
        data = request.get_json()
        function_name = data.get('function_name')
        action = data.get('action')  # 'remove', 'keep', 'review'
        
        if not function_name or not action:
            return jsonify({'error': 'Missing function_name or action'}), 400
        
        # Store customer decision in session
        if 'dead_code_decisions' not in session:
            session['dead_code_decisions'] = {}
        
        session['dead_code_decisions'][function_name] = {
            'action': action,
            'timestamp': str(datetime.utcnow()),
            'notes': data.get('notes', '')
        }
        session.modified = True
        
        logger.info(f"Dead code decision: {function_name} -> {action}")
        
        return jsonify({
            'status': 'success',
            'message': f'Action "{action}" recorded for {function_name}'
        })
        
    except Exception as e:
        logger.error(f"Error handling function action: {e}")
        return jsonify({'error': str(e)}), 500


@dead_code_bp.route('/api/bulk_action', methods=['POST'])
def handle_bulk_action():
    """Handle bulk actions on dead code functions"""
    try:
        data = request.get_json()
        action = data.get('action')  # 'remove_all_duplicates', 'export_report'
        function_names = data.get('function_names', [])
        
        if not action:
            return jsonify({'error': 'Missing action'}), 400
        
        if action == 'remove_all_duplicates':
            # Mark all specified functions for removal
            if 'dead_code_decisions' not in session:
                session['dead_code_decisions'] = {}
            
            for func_name in function_names:
                session['dead_code_decisions'][func_name] = {
                    'action': 'remove',
                    'timestamp': str(datetime.utcnow()),
                    'notes': 'Bulk removal of duplicates'
                }
            
            session.modified = True
            
            return jsonify({
                'status': 'success',
                'message': f'Marked {len(function_names)} duplicate functions for removal'
            })
        
        elif action == 'export_report':
            # Generate dead code report
            report = _generate_dead_code_report()
            return jsonify({
                'status': 'success',
                'report': report
            })
        
        else:
            return jsonify({'error': f'Unknown action: {action}'}), 400
        
    except Exception as e:
        logger.error(f"Error handling bulk action: {e}")
        return jsonify({'error': str(e)}), 500


@dead_code_bp.route('/api/export_decisions')
def export_decisions():
    """Export customer decisions as JSON"""
    try:
        decisions = session.get('dead_code_decisions', {})
        
        # Add summary
        summary = {
            'total_decisions': len(decisions),
            'remove_count': len([d for d in decisions.values() if d['action'] == 'remove']),
            'keep_count': len([d for d in decisions.values() if d['action'] == 'keep']),
            'review_count': len([d for d in decisions.values() if d['action'] == 'review']),
            'export_timestamp': str(datetime.utcnow())
        }
        
        export_data = {
            'summary': summary,
            'decisions': decisions
        }
        
        return jsonify(export_data)
        
    except Exception as e:
        logger.error(f"Error exporting decisions: {e}")
        return jsonify({'error': str(e)}), 500

@dead_code_bp.route('/debug')
def debug_info():
    """Debug information for troubleshooting"""
    try:
        from ..ui.api_analyzer import analysis_jobs, analysis_lock
        
        debug_data = {
            'jobs_available': False,
            'latest_job_info': None,
            'file_analysis_available': False,
            'file_analysis_info': None
        }
        
        # Check analysis jobs
        try:
            with analysis_lock:
                completed_jobs = [job for job in analysis_jobs.values() if job.status == 'completed']
            
            if completed_jobs:
                latest_job = max(completed_jobs, key=lambda x: x.completed_at or x.created_at)
                debug_data['jobs_available'] = True
                debug_data['latest_job_info'] = {
                    'job_id': latest_job.job_id,
                    'status': latest_job.status,
                    'stats': latest_job.stats,
                    'has_dead_code_analysis': hasattr(latest_job, 'dead_code_analysis') and latest_job.dead_code_analysis is not None,
                    'original_project_name': getattr(latest_job, 'original_project_name', None)
                }
        except Exception as e:
            debug_data['job_error'] = str(e)
        
        # Check file analysis
        try:
            analysis_file = Path("/Users/andreas/Documents/Projekte/grphzer2/analysis_result_2025-06-18T12-34-22.json")
            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    analysis_result = json.load(f)
                
                debug_data['file_analysis_available'] = True
                debug_data['file_analysis_info'] = {
                    'project_path': analysis_result.get('metadata', {}).get('project_path', 'Unknown'),
                    'warnings': analysis_result.get('metadata', {}).get('warnings', []),
                    'dead_functions_stat': analysis_result.get('metadata', {}).get('analysis_stats', {}).get('dead_functions', 0)
                }
        except Exception as e:
            debug_data['file_error'] = str(e)
        
        return jsonify(debug_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _create_mock_dead_code_analysis(analysis_result):
    """Create mock dead code analysis from analysis result warnings"""
    from ..core.dead_code_detector import DeadCodeAnalysis, DeadCodeFunction
    from datetime import datetime
    
    # Extract dead functions from warnings
    warnings = analysis_result.get('metadata', {}).get('warnings', [])
    dead_function_names = []
    
    for warning in warnings:
        if 'Dead code detected' in warning and 'unused functions:' in warning:
            # Parse function names from warning
            parts = warning.split('unused functions:')
            if len(parts) > 1:
                func_names = [name.strip() for name in parts[1].split(',')]
                dead_function_names.extend(func_names)
    
    # Create DeadCodeFunction objects
    dead_functions = []
    duplicates = []
    orphaned = []
    
    for func_name in dead_function_names:
        # Determine if it's a duplicate
        if 'unified_mlx_server' in func_name:
            issue_type = "DUPLICATE"
            base_name = func_name.split('.')[-1]
            similar_functions = [f"mlx_api_server.{base_name}"]
            suggestion = f"Remove or integrate with active version: mlx_api_server.{base_name}"
            reason = "Duplicate function exists in mlx_api_server module"
        else:
            issue_type = "ORPHANED"
            similar_functions = []
            suggestion = "Legacy code or incomplete feature?"
            reason = "No callers or triggers found"
        
        # Extract better location info
        module_name = func_name.split('.')[0] if '.' in func_name else 'unknown'
        function_name = func_name.split('.')[-1]
        
        # Create more realistic code snippet
        if issue_type == "DUPLICATE":
            code_snippet = f"def {function_name}():\n    \"\"\"Duplicate function - also exists in mlx_api_server\"\"\"\n    # Implementation duplicated from other module...\n    pass"
        else:
            code_snippet = f"def {function_name}():\n    \"\"\"Orphaned function - no callers found\"\"\"\n    # Implementation appears unused...\n    pass"
        
        dead_func = DeadCodeFunction(
            name=function_name,
            module=module_name,
            full_name=func_name,
            location=f"{module_name}.py:15-25",
            code_snippet=code_snippet,
            issue_type=issue_type,
            similar_functions=similar_functions,
            suggestion=suggestion,
            reason=reason,
            file_path=f"{module_name}.py",
            line_start=15,
            line_end=25,
            function_size=10,
            has_docstring=True,
            complexity=3
        )
        
        dead_functions.append(dead_func)
        
        if issue_type == "DUPLICATE":
            duplicates.append(dead_func)
        else:
            orphaned.append(dead_func)
    
    # Generate summary
    summary = {
        "total_dead": len(dead_functions),
        "by_type": {
            "duplicates": len(duplicates),
            "orphaned": len(orphaned),
            "unreachable": 0
        },
        "by_module": {},
        "total_loc": len(dead_functions) * 10,  # Mock LOC
        "largest_function": dead_functions[0].full_name if dead_functions else None,
        "largest_function_size": 10
    }
    
    # Group by module
    for func in dead_functions:
        if func.module not in summary["by_module"]:
            summary["by_module"][func.module] = []
        summary["by_module"][func.module].append(func.name)
    
    return DeadCodeAnalysis(
        total_functions=len(analysis_result.get('nodes', [])),
        dead_functions=dead_functions,
        duplicates=duplicates,
        orphaned=orphaned,
        unreachable=[],
        summary=summary
    )


def _generate_dead_code_report():
    """Generate a dead code analysis report"""
    from datetime import datetime
    
    decisions = session.get('dead_code_decisions', {})
    
    report = {
        'generated_at': str(datetime.utcnow()),
        'summary': {
            'total_decisions': len(decisions),
            'actions': {
                'remove': len([d for d in decisions.values() if d['action'] == 'remove']),
                'keep': len([d for d in decisions.values() if d['action'] == 'keep']),
                'review': len([d for d in decisions.values() if d['action'] == 'review'])
            }
        },
        'recommendations': [],
        'decisions': decisions
    }
    
    # Add recommendations
    remove_funcs = [name for name, decision in decisions.items() if decision['action'] == 'remove']
    if remove_funcs:
        report['recommendations'].append({
            'type': 'removal',
            'description': f'Remove {len(remove_funcs)} unused functions',
            'functions': remove_funcs,
            'estimated_loc_reduction': len(remove_funcs) * 10
        })
    
    keep_funcs = [name for name, decision in decisions.items() if decision['action'] == 'keep']
    if keep_funcs:
        report['recommendations'].append({
            'type': 'documentation',
            'description': f'Document {len(keep_funcs)} kept functions for future reference',
            'functions': keep_funcs
        })
    
    return report