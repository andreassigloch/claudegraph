"""
Configuration validation for the analyzer.
Validates configuration before analysis starts to prevent common errors.
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ConfigurationValidator:
    """Validates analysis configuration to prevent runtime errors."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_analysis_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration before analysis starts.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        self.errors = []
        self.warnings = []
        
        # Validate project settings
        self._validate_project_config(config.get('project', {}))
        
        # Validate analysis settings
        self._validate_analysis_config(config.get('analysis', {}))
        
        # Validate LLM providers
        self._validate_llm_providers(config.get('llm_providers', {}))
        
        # Validate output settings
        self._validate_output_config(config.get('output', {}))
        
        # Validate performance settings
        self._validate_performance_config(config.get('performance', {}))
        
        # Check required dependencies
        self._validate_dependencies()
        
        all_issues = self.errors + self.warnings
        is_valid = len(self.errors) == 0
        
        return is_valid, all_issues
    
    def _validate_project_config(self, project_config: Dict[str, Any]):
        """Validate project-specific configuration."""
        
        # Check include patterns
        include_patterns = project_config.get('include_patterns', [])
        if not include_patterns:
            self.warnings.append("No include patterns specified - analysis may not find any files")
        
        for pattern in include_patterns:
            if not isinstance(pattern, str):
                self.errors.append(f"Invalid include pattern (must be string): {pattern}")
        
        # Check exclude patterns
        exclude_patterns = project_config.get('exclude_patterns', [])
        for pattern in exclude_patterns:
            if not isinstance(pattern, str):
                self.errors.append(f"Invalid exclude pattern (must be string): {pattern}")
        
        # Validate limits
        max_loc = project_config.get('max_loc', 25000)
        if not isinstance(max_loc, int) or max_loc <= 0:
            self.errors.append(f"max_loc must be a positive integer, got: {max_loc}")
        
        max_file_size = project_config.get('max_file_size_kb', 500)
        if not isinstance(max_file_size, int) or max_file_size <= 0:
            self.errors.append(f"max_file_size_kb must be a positive integer, got: {max_file_size}")
    
    def _validate_analysis_config(self, analysis_config: Dict[str, Any]):
        """Validate analysis-specific settings."""
        
        # Validate confidence threshold
        confidence = analysis_config.get('confidence_threshold', 0.8)
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            self.errors.append(f"confidence_threshold must be between 0.0 and 1.0, got: {confidence}")
        
        # Validate boolean flags
        boolean_flags = ['enable_flow_analysis', 'include_test_functions']
        for flag in boolean_flags:
            value = analysis_config.get(flag)
            if value is not None and not isinstance(value, bool):
                self.errors.append(f"{flag} must be boolean, got: {value}")
        
        # Validate max description length
        max_desc_len = analysis_config.get('max_description_length', 200)
        if not isinstance(max_desc_len, int) or max_desc_len <= 0:
            self.errors.append(f"max_description_length must be positive integer, got: {max_desc_len}")
    
    def _validate_llm_providers(self, llm_config: Dict[str, Any]):
        """Validate LLM provider configurations."""
        
        if not llm_config:
            self.warnings.append("No LLM providers configured - only deterministic analysis available")
            return
        
        valid_providers = ['none', 'local', 'openai', 'anthropic']
        
        for provider_name, provider_config in llm_config.items():
            if not isinstance(provider_config, dict):
                self.errors.append(f"Provider {provider_name} config must be a dictionary")
                continue
            
            # Check provider type
            provider_type = provider_config.get('provider')
            if provider_type not in valid_providers:
                self.errors.append(f"Invalid provider type '{provider_type}' for {provider_name}. Valid: {valid_providers}")
            
            # Check required fields
            if 'enabled' not in provider_config:
                self.warnings.append(f"Provider {provider_name} missing 'enabled' field")
            
            # Validate API key patterns for external providers
            if provider_type in ['openai', 'anthropic']:
                api_key = provider_config.get('api_key', '')
                if api_key.startswith('${') and api_key.endswith('}'):
                    # Environment variable reference
                    env_var = api_key[2:-1]
                    if not os.getenv(env_var):
                        self.warnings.append(f"Environment variable {env_var} not set for {provider_name}")
                elif not api_key:
                    self.warnings.append(f"No API key configured for {provider_name}")
            
            # Validate timeout settings
            timeout = provider_config.get('timeout')
            if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
                self.errors.append(f"timeout for {provider_name} must be positive integer, got: {timeout}")
            
            # Validate retry settings
            max_retries = provider_config.get('max_retries')
            if max_retries is not None and (not isinstance(max_retries, int) or max_retries < 0):
                self.errors.append(f"max_retries for {provider_name} must be non-negative integer, got: {max_retries}")
    
    def _validate_output_config(self, output_config: Dict[str, Any]):
        """Validate output configuration."""
        
        # Check output format
        output_format = output_config.get('format', 'ontology_json')
        valid_formats = ['ontology_json', 'json', 'yaml']
        if output_format not in valid_formats:
            self.errors.append(f"Invalid output format '{output_format}'. Valid: {valid_formats}")
        
        # Validate indent setting
        indent = output_config.get('indent', 2)
        if not isinstance(indent, int) or indent < 0:
            self.errors.append(f"indent must be non-negative integer, got: {indent}")
        
        # Validate boolean flags
        boolean_flags = ['pretty_print', 'include_metadata', 'validate_output']
        for flag in boolean_flags:
            value = output_config.get(flag)
            if value is not None and not isinstance(value, bool):
                self.errors.append(f"output.{flag} must be boolean, got: {value}")
    
    def _validate_performance_config(self, perf_config: Dict[str, Any]):
        """Validate performance settings."""
        
        # Validate memory limit
        max_memory = perf_config.get('max_memory_mb', 1024)
        if not isinstance(max_memory, int) or max_memory <= 0:
            self.errors.append(f"max_memory_mb must be positive integer, got: {max_memory}")
        
        # Validate worker count
        max_workers = perf_config.get('max_workers', 4)
        if not isinstance(max_workers, int) or max_workers <= 0:
            self.errors.append(f"max_workers must be positive integer, got: {max_workers}")
        
        # Validate cache TTL
        cache_ttl = perf_config.get('cache_ttl_hours', 24)
        if not isinstance(cache_ttl, (int, float)) or cache_ttl <= 0:
            self.errors.append(f"cache_ttl_hours must be positive number, got: {cache_ttl}")
        
        # Check if worker count is reasonable
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if max_workers > cpu_count * 2:
            self.warnings.append(f"max_workers ({max_workers}) exceeds 2x CPU count ({cpu_count})")
    
    def _validate_dependencies(self):
        """Check for required dependencies."""
        required_modules = [
            'ast',
            'json',
            'pathlib',
            'logging',
            'uuid',
            'datetime',
            'typing'
        ]
        
        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError:
                self.errors.append(f"Missing required dependency: {module_name}")
        
        # Check optional dependencies
        optional_modules = {
            'yaml': 'YAML output support',
            'networkx': 'Advanced graph analysis',
            'click': 'CLI interface'
        }
        
        for module_name, description in optional_modules.items():
            try:
                __import__(module_name)
            except ImportError:
                self.warnings.append(f"Optional dependency missing: {module_name} ({description})")
    
    def validate_project_path(self, project_path: str) -> Tuple[bool, List[str]]:
        """
        Validate that the project path exists and is analyzable.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        path = Path(project_path)
        
        # Check if path exists
        if not path.exists():
            issues.append(f"Project path does not exist: {project_path}")
            return False, issues
        
        # Check if it's a directory
        if not path.is_dir():
            issues.append(f"Project path is not a directory: {project_path}")
            return False, issues
        
        # Check if directory is readable
        if not os.access(path, os.R_OK):
            issues.append(f"Project directory is not readable: {project_path}")
            return False, issues
        
        # Check for Python files
        python_files = list(path.rglob('*.py'))
        if not python_files:
            issues.append(f"No Python files found in project directory: {project_path}")
            return False, issues
        
        # Check directory size (basic check)
        total_files = len(list(path.rglob('*')))
        if total_files > 10000:
            issues.append(f"Project directory has many files ({total_files}) - analysis may be slow")
        
        return True, issues
    
    def validate_output_path(self, output_path: str) -> Tuple[bool, List[str]]:
        """
        Validate that the output path is writable.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        output_file = Path(output_path)
        
        # Check if parent directory exists
        parent_dir = output_file.parent
        if not parent_dir.exists():
            issues.append(f"Output directory does not exist: {parent_dir}")
            return False, issues
        
        # Check if parent directory is writable
        if not os.access(parent_dir, os.W_OK):
            issues.append(f"Output directory is not writable: {parent_dir}")
            return False, issues
        
        # Check if output file already exists and is writable
        if output_file.exists() and not os.access(output_file, os.W_OK):
            issues.append(f"Output file exists but is not writable: {output_path}")
            return False, issues
        
        # Warn about overwriting existing files
        if output_file.exists():
            issues.append(f"Output file already exists and will be overwritten: {output_path}")
        
        return True, issues


def validate_config_file(config_path: str) -> Tuple[bool, List[str]]:
    """
    Validate configuration file syntax and basic structure.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    config_file = Path(config_path)
    
    # Check if config file exists
    if not config_file.exists():
        issues.append(f"Configuration file not found: {config_path}")
        return False, issues
    
    # Check if config file is readable
    if not os.access(config_file, os.R_OK):
        issues.append(f"Configuration file is not readable: {config_path}")
        return False, issues
    
    # Try to parse the config file
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except ImportError:
        issues.append("PyYAML not available - cannot validate YAML config file")
        return False, issues
    except yaml.YAMLError as e:
        issues.append(f"Invalid YAML syntax in config file: {e}")
        return False, issues
    except Exception as e:
        issues.append(f"Error reading config file: {e}")
        return False, issues
    
    # Validate structure
    if not isinstance(config, dict):
        issues.append("Configuration file must contain a dictionary at root level")
        return False, issues
    
    # Check for required sections
    required_sections = ['project', 'llm_providers']
    for section in required_sections:
        if section not in config:
            issues.append(f"Missing required configuration section: {section}")
    
    return len(issues) == 0, issues