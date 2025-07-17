#!/usr/bin/env python3
"""
Configuration loader for Code Architecture Analyzer

Handles unified LLM configuration with secure API key management.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import re
from .config_validator import ConfigurationValidator, validate_config_file

logger = logging.getLogger(__name__)


def substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute environment variables in configuration values.
    
    Supports ${VAR_NAME} syntax.
    """
    if isinstance(value, str):
        # Find all ${VAR_NAME} patterns
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        
        for var_name in matches:
            env_value = os.getenv(var_name)
            if env_value is None:
                logger.warning(f"Environment variable {var_name} not set")
                env_value = ""  # Use empty string as fallback
            
            # Replace ${VAR_NAME} with actual value
            value = value.replace(f"${{{var_name}}}", env_value)
        
        return value
    
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    
    else:
        return value


def load_unified_config(config_path: str, llm_provider: str = 'none') -> Dict[str, Any]:
    """
    Load unified configuration file and configure for specified LLM provider.
    
    Args:
        config_path: Path to the configuration YAML file
        llm_provider: LLM provider to use ('none', 'local', 'openai', 'anthropic')
    
    Returns:
        Configuration dictionary ready for analyzer
    """
    try:
        # Validate config file syntax first
        is_valid, file_issues = validate_config_file(config_path)
        if not is_valid:
            for issue in file_issues:
                logger.error(f"Config file validation: {issue}")
            raise ValueError("Configuration file validation failed")
        
        # Load YAML configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        # Substitute environment variables
        config = substitute_env_vars(raw_config)
        
        # Validate configuration structure and values
        validator = ConfigurationValidator()
        is_valid, config_issues = validator.validate_analysis_config(config)
        
        # Log all issues
        for issue in config_issues:
            if 'must be' in issue or 'Invalid' in issue or 'Missing required' in issue:
                logger.error(f"Config validation error: {issue}")
            else:
                logger.warning(f"Config validation warning: {issue}")
        
        # Stop if there are critical errors
        if not is_valid:
            raise ValueError("Configuration validation failed - check logs for details")
        
        # Get LLM provider configuration
        llm_providers = config.get('llm_providers', {})
        
        if llm_provider not in llm_providers:
            logger.error(f"LLM provider '{llm_provider}' not found in configuration")
            raise ValueError(f"Unknown LLM provider: {llm_provider}")
        
        provider_config = llm_providers[llm_provider]
        
        if not provider_config.get('enabled', False):
            logger.warning(f"LLM provider '{llm_provider}' is disabled in configuration")
        
        # Build analyzer configuration
        analyzer_config = {
            # Project settings
            'project': config.get('project', {}),
            
            # Basic analysis settings
            'deterministic': {
                'confidence_threshold': config.get('analysis', {}).get('confidence_threshold', 0.8),
                'analyzers': {
                    'ast_parser': True,
                    'pattern_matcher': True,
                    'entry_point_detector': True
                }
            },
            
            # Flow analysis configuration
            'flow_analysis': {
                'enabled': config.get('analysis', {}).get('enable_flow_analysis', True),
                'llm_enhanced_descriptions': provider_config.get('flow_enhancement', False),
                'enhancement_provider': llm_provider if provider_config.get('flow_enhancement', False) else 'none',
                'max_description_length': config.get('analysis', {}).get('max_description_length', 200),
                'deterministic_fallback': True,
                'fchain_detection': {
                    'enabled': True,
                    'min_chain_length': 3,
                    'max_chains': 20
                }
            },
            
            # LLM configuration
            'llm': _build_llm_config(provider_config, llm_provider),
            
            # Graph settings
            'graph': {
                'nodes': {
                    'max_name_length': 25,
                    'naming_style': 'PascalCase',
                    'include_test_functions': config.get('analysis', {}).get('include_test_functions', False)
                },
                'relationships': {
                    'flow': {
                        'include_internal_calls': True,
                        'include_external_calls': True,
                        'min_confidence': 0.5,
                        'generate_descriptions': True
                    }
                }
            },
            
            # Output settings
            'output': config.get('output', {}),
            
            # Logging settings
            'logging': config.get('logging', {}),
            
            # Performance settings
            'performance': config.get('performance', {})
        }
        
        logger.info(f"Configuration loaded successfully with LLM provider: {llm_provider}")
        return analyzer_config
        
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def _build_llm_config(provider_config: Dict[str, Any], provider_name: str) -> Dict[str, Any]:
    """Build LLM configuration section based on provider."""
    
    base_config = {
        'provider': provider_name,
        'confidence_threshold': 0.7,
        'max_batch_size': 10,
        'enable_caching': True,
        'cache_ttl': 3600,
        'actor_enhancement': {
            'enabled': provider_config.get('actor_enhancement', False),
            'timeout_seconds': 10,
            'max_retries': 2,
            'cache_enabled': True,
            'fallback_on_failure': True,
            'enhancement_confidence_threshold': 0.7
        }
    }
    
    # Add provider-specific configuration
    if provider_name == 'none':
        base_config['actor_enhancement']['enabled'] = False
        
    elif provider_name == 'local':
        base_config['local'] = {
            'base_url': provider_config.get('base_url', 'http://localhost:1234'),
            'model': provider_config.get('model', 'local-model'),
            'timeout': provider_config.get('timeout', 30),
            'max_retries': provider_config.get('max_retries', 3),
            'headers': provider_config.get('headers', {})
        }
        
    elif provider_name == 'openai':
        base_config['openai'] = {
            'api_key': provider_config.get('api_key', ''),
            'model': provider_config.get('model', 'gpt-4'),
            'base_url': provider_config.get('base_url', 'https://api.openai.com/v1'),
            'timeout': provider_config.get('timeout', 30),
            'max_retries': provider_config.get('max_retries', 3),
            'temperature': provider_config.get('temperature', 0.1),
            'max_tokens': provider_config.get('max_tokens', 1000)
        }
        
    elif provider_name == 'anthropic':
        base_config['anthropic'] = {
            'api_key': provider_config.get('api_key', ''),
            'model': provider_config.get('model', 'claude-3-sonnet-20240229'),
            'base_url': provider_config.get('base_url', 'https://api.anthropic.com'),
            'timeout': provider_config.get('timeout', 30),
            'max_retries': provider_config.get('max_retries', 3),
            'temperature': provider_config.get('temperature', 0.1),
            'max_tokens': provider_config.get('max_tokens', 1000)
        }
    
    # Add default prompts
    base_config['prompts'] = {
        'actor_enhancement': """You are an expert software architect. Analyze this code actor and generate a meaningful name and description.

Actor Type: {actor_type}
Library: {library}
Code Snippet: {code_snippet}
Function Context: {function_context}
File Path: {file_path}
Target/URL: {target}

Respond with ONLY a JSON object:
{
  "name": "PascalCaseActorName",
  "description": "Brief description of what this actor does"
}"""
    }
    
    return base_config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure and required fields.
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Check required top-level sections
        required_sections = ['project', 'deterministic', 'flow_analysis', 'llm']
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate LLM provider
        llm_config = config.get('llm', {})
        provider = llm_config.get('provider', 'none')
        
        if provider not in ['none', 'local', 'openai', 'anthropic']:
            logger.error(f"Invalid LLM provider: {provider}")
            return False
        
        # Validate API keys for external providers
        if provider == 'openai':
            openai_config = llm_config.get('openai', {})
            api_key = openai_config.get('api_key', '')
            if not api_key or api_key.startswith('${'):
                logger.warning("OpenAI API key not set - LLM features will be disabled")
        
        elif provider == 'anthropic':
            anthropic_config = llm_config.get('anthropic', {})
            api_key = anthropic_config.get('api_key', '')
            if not api_key or api_key.startswith('${'):
                logger.warning("Anthropic API key not set - LLM features will be disabled")
        
        logger.info("Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def get_default_config_path() -> str:
    """Get the default configuration file path."""
    current_dir = Path(__file__).parent
    config_dir = current_dir.parent.parent / 'config'
    default_config = config_dir / 'llm_config.yaml'
    
    if not default_config.exists():
        logger.error(f"Default configuration file not found: {default_config}")
        raise FileNotFoundError(f"Default configuration file not found: {default_config}")
    
    return str(default_config)