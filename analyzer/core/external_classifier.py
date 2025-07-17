#!/usr/bin/env python3
"""
External Call Classifier for Code Architecture Analyzer

Classifies external calls and imports to distinguish between:
- Interesting system actors (databases, external APIs, file systems)
- Uninteresting utility libraries (math, string processing, etc.)

This helps determine which external dependencies should become ACTOR nodes
versus which should be ignored in flow analysis.
"""

import re
import logging
from enum import Enum
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


class ExternalType(Enum):
    """Classification types for external calls."""
    SYSTEM_ACTOR = "system_actor"      # External systems (databases, APIs, filesystems)
    UTILITY_LIBRARY = "utility_library"  # Utility/computation libraries
    FRAMEWORK = "framework"            # Web frameworks, testing frameworks
    UNKNOWN = "unknown"               # Unclassified external calls


@dataclass
class ExternalClassification:
    """Result of external call classification."""
    module_name: str
    classification: ExternalType
    confidence: float
    actor_type: Optional[str] = None  # Specific actor type if SYSTEM_ACTOR
    reasoning: str = ""
    should_create_actor: bool = False


class ExternalClassifier:
    """Classifies external calls and imports."""
    
    # Known system actor patterns (databases, external APIs, file systems)
    SYSTEM_ACTOR_PATTERNS = {
        # Databases
        'neo4j': {'type': 'Database', 'confidence': 0.95, 'actor_type': 'Neo4jDatabase'},
        'pymongo': {'type': 'Database', 'confidence': 0.95, 'actor_type': 'MongoDatabase'},
        'psycopg2': {'type': 'Database', 'confidence': 0.95, 'actor_type': 'PostgreSQLDatabase'},
        'sqlite3': {'type': 'Database', 'confidence': 0.95, 'actor_type': 'SQLiteDatabase'},
        'redis': {'type': 'Database', 'confidence': 0.95, 'actor_type': 'RedisDatabase'},
        'sqlalchemy': {'type': 'Database', 'confidence': 0.90, 'actor_type': 'SQLDatabase'},
        'cx_Oracle': {'type': 'Database', 'confidence': 0.95, 'actor_type': 'OracleDatabase'},
        'pymysql': {'type': 'Database', 'confidence': 0.95, 'actor_type': 'MySQLDatabase'},
        
        # HTTP Clients / External APIs
        'requests': {'type': 'HttpClient', 'confidence': 0.90, 'actor_type': 'HttpClient'},
        'httpx': {'type': 'HttpClient', 'confidence': 0.90, 'actor_type': 'HttpClient'},
        'aiohttp': {'type': 'HttpClient', 'confidence': 0.90, 'actor_type': 'HttpClient'},
        'urllib.request': {'type': 'HttpClient', 'confidence': 0.85, 'actor_type': 'HttpClient'},
        'urllib3': {'type': 'HttpClient', 'confidence': 0.85, 'actor_type': 'HttpClient'},
        
        # File Systems / Cloud Storage
        'boto3': {'type': 'CloudStorage', 'confidence': 0.95, 'actor_type': 'AWSStorage'},
        'azure.storage': {'type': 'CloudStorage', 'confidence': 0.95, 'actor_type': 'AzureStorage'},
        'google.cloud.storage': {'type': 'CloudStorage', 'confidence': 0.95, 'actor_type': 'GCPStorage'},
        'paramiko': {'type': 'FileSystem', 'confidence': 0.90, 'actor_type': 'SSHFileSystem'},
        'ftplib': {'type': 'FileSystem', 'confidence': 0.90, 'actor_type': 'FTPFileSystem'},
        
        # Message Queues / Streaming
        'pika': {'type': 'MessageQueue', 'confidence': 0.95, 'actor_type': 'RabbitMQ'},
        'kafka': {'type': 'MessageQueue', 'confidence': 0.95, 'actor_type': 'KafkaQueue'},
        'celery': {'type': 'MessageQueue', 'confidence': 0.90, 'actor_type': 'CeleryQueue'},
        
        # External Services
        'stripe': {'type': 'ExternalAPI', 'confidence': 0.95, 'actor_type': 'PaymentService'},
        'twilio': {'type': 'ExternalAPI', 'confidence': 0.95, 'actor_type': 'SMSService'},
        'sendgrid': {'type': 'ExternalAPI', 'confidence': 0.95, 'actor_type': 'EmailService'},
    }
    
    # Known utility library patterns (computation, formatting, etc.)
    UTILITY_LIBRARY_PATTERNS = {
        # Standard Library - Core
        'os': {'confidence': 0.99, 'reasoning': 'Standard library OS interface'},
        'sys': {'confidence': 0.99, 'reasoning': 'Standard library system interface'},
        'time': {'confidence': 0.95, 'reasoning': 'Standard library time utilities'},
        'datetime': {'confidence': 0.95, 'reasoning': 'Standard library datetime utilities'},
        'math': {'confidence': 0.99, 'reasoning': 'Standard library math functions'},
        'random': {'confidence': 0.95, 'reasoning': 'Standard library random utilities'},
        'uuid': {'confidence': 0.95, 'reasoning': 'Standard library UUID generation'},
        'hashlib': {'confidence': 0.95, 'reasoning': 'Standard library hashing'},
        'base64': {'confidence': 0.95, 'reasoning': 'Standard library encoding'},
        'json': {'confidence': 0.90, 'reasoning': 'Standard library JSON processing'},
        'csv': {'confidence': 0.90, 'reasoning': 'Standard library CSV processing'},
        'xml': {'confidence': 0.90, 'reasoning': 'Standard library XML processing'},
        're': {'confidence': 0.95, 'reasoning': 'Standard library regex'},
        'string': {'confidence': 0.95, 'reasoning': 'Standard library string utilities'},
        'collections': {'confidence': 0.95, 'reasoning': 'Standard library collections'},
        'itertools': {'confidence': 0.95, 'reasoning': 'Standard library iteration utilities'},
        'functools': {'confidence': 0.95, 'reasoning': 'Standard library functional utilities'},
        'typing': {'confidence': 0.99, 'reasoning': 'Standard library type annotations'},
        'dataclasses': {'confidence': 0.95, 'reasoning': 'Standard library dataclasses'},
        'enum': {'confidence': 0.95, 'reasoning': 'Standard library enumerations'},
        'pathlib': {'confidence': 0.85, 'reasoning': 'Standard library path utilities'},
        'logging': {'confidence': 0.95, 'reasoning': 'Standard library logging'},
        'configparser': {'confidence': 0.85, 'reasoning': 'Standard library config parsing'},
        'argparse': {'confidence': 0.95, 'reasoning': 'Standard library argument parsing'},
        'subprocess': {'confidence': 0.80, 'reasoning': 'Standard library process execution'},
        'threading': {'confidence': 0.90, 'reasoning': 'Standard library threading'},
        'multiprocessing': {'confidence': 0.90, 'reasoning': 'Standard library multiprocessing'},
        'asyncio': {'confidence': 0.90, 'reasoning': 'Standard library async utilities'},
        'concurrent': {'confidence': 0.90, 'reasoning': 'Standard library concurrency'},
        
        # Data Processing Libraries
        'pandas': {'confidence': 0.90, 'reasoning': 'Data manipulation library'},
        'numpy': {'confidence': 0.95, 'reasoning': 'Numerical computation library'},
        'scipy': {'confidence': 0.95, 'reasoning': 'Scientific computation library'},
        'matplotlib': {'confidence': 0.95, 'reasoning': 'Plotting library'},
        'seaborn': {'confidence': 0.95, 'reasoning': 'Statistical plotting library'},
        'plotly': {'confidence': 0.90, 'reasoning': 'Interactive plotting library'},
        
        # Machine Learning Libraries
        'sklearn': {'confidence': 0.95, 'reasoning': 'Machine learning library'},
        'tensorflow': {'confidence': 0.95, 'reasoning': 'Machine learning framework'},
        'torch': {'confidence': 0.95, 'reasoning': 'Machine learning framework'},
        'keras': {'confidence': 0.95, 'reasoning': 'Machine learning framework'},
        
        # Validation and Parsing
        'pydantic': {'confidence': 0.85, 'reasoning': 'Data validation library'},
        'marshmallow': {'confidence': 0.85, 'reasoning': 'Data serialization library'},
        'jsonschema': {'confidence': 0.90, 'reasoning': 'JSON schema validation'},
        'yaml': {'confidence': 0.85, 'reasoning': 'YAML parsing library'},
        'toml': {'confidence': 0.85, 'reasoning': 'TOML parsing library'},
        'lxml': {'confidence': 0.85, 'reasoning': 'XML parsing library'},
        'beautifulsoup4': {'confidence': 0.85, 'reasoning': 'HTML parsing library'},
        
        # Utility Libraries
        'click': {'confidence': 0.90, 'reasoning': 'CLI utility library'},
        'rich': {'confidence': 0.95, 'reasoning': 'Terminal formatting library'},
        'colorama': {'confidence': 0.95, 'reasoning': 'Terminal color library'},
        'tqdm': {'confidence': 0.95, 'reasoning': 'Progress bar library'},
        'retry': {'confidence': 0.90, 'reasoning': 'Retry utility library'},
        'tenacity': {'confidence': 0.90, 'reasoning': 'Retry utility library'},
        'python-dotenv': {'confidence': 0.85, 'reasoning': 'Environment variable utility'},
        'cryptography': {'confidence': 0.85, 'reasoning': 'Cryptography utility library'},
        'pillow': {'confidence': 0.90, 'reasoning': 'Image processing library'},
        'imageio': {'confidence': 0.90, 'reasoning': 'Image I/O library'},
    }
    
    # Framework patterns (might be interesting for architectural analysis)
    FRAMEWORK_PATTERNS = {
        'flask': {'confidence': 0.95, 'reasoning': 'Web framework'},
        'fastapi': {'confidence': 0.95, 'reasoning': 'Web framework'},
        'django': {'confidence': 0.95, 'reasoning': 'Web framework'},
        'tornado': {'confidence': 0.95, 'reasoning': 'Web framework'},
        'pytest': {'confidence': 0.95, 'reasoning': 'Testing framework'},
        'unittest': {'confidence': 0.95, 'reasoning': 'Testing framework'},
        'nose': {'confidence': 0.95, 'reasoning': 'Testing framework'},
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the external classifier."""
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.8)
        self.create_framework_actors = self.config.get('create_framework_actors', False)
        
        # Load custom patterns from config
        self.custom_system_patterns = self.config.get('custom_system_actors', {})
        self.custom_utility_patterns = self.config.get('custom_utilities', {})
        
        logger.debug(f"ExternalClassifier initialized with threshold {self.confidence_threshold}")
    
    def classify_import(self, module_name: str) -> ExternalClassification:
        """
        Classify an import statement.
        
        Args:
            module_name: The imported module name
            
        Returns:
            ExternalClassification result
        """
        # Normalize module name for classification
        normalized_name = self._normalize_module_name(module_name)
        
        # Check system actor patterns first
        system_result = self._check_system_patterns(normalized_name, module_name)
        if system_result:
            return system_result
        
        # Check utility library patterns
        utility_result = self._check_utility_patterns(normalized_name, module_name)
        if utility_result:
            return utility_result
        
        # Check framework patterns
        framework_result = self._check_framework_patterns(normalized_name, module_name)
        if framework_result:
            return framework_result
        
        # Use heuristics for unknown modules
        heuristic_result = self._apply_heuristics(normalized_name, module_name)
        return heuristic_result
    
    def classify_call(self, call_name: str, module_context: str = "") -> ExternalClassification:
        """
        Classify a function/method call.
        
        Args:
            call_name: The function/method being called
            module_context: Context about which module this call is from
            
        Returns:
            ExternalClassification result
        """
        # For method calls, extract the module/class part
        if '.' in call_name:
            parts = call_name.split('.')
            # Try to classify based on the module/class prefix
            for i in range(len(parts)):
                prefix = '.'.join(parts[:i+1])
                classification = self.classify_import(prefix)
                if classification.classification != ExternalType.UNKNOWN:
                    return classification
        
        # If we have module context, use that for classification
        if module_context:
            return self.classify_import(module_context)
        
        # Default to unknown
        return ExternalClassification(
            module_name=call_name,
            classification=ExternalType.UNKNOWN,
            confidence=0.0,
            reasoning="No module context available for classification"
        )
    
    def should_create_actor(self, classification: ExternalClassification) -> bool:
        """
        Determine if an actor should be created for this external classification.
        
        Args:
            classification: The classification result
            
        Returns:
            True if an actor should be created
        """
        if classification.classification == ExternalType.SYSTEM_ACTOR:
            return classification.confidence >= self.confidence_threshold
        
        if classification.classification == ExternalType.FRAMEWORK and self.create_framework_actors:
            return classification.confidence >= self.confidence_threshold
        
        return False
    
    def _normalize_module_name(self, module_name: str) -> str:
        """Normalize module name for pattern matching."""
        # Remove common prefixes and clean up
        name = module_name.lower().strip()
        
        # Handle package.module patterns - try both full name and root package
        return name
    
    def _check_system_patterns(self, normalized_name: str, original_name: str) -> Optional[ExternalClassification]:
        """Check against known system actor patterns."""
        # Check custom patterns first
        patterns = {**self.SYSTEM_ACTOR_PATTERNS, **self.custom_system_patterns}
        
        for pattern, info in patterns.items():
            if normalized_name == pattern or normalized_name.startswith(pattern + '.'):
                return ExternalClassification(
                    module_name=original_name,
                    classification=ExternalType.SYSTEM_ACTOR,
                    confidence=info['confidence'],
                    actor_type=info.get('actor_type', info.get('type', 'ExternalSystem')),
                    reasoning=f"Matched system actor pattern: {pattern}",
                    should_create_actor=True
                )
        
        return None
    
    def _check_utility_patterns(self, normalized_name: str, original_name: str) -> Optional[ExternalClassification]:
        """Check against known utility library patterns."""
        # Check custom patterns first
        patterns = {**self.UTILITY_LIBRARY_PATTERNS, **self.custom_utility_patterns}
        
        for pattern, info in patterns.items():
            if normalized_name == pattern or normalized_name.startswith(pattern + '.'):
                return ExternalClassification(
                    module_name=original_name,
                    classification=ExternalType.UTILITY_LIBRARY,
                    confidence=info['confidence'],
                    reasoning=info['reasoning'],
                    should_create_actor=False
                )
        
        return None
    
    def _check_framework_patterns(self, normalized_name: str, original_name: str) -> Optional[ExternalClassification]:
        """Check against known framework patterns."""
        for pattern, info in self.FRAMEWORK_PATTERNS.items():
            if normalized_name == pattern or normalized_name.startswith(pattern + '.'):
                return ExternalClassification(
                    module_name=original_name,
                    classification=ExternalType.FRAMEWORK,
                    confidence=info['confidence'],
                    reasoning=info['reasoning'],
                    should_create_actor=self.create_framework_actors
                )
        
        return None
    
    def _apply_heuristics(self, normalized_name: str, original_name: str) -> ExternalClassification:
        """Apply heuristics to classify unknown modules."""
        confidence = 0.3  # Low confidence for heuristic classification
        reasoning = "Heuristic classification"
        
        # Heuristic 1: Database-related keywords
        db_keywords = ['db', 'database', 'sql', 'mongo', 'redis', 'elastic', 'search', 'index']
        if any(keyword in normalized_name for keyword in db_keywords):
            return ExternalClassification(
                module_name=original_name,
                classification=ExternalType.SYSTEM_ACTOR,
                confidence=confidence,
                actor_type='Database',
                reasoning=f"{reasoning}: contains database keywords",
                should_create_actor=False  # Low confidence, don't auto-create
            )
        
        # Heuristic 2: HTTP/API keywords
        api_keywords = ['http', 'api', 'client', 'rest', 'graphql', 'rpc']
        if any(keyword in normalized_name for keyword in api_keywords):
            return ExternalClassification(
                module_name=original_name,
                classification=ExternalType.SYSTEM_ACTOR,
                confidence=confidence,
                actor_type='HttpClient',
                reasoning=f"{reasoning}: contains API keywords",
                should_create_actor=False
            )
        
        # Heuristic 3: File system keywords
        fs_keywords = ['file', 'storage', 'disk', 'path', 'directory']
        if any(keyword in normalized_name for keyword in fs_keywords):
            return ExternalClassification(
                module_name=original_name,
                classification=ExternalType.SYSTEM_ACTOR,
                confidence=confidence,
                actor_type='FileSystem',
                reasoning=f"{reasoning}: contains file system keywords",
                should_create_actor=False
            )
        
        # Heuristic 4: Utility keywords
        util_keywords = ['util', 'helper', 'tool', 'lib', 'common', 'shared']
        if any(keyword in normalized_name for keyword in util_keywords):
            return ExternalClassification(
                module_name=original_name,
                classification=ExternalType.UTILITY_LIBRARY,
                confidence=confidence,
                reasoning=f"{reasoning}: contains utility keywords",
                should_create_actor=False
            )
        
        # Default: Unknown
        return ExternalClassification(
            module_name=original_name,
            classification=ExternalType.UNKNOWN,
            confidence=0.0,
            reasoning="No matching patterns or heuristics",
            should_create_actor=False
        )