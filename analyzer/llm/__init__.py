#!/usr/bin/env python3
"""
LLM Module for Code Architecture Analyzer

Provides LLM integration capabilities including client management,
batch processing, and classification services for ambiguous code patterns.
"""

from .client import (
    LLMManager,
    LLMClientFactory,
    LLMClientBase,
    LocalLLMClient,
    LLMRequest,
    LLMResponse,
    ClassificationResult,
    LLMClientError
)

from .batch_processor import (
    LLMBatchProcessor,
    ClassificationItem,
    BatchRequest,
    BatchResult
)

__all__ = [
    # Client components
    'LLMManager',
    'LLMClientFactory', 
    'LLMClientBase',
    'LocalLLMClient',
    'LLMRequest',
    'LLMResponse',
    'ClassificationResult',
    'LLMClientError',
    
    # Batch processing components
    'LLMBatchProcessor',
    'ClassificationItem',
    'BatchRequest',
    'BatchResult'
]

# Module-level convenience functions
def create_llm_manager(config=None):
    """Create LLM manager with configuration."""
    return LLMManager(config)

def create_batch_processor(config=None):
    """Create batch processor with configuration."""
    return LLMBatchProcessor(config)

# Version info
__version__ = '1.0.0'