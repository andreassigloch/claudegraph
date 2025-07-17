#!/usr/bin/env python3
"""
LLM Batch Processor for Code Architecture Analyzer

Handles efficient batch processing of ambiguous code patterns for AI-assisted
classification with cost optimization and rate limiting.
"""

import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .client import LLMManager, LLMRequest, LLMResponse, ClassificationResult

logger = logging.getLogger(__name__)


@dataclass
class ClassificationItem:
    """Item to be classified by LLM."""
    id: str
    code_snippet: str
    context: Dict[str, Any]
    evidence: Dict[str, Any]
    module_name: str
    function_name: Optional[str] = None
    confidence: float = 0.0
    pattern_name: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchRequest:
    """Batch of items for LLM processing."""
    batch_id: str
    items: List[ClassificationItem]
    prompt_template: str
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 500
    created_at: float = field(default_factory=time.time)


@dataclass
class BatchResult:
    """Result of batch processing."""
    batch_id: str
    success: bool
    results: Dict[str, ClassificationResult] = field(default_factory=dict)
    failed_items: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    cost_estimate: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMBatchProcessor:
    """
    Efficient batch processor for LLM-based code classification.
    
    Handles batching, rate limiting, cost optimization, and result processing
    for large-scale code pattern classification.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize batch processor with configuration."""
        self.config = config or {}
        
        # Batch processing settings
        batch_config = self.config.get('llm', {}).get('batch', {})
        self.max_batch_size = batch_config.get('max_batch_size', 10)
        self.max_concurrent_batches = batch_config.get('max_concurrent_batches', 3)
        self.request_delay = batch_config.get('request_delay', 1.0)  # seconds between requests
        self.max_retries = batch_config.get('max_retries', 3)
        
        # Cost optimization
        self.daily_cost_limit = batch_config.get('daily_cost_limit', 10.0)  # USD
        self.current_daily_cost = 0.0
        
        # Rate limiting
        self.requests_per_minute = batch_config.get('requests_per_minute', 60)
        self.request_timestamps = []
        
        # Initialize LLM manager
        self.llm_manager = LLMManager(self.config)
        
        # Cache for similar patterns
        self.classification_cache = {}
        
        logger.info("LLM Batch Processor initialized")
    
    def process_detection_matches(
        self, 
        detection_matches: List[Any],
        threshold: float = 0.7
    ) -> Dict[str, ClassificationResult]:
        """
        Process detection matches that need LLM classification.
        
        Args:
            detection_matches: List of ambiguous detection matches
            threshold: Confidence threshold for requiring LLM classification
            
        Returns:
            Dictionary mapping match IDs to classification results
        """
        # Filter matches that need LLM classification
        ambiguous_matches = [
            match for match in detection_matches 
            if match.confidence < threshold
        ]
        
        if not ambiguous_matches:
            logger.info("No matches require LLM classification")
            return {}
        
        logger.info(f"Processing {len(ambiguous_matches)} ambiguous matches with LLM")
        
        # Convert to classification items
        classification_items = self._create_classification_items(ambiguous_matches)
        
        # Check cache first
        cached_results, uncached_items = self._check_cache(classification_items)
        
        # Process uncached items in batches
        batch_results = {}
        if uncached_items:
            batch_results = self._process_batches(uncached_items)
        
        # Combine cached and batch results
        final_results = {**cached_results, **batch_results}
        
        # Update cache with new results
        self._update_cache(batch_results)
        
        logger.info(f"LLM classification completed: {len(final_results)} results")
        return final_results
    
    def _create_classification_items(self, detection_matches: List[Any]) -> List[ClassificationItem]:
        """Convert detection matches to classification items."""
        items = []
        
        for i, match in enumerate(detection_matches):
            item = ClassificationItem(
                id=f"match_{i}_{int(time.time())}",
                code_snippet=self._extract_code_snippet(match),
                context={
                    "module_name": match.module_name,
                    "function_name": match.function_name,
                    "line_numbers": match.line_numbers,
                    "actor_type": match.actor_type.value if match.actor_type else "Unknown"
                },
                evidence=match.evidence,
                module_name=match.module_name or "unknown",
                function_name=match.function_name,
                confidence=match.confidence,
                pattern_name=match.pattern_name,
                metadata={
                    "original_match": match
                }
            )
            items.append(item)
        
        return items
    
    def _extract_code_snippet(self, match: Any) -> str:
        """Extract relevant code snippet from detection match."""
        # Try to get code from evidence
        if 'code_snippet' in match.evidence:
            return match.evidence['code_snippet']
        
        # Fallback to constructing from available information
        snippet_parts = []
        
        if 'imports' in match.evidence:
            for imp in match.evidence['imports'][:3]:  # Limit to first 3 imports
                snippet_parts.append(f"import {imp}")
        
        if 'function_calls' in match.evidence:
            for call in match.evidence['function_calls'][:3]:
                snippet_parts.append(f"{call}()")
        
        if 'decorators' in match.evidence:
            for decorator in match.evidence['decorators'][:2]:
                snippet_parts.append(f"@{decorator}")
        
        return '\n'.join(snippet_parts) or "# Code snippet not available"
    
    def _check_cache(self, items: List[ClassificationItem]) -> Tuple[Dict[str, ClassificationResult], List[ClassificationItem]]:
        """Check cache for existing classifications."""
        cached_results = {}
        uncached_items = []
        
        for item in items:
            cache_key = self._generate_cache_key(item)
            
            if cache_key in self.classification_cache:
                cached_results[item.id] = self.classification_cache[cache_key]
                logger.debug(f"Using cached result for {item.id}")
            else:
                uncached_items.append(item)
        
        logger.info(f"Cache: {len(cached_results)} hits, {len(uncached_items)} misses")
        return cached_results, uncached_items
    
    def _generate_cache_key(self, item: ClassificationItem) -> str:
        """Generate cache key for classification item."""
        # Create key based on code pattern and evidence
        key_components = [
            item.code_snippet,
            str(sorted(item.evidence.items())),
            item.pattern_name
        ]
        return hash(tuple(key_components))
    
    def _process_batches(self, items: List[ClassificationItem]) -> Dict[str, ClassificationResult]:
        """Process classification items in optimized batches."""
        # Check cost and rate limits
        if not self._check_limits():
            logger.warning("Rate or cost limits exceeded, skipping LLM processing")
            return {}
        
        # Create batches
        batches = self._create_batches(items)
        logger.info(f"Created {len(batches)} batches for processing")
        
        all_results = {}
        
        # Process batches with concurrency control
        with ThreadPoolExecutor(max_workers=self.max_concurrent_batches) as executor:
            # Submit batch processing tasks
            future_to_batch = {
                executor.submit(self._process_single_batch, batch): batch.batch_id
                for batch in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_result = future.result()
                    if batch_result.success:
                        all_results.update(batch_result.results)
                        self.current_daily_cost += batch_result.cost_estimate
                    else:
                        logger.error(f"Batch {batch_id} failed: {batch_result.error_message}")
                        
                except Exception as e:
                    logger.error(f"Batch {batch_id} processing error: {e}")
                
                # Rate limiting delay
                time.sleep(self.request_delay)
        
        return all_results
    
    def _create_batches(self, items: List[ClassificationItem]) -> List[BatchRequest]:
        """Create optimized batches from classification items."""
        batches = []
        
        for i in range(0, len(items), self.max_batch_size):
            batch_items = items[i:i + self.max_batch_size]
            
            batch = BatchRequest(
                batch_id=f"batch_{i//self.max_batch_size}_{int(time.time())}",
                items=batch_items,
                prompt_template=self._get_classification_prompt(),
                model=self.config.get('llm', {}).get('model', 'gpt-3.5-turbo')
            )
            batches.append(batch)
        
        return batches
    
    def _process_single_batch(self, batch: BatchRequest) -> BatchResult:
        """Process a single batch of classification items."""
        start_time = time.time()
        
        try:
            # Build combined prompt for batch
            combined_prompt = self._build_batch_prompt(batch)
            
            # Create LLM request
            llm_request = LLMRequest(
                prompt=combined_prompt,
                model=batch.model,
                temperature=batch.temperature,
                max_tokens=batch.max_tokens * len(batch.items),  # Scale tokens for batch
                timeout=60
            )
            
            # Make LLM call with retries
            response = self._call_with_retries(llm_request)
            
            if not response:
                return BatchResult(
                    batch_id=batch.batch_id,
                    success=False,
                    error_message="LLM call failed after retries",
                    processing_time=time.time() - start_time
                )
            
            # Parse batch response
            results = self._parse_batch_response(response, batch.items)
            
            # Estimate cost
            cost_estimate = self._estimate_cost(response, batch.model)
            
            return BatchResult(
                batch_id=batch.batch_id,
                success=True,
                results=results,
                processing_time=time.time() - start_time,
                cost_estimate=cost_estimate,
                metadata={
                    "item_count": len(batch.items),
                    "model": batch.model,
                    "tokens_used": response.usage.get('total_tokens', 0) if response.usage else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Batch {batch.batch_id} processing failed: {e}")
            return BatchResult(
                batch_id=batch.batch_id,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _build_batch_prompt(self, batch: BatchRequest) -> str:
        """Build combined prompt for batch processing."""
        prompt_parts = [
            "You are an expert code analyzer. Classify the following code patterns into actor types.",
            "For each item, respond with JSON in this format:",
            '{"item_id": "ID", "classification": "ActorType", "confidence": 0.8, "reasoning": "explanation"}',
            "",
            "Actor types: HttpClient, Database, FileSystem, WebEndpoint, MessageQueue, ConfigManager, CloudService, ExternalApi, Cache, Monitor, Unknown",
            "",
            "Code patterns to classify:"
        ]
        
        for item in batch.items:
            prompt_parts.extend([
                f"",
                f"ITEM_ID: {item.id}",
                f"MODULE: {item.module_name}",
                f"FUNCTION: {item.function_name or 'N/A'}",
                f"CODE:",
                f"```python",
                item.code_snippet,
                f"```",
                f"EVIDENCE: {json.dumps(item.evidence, indent=2)}",
                f"PATTERN: {item.pattern_name}",
                "---"
            ])
        
        prompt_parts.extend([
            "",
            "Provide a JSON array with one classification object per item.",
            "Focus on the imports, function calls, and patterns to determine the actor type.",
            "Be conservative with confidence scores - use 0.9+ only for very clear patterns."
        ])
        
        return '\n'.join(prompt_parts)
    
    def _call_with_retries(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Make LLM call with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Update rate limiting
                self._update_rate_limit()
                
                # Make the call
                response = self.llm_manager.call(request)
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                
                # Exponential backoff
                if attempt < self.max_retries - 1:
                    delay = (2 ** attempt) * self.request_delay
                    time.sleep(delay)
        
        logger.error(f"LLM call failed after {self.max_retries} attempts: {last_error}")
        return None
    
    def _parse_batch_response(self, response: LLMResponse, items: List[ClassificationItem]) -> Dict[str, ClassificationResult]:
        """Parse batch response into individual classification results."""
        results = {}
        
        try:
            # Try to parse as JSON array
            content = response.content.strip()
            
            # Handle different response formats
            if content.startswith('[') and content.endswith(']'):
                # Direct JSON array
                classifications = json.loads(content)
            else:
                # Try to extract JSON from markdown or other formatting
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    classifications = json.loads(json_match.group())
                else:
                    # Fallback: split by item and parse individually
                    classifications = self._parse_unstructured_response(content, items)
            
            # Process each classification
            for classification in classifications:
                if isinstance(classification, dict) and 'item_id' in classification:
                    item_id = classification['item_id']
                    
                    result = ClassificationResult(
                        classification=classification.get('classification', 'Unknown'),
                        confidence=float(classification.get('confidence', 0.5)),
                        reasoning=classification.get('reasoning', 'No reasoning provided'),
                        item_id=item_id
                    )
                    
                    results[item_id] = result
            
            # Fill missing items with default results
            for item in items:
                if item.id not in results:
                    results[item.id] = ClassificationResult(
                        classification='Unknown',
                        confidence=0.1,
                        reasoning='Failed to parse LLM response for this item',
                        item_id=item.id
                    )
            
        except Exception as e:
            logger.error(f"Failed to parse batch response: {e}")
            logger.debug(f"Raw response: {response.content}")
            
            # Return default results for all items
            for item in items:
                results[item.id] = ClassificationResult(
                    classification='Unknown',
                    confidence=0.1,
                    reasoning=f'Response parsing failed: {e}',
                    item_id=item.id
                )
        
        return results
    
    def _parse_unstructured_response(self, content: str, items: List[ClassificationItem]) -> List[Dict]:
        """Parse unstructured response as fallback."""
        classifications = []
        
        # Simple pattern matching for fallback
        lines = content.split('\n')
        current_item = None
        
        for line in lines:
            line = line.strip()
            
            # Look for item IDs
            for item in items:
                if item.id in line:
                    current_item = item.id
                    break
            
            # Look for classification keywords
            if current_item:
                for actor_type in ['HttpClient', 'Database', 'FileSystem', 'WebEndpoint', 'MessageQueue']:
                    if actor_type.lower() in line.lower():
                        classifications.append({
                            'item_id': current_item,
                            'classification': actor_type,
                            'confidence': 0.6,
                            'reasoning': 'Extracted from unstructured response'
                        })
                        current_item = None
                        break
        
        return classifications
    
    def _estimate_cost(self, response: LLMResponse, model: str) -> float:
        """Estimate API cost for the response."""
        if not response.usage:
            return 0.01  # Default small cost
        
        # Rough cost estimates (as of 2024)
        cost_per_1k_tokens = {
            'gpt-3.5-turbo': 0.002,
            'gpt-4': 0.03,
            'gpt-4-turbo': 0.01,
            'local': 0.0  # No cost for local models
        }
        
        rate = cost_per_1k_tokens.get(model, 0.002)
        total_tokens = response.usage.get('total_tokens', 1000)
        
        return (total_tokens / 1000) * rate
    
    def _check_limits(self) -> bool:
        """Check if rate and cost limits allow processing."""
        # Check daily cost limit
        if self.current_daily_cost >= self.daily_cost_limit:
            logger.warning(f"Daily cost limit reached: ${self.current_daily_cost:.2f}")
            return False
        
        # Check rate limit
        current_time = time.time()
        recent_requests = [t for t in self.request_timestamps if current_time - t < 60]
        
        if len(recent_requests) >= self.requests_per_minute:
            logger.warning(f"Rate limit reached: {len(recent_requests)} requests in last minute")
            return False
        
        return True
    
    def _update_rate_limit(self):
        """Update rate limiting timestamps."""
        self.request_timestamps.append(time.time())
        # Keep only last minute of timestamps
        cutoff = time.time() - 60
        self.request_timestamps = [t for t in self.request_timestamps if t > cutoff]
    
    def _update_cache(self, results: Dict[str, ClassificationResult]):
        """Update classification cache with new results."""
        # For now, simple in-memory cache
        # In production, this could be Redis or persistent storage
        for item_id, result in results.items():
            if result.confidence > 0.7:  # Only cache high-confidence results
                # Note: We'd need to store the original item to generate proper cache key
                pass
    
    def _get_classification_prompt(self) -> str:
        """Get the base prompt template for classification."""
        return """
You are an expert code analyzer specializing in identifying external system interactions.
Analyze the provided code patterns and classify them into the most appropriate actor type.

Actor Types:
- HttpClient: HTTP requests, API calls, web service interactions
- Database: SQL queries, ORM operations, database connections
- FileSystem: File I/O operations, path manipulations, file system access
- WebEndpoint: HTTP endpoints, API routes, web service definitions
- MessageQueue: Message queue operations, pub/sub patterns
- ConfigManager: Configuration file access, environment variables
- CloudService: Cloud provider APIs, cloud service integrations
- ExternalApi: Third-party API integrations, external service calls
- Cache: Caching operations, cache storage access
- Monitor: Logging, monitoring, metrics collection
- Unknown: Patterns that don't clearly fit other categories

Consider:
1. Import statements and their usage patterns
2. Function calls and method invocations
3. Decorators and annotations
4. String patterns that suggest external interactions
5. Context clues from variable names and code structure

Provide high confidence (0.8+) only for clear, unambiguous patterns.
Use medium confidence (0.5-0.7) for probable but not certain classifications.
Use low confidence (<0.5) for uncertain or ambiguous patterns.
"""
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            "daily_cost": self.current_daily_cost,
            "daily_cost_limit": self.daily_cost_limit,
            "recent_requests": len([t for t in self.request_timestamps if time.time() - t < 60]),
            "requests_per_minute_limit": self.requests_per_minute,
            "cache_size": len(self.classification_cache),
            "batch_settings": {
                "max_batch_size": self.max_batch_size,
                "max_concurrent_batches": self.max_concurrent_batches,
                "request_delay": self.request_delay,
                "max_retries": self.max_retries
            }
        }
    
    def reset_daily_cost(self):
        """Reset daily cost counter (typically called daily)."""
        self.current_daily_cost = 0.0
        logger.info("Daily cost counter reset")
    
    def clear_cache(self):
        """Clear the classification cache."""
        self.classification_cache.clear()
        logger.info("Classification cache cleared")