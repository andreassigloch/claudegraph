#!/usr/bin/env python3
"""
Actor Enhancement Service for Code Architecture Analyzer

Enhances detected actors with meaningful names and descriptions using LLM analysis.
Transforms generic technical identifiers into business-contextual entities.
"""

import logging
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

from .client import LLMRequest

logger = logging.getLogger(__name__)


@dataclass
class ActorDetection:
    """Represents a detected actor before enhancement."""
    type: str  # HttpClient, Database, FileSystem, etc.
    library: str  # requests, sqlite3, os, etc.
    code_snippet: str
    function_context: str = ""
    file_path: str = ""
    confidence: float = 0.0
    url_or_target: str = ""  # API URL, DB file, etc.


@dataclass
class EnhancedActor:
    """Represents an enhanced actor with meaningful name and description."""
    name: str
    description: str
    enhanced: bool = True
    enhancement_confidence: float = 0.0
    original_name: str = ""
    cache_hit: bool = False
    processing_time: float = 0.0


@dataclass
class EnhancementCache:
    """Simple in-memory cache for actor enhancements."""
    entries: Dict[str, EnhancedActor] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    max_size: int = 1000
    
    def get_cache_key(self, detection: ActorDetection) -> str:
        """Generate cache key from detection pattern."""
        # Create a hash from the key characteristics
        key_data = f"{detection.type}:{detection.library}:{detection.url_or_target}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def get(self, detection: ActorDetection) -> Optional[EnhancedActor]:
        """Get cached enhancement if available."""
        key = self.get_cache_key(detection)
        if key in self.entries:
            self.hits += 1
            cached = self.entries[key]
            cached.cache_hit = True
            return cached
        self.misses += 1
        return None
    
    def put(self, detection: ActorDetection, enhanced: EnhancedActor):
        """Store enhancement in cache."""
        if len(self.entries) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.entries))
            del self.entries[oldest_key]
        
        key = self.get_cache_key(detection)
        self.entries[key] = enhanced
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "entries": len(self.entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


class ActorEnhancementService:
    """
    Service for enhancing actor names and descriptions using LLM analysis.
    
    Transforms generic technical actor identifiers into meaningful business entities
    by analyzing code context and generating appropriate names and descriptions.
    """
    
    def __init__(self, config: Dict[str, Any], llm_client=None):
        """Initialize the enhancement service."""
        self.config = config
        self.llm_client = llm_client
        self.cache = EnhancementCache()
        
        # Enhancement settings
        enhancement_config = config.get('llm', {}).get('actor_enhancement', {})
        self.enabled = enhancement_config.get('enabled', True)
        self.timeout_seconds = enhancement_config.get('timeout_seconds', 5)
        self.max_retries = enhancement_config.get('max_retries', 2)
        self.cache_enabled = enhancement_config.get('cache_enabled', True)
        self.fallback_on_failure = enhancement_config.get('fallback_on_failure', True)
        self.confidence_threshold = enhancement_config.get('enhancement_confidence_threshold', 0.7)
        
        # Cache settings
        self.cache.max_size = enhancement_config.get('cache_size_limit', 1000)
        
        logger.info(f"Actor enhancement service initialized (enabled: {self.enabled})")
    
    def enhance_actor(self, detection: ActorDetection, original_name: str) -> EnhancedActor:
        """
        Enhance a single actor detection with meaningful name and description.
        
        Args:
            detection: The actor detection to enhance
            original_name: The original generic name
            
        Returns:
            EnhancedActor with improved name and description
        """
        start_time = time.time()
        
        # Return original if enhancement disabled
        if not self.enabled:
            return self._create_fallback_actor(original_name, detection, start_time)
        
        # Check cache first
        if self.cache_enabled:
            cached = self.cache.get(detection)
            if cached:
                logger.debug(f"Cache hit for actor: {cached.name}")
                return cached
        
        # Try LLM enhancement
        try:
            enhanced = self._enhance_with_llm(detection, original_name)
            enhanced.processing_time = time.time() - start_time
            enhanced.original_name = original_name
            
            # Cache successful enhancement
            if self.cache_enabled and enhanced.enhancement_confidence >= self.confidence_threshold:
                self.cache.put(detection, enhanced)
            
            logger.debug(f"Enhanced actor: {original_name} -> {enhanced.name}")
            return enhanced
            
        except Exception as e:
            logger.warning(f"Actor enhancement failed for {original_name}: {str(e)}")
            
            if self.fallback_on_failure:
                return self._create_fallback_actor(original_name, detection, start_time)
            else:
                raise e
    
    def enhance_actors(self, detections: List[Tuple[ActorDetection, str]]) -> List[EnhancedActor]:
        """
        Enhance multiple actor detections.
        
        Args:
            detections: List of (detection, original_name) tuples
            
        Returns:
            List of enhanced actors
        """
        enhanced_actors = []
        
        for detection, original_name in detections:
            try:
                enhanced = self.enhance_actor(detection, original_name)
                enhanced_actors.append(enhanced)
            except Exception as e:
                logger.error(f"Failed to enhance actor {original_name}: {str(e)}")
                # Add fallback actor
                enhanced_actors.append(
                    self._create_fallback_actor(original_name, detection, 0.0)
                )
        
        logger.info(f"Enhanced {len(enhanced_actors)} actors")
        return enhanced_actors
    
    def _enhance_with_llm(self, detection: ActorDetection, original_name: str) -> EnhancedActor:
        """Enhance actor using LLM analysis."""
        if not self.llm_client:
            raise ValueError("LLM client not available for enhancement")
        
        # Build prompt
        prompt = self._build_enhancement_prompt(detection)
        
        # Call LLM with retries
        for attempt in range(self.max_retries + 1):
            try:
                # Create LLM request
                request = LLMRequest(
                    prompt=prompt,
                    model=getattr(self.llm_client, 'model', 'claude-3-sonnet-20240229'),
                    max_tokens=100,
                    temperature=0.1,
                    timeout=self.timeout_seconds
                )
                
                response = self.llm_client.call(request)
                
                # Parse response (response.content contains the text)
                enhanced = self._parse_llm_response(response.content, original_name)
                if enhanced:
                    return enhanced
                
            except Exception as e:
                logger.debug(f"LLM call attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries:
                    raise e
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        raise ValueError("All LLM enhancement attempts failed")
    
    def _build_enhancement_prompt(self, detection: ActorDetection) -> str:
        """Build LLM prompt for actor enhancement."""
        # Extract key context information
        url_info = f"\nURL/Target: {detection.url_or_target}" if detection.url_or_target else ""
        function_info = f"\nFunction: {detection.function_context}" if detection.function_context else ""
        file_info = f"\nFile: {Path(detection.file_path).name}" if detection.file_path else ""
        
        prompt = f"""Analyze this external system interaction and provide a meaningful name and description:

Code: {detection.code_snippet}
Library: {detection.library}
Type: {detection.type}{url_info}{function_info}{file_info}

Provide:
1. Name: Concise identifier (max 20 chars, PascalCase, no spaces)
2. Description: What this external actor does (max 80 chars)

Format: Name|Description
Examples:
- StripePaymentAPI|Payment processing service for credit card transactions
- UserDatabase|Local database storing user profiles and preferences
- ConfigurationFiles|Application settings and runtime configuration

Response:"""
        
        return prompt
    
    def _parse_llm_response(self, response: str, original_name: str) -> Optional[EnhancedActor]:
        """Parse LLM response into EnhancedActor."""
        try:
            response = response.strip()
            name = ""
            description = ""
            
            # Handle different response formats
            if '|' in response:
                parts = response.split('|', 1)
                name = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ""
            elif ':' in response and '\n' in response:
                # Handle "Name: X\nDescription: Y" format
                lines = response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('Name:'):
                        name = line.split(':', 1)[1].strip()
                    elif line.startswith('Description:'):
                        description = line.split(':', 1)[1].strip()
                    elif line.startswith('1.') and not name:
                        name = line.split('.', 1)[1].strip()
                    elif line.startswith('2.') and not description:
                        description = line.split('.', 1)[1].strip()
            elif response.startswith('1.') and '\n' in response:
                # Handle numbered format without colons
                lines = response.split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line.startswith('1.') and not name:
                        name = line.split('.', 1)[1].strip()
                    elif line.startswith('2.') and not description:
                        description = line.split('.', 1)[1].strip()
            else:
                # Single line response, use as name
                name = response
                description = f"External {original_name.lower()} component"
            
            # Validate name
            if not name or len(name) > 25:
                logger.warning(f"Invalid name from LLM: '{name}'")
                return None
            
            # Clean up name (ensure PascalCase, no spaces)
            name = self._clean_actor_name(name)
            
            # Validate description
            if len(description) > 100:
                description = description[:97] + "..."
            
            # Determine confidence based on response quality
            confidence = self._calculate_enhancement_confidence(name, description, original_name)
            
            return EnhancedActor(
                name=name,
                description=description,
                enhanced=True,
                enhancement_confidence=confidence
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {str(e)}")
            return None
    
    def _clean_actor_name(self, name: str) -> str:
        """Clean and format actor name."""
        # Remove quotes and unwanted characters
        name = name.strip('"\'`()[]{}')
        
        # Handle PascalCase conversion more carefully
        if '_' in name or '-' in name or ' ' in name:
            # Only convert if it has separators
            words = name.replace('_', ' ').replace('-', ' ').split()
            if words:
                name = ''.join(word.capitalize() for word in words)
        else:
            # If no separators, preserve existing case but ensure first letter is uppercase
            if name:
                name = name[0].upper() + name[1:] if len(name) > 1 else name.upper()
        
        # Ensure it starts with a letter
        if name and not name[0].isalpha():
            # Remove leading numbers/symbols and prefix with External
            clean_name = ''.join(c for c in name if c.isalnum())
            name = 'External' + clean_name if clean_name else 'ExternalActor'
        
        # Limit length
        if len(name) > 20:
            name = name[:20]
        
        return name or 'UnknownActor'
    
    def _calculate_enhancement_confidence(self, name: str, description: str, original_name: str) -> float:
        """Calculate confidence score for the enhancement."""
        confidence = 0.5  # Base confidence
        
        # Bonus for meaningful name (not generic)
        if name.lower() not in ['unknown', 'external', 'actor', 'system']:
            confidence += 0.2
        
        # Bonus for specific description
        if len(description) > 20 and 'unknown' not in description.lower():
            confidence += 0.2
        
        # Bonus for avoiding original generic terms
        if original_name.lower() not in name.lower():
            confidence += 0.1
        
        # Bonus for business-relevant terms
        business_terms = ['api', 'service', 'database', 'storage', 'system', 'client']
        if any(term in name.lower() or term in description.lower() for term in business_terms):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _create_fallback_actor(self, original_name: str, detection: ActorDetection, start_time: float) -> EnhancedActor:
        """Create fallback actor when enhancement fails."""
        # Try to improve the description slightly without LLM
        improved_desc = self._improve_description_fallback(detection)
        
        return EnhancedActor(
            name=original_name,
            description=improved_desc,
            enhanced=False,
            enhancement_confidence=0.0,
            original_name=original_name,
            processing_time=time.time() - start_time
        )
    
    def _improve_description_fallback(self, detection: ActorDetection) -> str:
        """Improve description without LLM for fallback."""
        base_desc = f"{detection.type} using {detection.library}"
        
        # Add context if available
        if detection.url_or_target:
            if 'api.' in detection.url_or_target.lower():
                base_desc += " for API communication"
            elif any(db in detection.url_or_target.lower() for db in ['db', 'database', 'sql']):
                base_desc += " for data storage"
        
        if detection.function_context:
            if 'payment' in detection.function_context.lower():
                base_desc += " in payment processing"
            elif 'user' in detection.function_context.lower():
                base_desc += " for user management"
            elif 'config' in detection.function_context.lower():
                base_desc += " for configuration"
        
        return base_desc
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get enhancement service statistics."""
        cache_stats = self.cache.get_stats() if self.cache_enabled else {}
        
        return {
            "enabled": self.enabled,
            "cache_enabled": self.cache_enabled,
            "cache_stats": cache_stats,
            "config": {
                "timeout_seconds": self.timeout_seconds,
                "max_retries": self.max_retries,
                "confidence_threshold": self.confidence_threshold
            }
        }