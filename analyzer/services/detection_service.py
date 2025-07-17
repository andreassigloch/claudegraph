#!/usr/bin/env python3
"""
Detection Service for Code Architecture Analyzer

Coordinates actor detection across multiple pattern matchers and provides
unified detection interface. Abstracts the complexity of detection coordination
from analysis services.
"""

import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from ..core.ast_parser import ASTParseResult
from ..detection.pattern_matcher import PatternMatcher, ActorDetectionResult
from ..llm.actor_enhancer import ActorEnhancementService
from ..llm.client import LLMManager
from .cache_service import CacheService

logger = logging.getLogger(__name__)


class DetectionStrategy(ABC):
    """Abstract interface for detection strategies."""
    
    @abstractmethod
    def detect_actors(self, ast_results: List[ASTParseResult]) -> List[ActorDetectionResult]:
        """Detect actors using this strategy."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name."""
        pass


class PatternMatchingStrategy(DetectionStrategy):
    """Detection strategy using pattern matching."""
    
    def __init__(self, pattern_matcher: PatternMatcher):
        self.pattern_matcher = pattern_matcher
    
    def detect_actors(self, ast_results: List[ASTParseResult]) -> List[ActorDetectionResult]:
        """Detect actors using pattern matching."""
        actor_results = []
        
        for ast_result in ast_results:
            try:
                actor_result = self.pattern_matcher.detect_actors(ast_result)
                actor_results.append(actor_result)
            except Exception as e:
                logger.warning(f"Pattern matching failed for {ast_result.module_name}: {e}")
                # Create empty result to maintain consistency
                actor_results.append(ActorDetectionResult(
                    module_name=ast_result.module_name,
                    detected_actors=[],
                    high_confidence_matches=[],
                    ambiguous_matches=[]
                ))
        
        return actor_results
    
    def get_name(self) -> str:
        return "PatternMatching"


class HybridDetectionStrategy(DetectionStrategy):
    """Detection strategy combining multiple approaches."""
    
    def __init__(self, 
                 pattern_matcher: PatternMatcher,
                 enhancement_service: Optional[ActorEnhancementService] = None):
        self.pattern_matcher = pattern_matcher
        self.enhancement_service = enhancement_service
    
    def detect_actors(self, ast_results: List[ASTParseResult]) -> List[ActorDetectionResult]:
        """Detect actors using hybrid approach."""
        # Start with pattern matching
        actor_results = []
        
        for ast_result in ast_results:
            try:
                # Primary detection via pattern matching
                actor_result = self.pattern_matcher.detect_actors(ast_result)
                
                # Enhance with LLM if available and there are ambiguous matches
                if (self.enhancement_service and 
                    actor_result.ambiguous_matches and 
                    len(actor_result.ambiguous_matches) > 0):
                    
                    try:
                        enhanced_result = self.enhancement_service.enhance_actors(
                            actor_result.ambiguous_matches,
                            ast_result
                        )
                        
                        # Merge enhanced results
                        if enhanced_result:
                            actor_result.high_confidence_matches.extend(
                                enhanced_result.get('enhanced_actors', [])
                            )
                            # Remove enhanced actors from ambiguous list
                            enhanced_names = {a.get('name', '') for a in enhanced_result.get('enhanced_actors', [])}
                            actor_result.ambiguous_matches = [
                                m for m in actor_result.ambiguous_matches
                                if m.actor_name not in enhanced_names
                            ]
                    
                    except Exception as e:
                        logger.warning(f"LLM enhancement failed for {ast_result.module_name}: {e}")
                
                actor_results.append(actor_result)
                
            except Exception as e:
                logger.warning(f"Hybrid detection failed for {ast_result.module_name}: {e}")
                actor_results.append(ActorDetectionResult(
                    module_name=ast_result.module_name,
                    detected_actors=[],
                    high_confidence_matches=[],
                    ambiguous_matches=[]
                ))
        
        return actor_results
    
    def get_name(self) -> str:
        return "Hybrid"


class DetectionService:
    """
    Service that coordinates actor detection across the codebase.
    
    Provides unified interface for actor detection with pluggable strategies
    and caching support. Abstracts detection complexity from analysis services.
    """
    
    def __init__(self, 
                 cache_service: CacheService,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize detection service."""
        self.cache_service = cache_service
        self.config = config or {}
        
        # Detection configuration
        detection_config = self.config.get('detection', {})
        self.strategy_type = detection_config.get('strategy', 'pattern_matching')
        self.enable_caching = detection_config.get('enable_caching', True)
        self.cache_ttl = detection_config.get('cache_ttl', 1800)  # 30 minutes
        
        # Initialize LLM components if needed
        self.llm_client = None
        self.enhancement_service = None
        
        if self.strategy_type == 'hybrid':
            try:
                self.llm_client = LLMManager(self.config).get_client()
                if self.llm_client:
                    self.enhancement_service = ActorEnhancementService(self.config, self.llm_client)
                    logger.info("LLM enhancement service initialized for hybrid detection")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM components: {e}")
                self.strategy_type = 'pattern_matching'  # Fallback
        
        # Initialize detection strategy
        self.strategy = self._create_detection_strategy()
        
        logger.info(f"Detection service initialized with {self.strategy.get_name()} strategy")
    
    def detect_actors(self, 
                     ast_results: List[ASTParseResult],
                     config: Optional[Dict[str, Any]] = None) -> List[ActorDetectionResult]:
        """
        Detect actors across all AST results.
        
        Args:
            ast_results: List of parsed AST results
            config: Optional configuration override
            
        Returns:
            List of actor detection results
        """
        if not ast_results:
            logger.warning("No AST results provided for actor detection")
            return []
        
        # Check cache if enabled
        if self.enable_caching:
            cache_key = self._generate_cache_key(ast_results)
            cached_results = self.cache_service.get(cache_key)
            
            if cached_results:
                logger.debug(f"Using cached actor detection results for {len(ast_results)} modules")
                return cached_results
        
        # Perform detection
        logger.info(f"Starting actor detection for {len(ast_results)} modules using {self.strategy.get_name()} strategy")
        
        try:
            actor_results = self.strategy.detect_actors(ast_results)
            
            # Cache results if enabled
            if self.enable_caching and actor_results:
                cache_key = self._generate_cache_key(ast_results)
                self.cache_service.put(cache_key, actor_results, ttl=self.cache_ttl)
            
            # Log detection summary
            total_actors = sum(len(result.detected_actors) for result in actor_results)
            total_high_confidence = sum(len(result.high_confidence_matches) for result in actor_results)
            total_ambiguous = sum(len(result.ambiguous_matches) for result in actor_results)
            
            logger.info(f"Actor detection completed: {total_actors} total actors, "
                       f"{total_high_confidence} high confidence, {total_ambiguous} ambiguous")
            
            return actor_results
            
        except Exception as e:
            logger.error(f"Actor detection failed: {e}")
            return []
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection service statistics."""
        stats = {
            'strategy': self.strategy.get_name(),
            'caching_enabled': self.enable_caching,
            'llm_available': self.llm_client is not None,
            'enhancement_available': self.enhancement_service is not None
        }
        
        # Add cache statistics if available
        if self.enable_caching:
            cache_stats = self.cache_service.get_statistics()
            stats['cache_statistics'] = cache_stats
        
        return stats
    
    def validate_configuration(self) -> List[str]:
        """Validate detection service configuration."""
        issues = []
        
        # Check strategy configuration
        valid_strategies = ['pattern_matching', 'hybrid']
        if self.strategy_type not in valid_strategies:
            issues.append(f"Invalid detection strategy: {self.strategy_type}")
        
        # Check LLM configuration for hybrid strategy
        if self.strategy_type == 'hybrid' and not self.llm_client:
            issues.append("Hybrid strategy requires LLM client but none available")
        
        # Check cache configuration
        if self.enable_caching and not self.cache_service:
            issues.append("Caching enabled but no cache service available")
        
        return issues
    
    def _create_detection_strategy(self) -> DetectionStrategy:
        """Create detection strategy based on configuration."""
        # Initialize pattern matcher
        pattern_matcher = PatternMatcher(self.config, self.enhancement_service)
        
        if self.strategy_type == 'hybrid' and self.enhancement_service:
            return HybridDetectionStrategy(pattern_matcher, self.enhancement_service)
        else:
            return PatternMatchingStrategy(pattern_matcher)
    
    def _generate_cache_key(self, ast_results: List[ASTParseResult]) -> str:
        """Generate cache key for AST results."""
        # Create key based on module names and file modification times
        key_parts = []
        
        for ast_result in ast_results[:10]:  # Limit to first 10 for key size
            key_parts.append(f"{ast_result.module_name}:{len(ast_result.functions)}")
        
        # Add strategy and configuration hash
        key_parts.append(f"strategy:{self.strategy_type}")
        
        cache_key = "detection:" + ":".join(key_parts)
        
        # Truncate if too long
        if len(cache_key) > 200:
            import hashlib
            cache_key = f"detection:{hashlib.md5(cache_key.encode()).hexdigest()}"
        
        return cache_key