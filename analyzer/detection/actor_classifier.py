#!/usr/bin/env python3
"""
Actor Classifier for Code Architecture Analyzer

Consolidates detection results from multiple specialized detectors and provides
unified actor classification with confidence scoring and ambiguity resolution.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum

from .pattern_matcher import ActorType, DetectionMatch, ActorDetectionResult, ConfidenceLevel

logger = logging.getLogger(__name__)


class ClassificationStrategy(Enum):
    """Strategies for consolidating multiple detections."""
    HIGHEST_CONFIDENCE = "highest_confidence"
    WEIGHTED_AVERAGE = "weighted_average"
    EVIDENCE_COUNT = "evidence_count"
    CONSENSUS = "consensus"


@dataclass
class ConsolidatedActor:
    """Represents a consolidated actor from multiple detection sources."""
    actor_type: ActorType
    confidence: float
    primary_library: str
    evidence_sources: List[str] = field(default_factory=list)
    detection_methods: List[str] = field(default_factory=list)
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    ambiguity_score: float = 0.0
    requires_review: bool = False
    module_usage: List[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Complete result of actor classification for a module."""
    module_name: str
    consolidated_actors: List[ConsolidatedActor] = field(default_factory=list)
    high_confidence_actors: List[ConsolidatedActor] = field(default_factory=list)
    ambiguous_actors: List[ConsolidatedActor] = field(default_factory=list)
    conflicting_detections: List[Tuple[DetectionMatch, DetectionMatch]] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)


class ActorClassifier:
    """
    Consolidates and classifies actors from multiple detection sources.
    
    Takes results from specialized detectors (HTTP, Database, Filesystem, Endpoint)
    and provides unified actor classification with confidence scoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize actor classifier with configuration."""
        self.config = config or {}
        
        # Classification settings
        classification_config = self.config.get('classification', {})
        self.confidence_threshold = classification_config.get('confidence_threshold', 0.8)
        self.ambiguity_threshold = classification_config.get('ambiguity_threshold', 0.3)
        self.min_evidence_count = classification_config.get('min_evidence_count', 1)
        self.strategy = ClassificationStrategy(
            classification_config.get('strategy', 'weighted_average')
        )
        
        # Consolidation weights for different detection methods
        self.method_weights = {
            'import_analysis': 0.7,
            'call_analysis': 0.9,
            'decorator_analysis': 0.95,
            'string_analysis': 0.5,
            'class_analysis': 0.8
        }
        
        # Library priority for conflict resolution
        self.library_priority = {
            'requests': 0.9,
            'httpx': 0.85,
            'aiohttp': 0.8,
            'urllib': 0.6,
            'sqlalchemy': 0.9,
            'django': 0.85,
            'psycopg2': 0.8,
            'pymongo': 0.8,
            'sqlite3': 0.7,
            'pathlib': 0.9,
            'os': 0.7,
            'shutil': 0.8,
            'flask': 0.9,
            'fastapi': 0.95,
            'django_rest': 0.9
        }
        
        logger.debug("Actor classifier initialized")
    
    def classify_actors(self, detection_results: List[ActorDetectionResult]) -> List[ClassificationResult]:
        """
        Classify actors from multiple detection results.
        
        Args:
            detection_results: List of detection results from various detectors
            
        Returns:
            List of classification results with consolidated actors
        """
        classification_results = []
        
        try:
            # Group detection results by module
            module_detections = self._group_by_module(detection_results)
            
            for module_name, detections in module_detections.items():
                # Classify actors for this module
                result = self._classify_module_actors(module_name, detections)
                classification_results.append(result)
            
            logger.info(f"Classified actors for {len(classification_results)} modules")
            
        except Exception as e:
            logger.error(f"Actor classification failed: {e}")
        
        return classification_results
    
    def _group_by_module(self, detection_results: List[ActorDetectionResult]) -> Dict[str, List[DetectionMatch]]:
        """Group detection results by module name."""
        module_detections = defaultdict(list)
        
        for result in detection_results:
            module_detections[result.module_name].extend(result.detected_actors)
        
        return dict(module_detections)
    
    def _classify_module_actors(self, module_name: str, detections: List[DetectionMatch]) -> ClassificationResult:
        """Classify actors for a single module."""
        result = ClassificationResult(module_name=module_name)
        
        try:
            # Group detections by actor type
            actor_groups = self._group_by_actor_type(detections)
            
            # Consolidate each actor type
            for actor_type, matches in actor_groups.items():
                consolidated = self._consolidate_actor_detections(actor_type, matches)
                if consolidated:
                    result.consolidated_actors.append(consolidated)
            
            # Classify by confidence
            self._classify_by_confidence(result)
            
            # Detect conflicts
            result.conflicting_detections = self._detect_conflicts(detections)
            
            # Generate statistics
            result.statistics = self._generate_classification_stats(result, detections)
            
            logger.debug(f"Classified {len(result.consolidated_actors)} actors for {module_name}")
            
        except Exception as e:
            logger.error(f"Module classification failed for {module_name}: {e}")
            result.statistics['error'] = str(e)
        
        return result
    
    def _group_by_actor_type(self, detections: List[DetectionMatch]) -> Dict[ActorType, List[DetectionMatch]]:
        """Group detections by actor type."""
        groups = defaultdict(list)
        
        for detection in detections:
            groups[detection.actor_type].append(detection)
        
        return dict(groups)
    
    def _consolidate_actor_detections(self, actor_type: ActorType, matches: List[DetectionMatch]) -> Optional[ConsolidatedActor]:
        """Consolidate multiple detections of the same actor type."""
        if not matches:
            return None
        
        try:
            # Extract evidence from all matches
            evidence_sources = []
            detection_methods = []
            libraries = []
            confidences = []
            modules = []
            
            supporting_evidence = {
                'import_count': 0,
                'call_count': 0,
                'decorator_count': 0,
                'string_count': 0,
                'total_evidence': len(matches)
            }
            
            for match in matches:
                # Collect evidence sources
                if match.evidence:
                    evidence_sources.extend(match.evidence.keys())
                
                # Collect detection methods
                method = match.context.get('detection_method', 'unknown')
                detection_methods.append(method)
                
                # Count method types
                if 'import' in method:
                    supporting_evidence['import_count'] += 1
                elif 'call' in method:
                    supporting_evidence['call_count'] += 1
                elif 'decorator' in method:
                    supporting_evidence['decorator_count'] += 1
                elif 'string' in method:
                    supporting_evidence['string_count'] += 1
                
                # Collect library information
                library = match.context.get('library_type', 'unknown')
                if library != 'unknown':
                    libraries.append(library)
                
                # Collect confidences
                confidences.append(match.confidence)
                
                # Collect modules
                if match.module_name:
                    modules.append(match.module_name)
            
            # Determine primary library
            primary_library = self._determine_primary_library(libraries)
            
            # Calculate consolidated confidence
            consolidated_confidence = self._calculate_consolidated_confidence(matches, confidences)
            
            # Calculate ambiguity score
            ambiguity_score = self._calculate_ambiguity_score(matches, libraries, confidences)
            
            # Determine if review is required
            requires_review = (
                ambiguity_score > self.ambiguity_threshold or
                consolidated_confidence < self.confidence_threshold or
                len(set(libraries)) > 2  # Multiple conflicting libraries
            )
            
            consolidated = ConsolidatedActor(
                actor_type=actor_type,
                confidence=consolidated_confidence,
                primary_library=primary_library,
                evidence_sources=list(set(evidence_sources)),
                detection_methods=list(set(detection_methods)),
                supporting_evidence=supporting_evidence,
                ambiguity_score=ambiguity_score,
                requires_review=requires_review,
                module_usage=list(set(modules))
            )
            
            return consolidated
            
        except Exception as e:
            logger.error(f"Error consolidating {actor_type} detections: {e}")
            return None
    
    def _determine_primary_library(self, libraries: List[str]) -> str:
        """Determine the primary library from multiple detections."""
        if not libraries:
            return 'unknown'
        
        # Count library occurrences
        library_counts = Counter(libraries)
        
        # If there's a clear winner, use it
        most_common = library_counts.most_common(1)[0]
        if most_common[1] > len(libraries) / 2:
            return most_common[0]
        
        # Otherwise, use priority-based selection
        prioritized_libraries = sorted(
            set(libraries),
            key=lambda lib: self.library_priority.get(lib, 0.5),
            reverse=True
        )
        
        return prioritized_libraries[0] if prioritized_libraries else 'unknown'
    
    def _calculate_consolidated_confidence(self, matches: List[DetectionMatch], confidences: List[float]) -> float:
        """Calculate consolidated confidence from multiple matches."""
        if not confidences:
            return 0.0
        
        if self.strategy == ClassificationStrategy.HIGHEST_CONFIDENCE:
            return max(confidences)
        
        elif self.strategy == ClassificationStrategy.WEIGHTED_AVERAGE:
            # Weight by detection method
            weighted_sum = 0.0
            total_weight = 0.0
            
            for match, confidence in zip(matches, confidences):
                method = match.context.get('detection_method', 'unknown')
                weight = self._get_method_weight(method)
                weighted_sum += confidence * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        elif self.strategy == ClassificationStrategy.EVIDENCE_COUNT:
            # Boost confidence based on evidence count
            base_confidence = max(confidences)
            evidence_boost = min(0.1 * (len(matches) - 1), 0.3)
            return min(base_confidence + evidence_boost, 1.0)
        
        elif self.strategy == ClassificationStrategy.CONSENSUS:
            # Use consensus of high-confidence matches
            high_conf_matches = [c for c in confidences if c >= 0.7]
            if high_conf_matches:
                return sum(high_conf_matches) / len(high_conf_matches)
            else:
                return max(confidences)
        
        else:
            # Default to simple average
            return sum(confidences) / len(confidences)
    
    def _get_method_weight(self, method: str) -> float:
        """Get weight for detection method."""
        for method_key, weight in self.method_weights.items():
            if method_key in method:
                return weight
        return 0.5  # Default weight
    
    def _calculate_ambiguity_score(self, matches: List[DetectionMatch], 
                                  libraries: List[str], confidences: List[float]) -> float:
        """Calculate ambiguity score for the detections."""
        if not matches:
            return 1.0  # Maximum ambiguity
        
        ambiguity_factors = []
        
        # Library diversity factor
        unique_libraries = set(lib for lib in libraries if lib != 'unknown')
        if len(unique_libraries) > 1:
            library_diversity = len(unique_libraries) / len(libraries)
            ambiguity_factors.append(library_diversity * 0.4)
        
        # Confidence variance factor
        if len(confidences) > 1:
            confidence_variance = max(confidences) - min(confidences)
            ambiguity_factors.append(confidence_variance * 0.3)
        
        # Low confidence factor
        low_conf_ratio = sum(1 for c in confidences if c < 0.7) / len(confidences)
        ambiguity_factors.append(low_conf_ratio * 0.3)
        
        return sum(ambiguity_factors)
    
    def _classify_by_confidence(self, result: ClassificationResult) -> None:
        """Classify consolidated actors by confidence level."""
        for actor in result.consolidated_actors:
            if actor.confidence >= self.confidence_threshold and not actor.requires_review:
                result.high_confidence_actors.append(actor)
            else:
                result.ambiguous_actors.append(actor)
    
    def _detect_conflicts(self, detections: List[DetectionMatch]) -> List[Tuple[DetectionMatch, DetectionMatch]]:
        """Detect conflicting detections."""
        conflicts = []
        
        # Group by function/location for conflict detection
        location_groups = defaultdict(list)
        for detection in detections:
            location_key = (detection.function_name, detection.line_numbers[0] if detection.line_numbers else 0)
            location_groups[location_key].append(detection)
        
        # Find conflicts within each location
        for location, matches in location_groups.items():
            if len(matches) > 1:
                # Check for conflicting actor types at same location
                actor_types = set(match.actor_type for match in matches)
                if len(actor_types) > 1:
                    # Add all pairwise conflicts
                    for i, match1 in enumerate(matches):
                        for match2 in matches[i+1:]:
                            if match1.actor_type != match2.actor_type:
                                conflicts.append((match1, match2))
        
        return conflicts
    
    def _generate_classification_stats(self, result: ClassificationResult, 
                                     original_detections: List[DetectionMatch]) -> Dict[str, Any]:
        """Generate statistics for classification result."""
        stats = {
            'total_detections': len(original_detections),
            'consolidated_actors': len(result.consolidated_actors),
            'high_confidence_actors': len(result.high_confidence_actors),
            'ambiguous_actors': len(result.ambiguous_actors),
            'conflicts_detected': len(result.conflicting_detections),
            'consolidation_ratio': len(result.consolidated_actors) / max(1, len(original_detections)),
            'actor_type_distribution': {},
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'detection_method_stats': {},
            'library_distribution': {},
            'evidence_strength': {}
        }
        
        # Actor type distribution
        for actor in result.consolidated_actors:
            actor_type = actor.actor_type.value if hasattr(actor.actor_type, 'value') else str(actor.actor_type)
            stats['actor_type_distribution'][actor_type] = stats['actor_type_distribution'].get(actor_type, 0) + 1
        
        # Confidence distribution
        for actor in result.consolidated_actors:
            if actor.confidence >= 0.8:
                stats['confidence_distribution']['high'] += 1
            elif actor.confidence >= 0.6:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
        
        # Detection method statistics
        all_methods = []
        for actor in result.consolidated_actors:
            all_methods.extend(actor.detection_methods)
        
        method_counts = Counter(all_methods)
        stats['detection_method_stats'] = dict(method_counts)
        
        # Library distribution
        library_counts = Counter(actor.primary_library for actor in result.consolidated_actors)
        stats['library_distribution'] = dict(library_counts)
        
        # Evidence strength analysis
        for actor in result.consolidated_actors:
            evidence = actor.supporting_evidence
            strength = (
                evidence.get('import_count', 0) * 0.7 +
                evidence.get('call_count', 0) * 0.9 +
                evidence.get('decorator_count', 0) * 0.95 +
                evidence.get('string_count', 0) * 0.5
            )
            stats['evidence_strength'][actor.primary_library] = strength
        
        return stats
    
    def get_review_candidates(self, classification_results: List[ClassificationResult]) -> List[ConsolidatedActor]:
        """Get actors that require manual review."""
        review_candidates = []
        
        for result in classification_results:
            review_candidates.extend(result.ambiguous_actors)
            
            # Also include high-confidence actors with high ambiguity
            for actor in result.high_confidence_actors:
                if actor.ambiguity_score > self.ambiguity_threshold:
                    review_candidates.append(actor)
        
        # Sort by ambiguity score (highest first)
        review_candidates.sort(key=lambda a: a.ambiguity_score, reverse=True)
        
        return review_candidates
    
    def get_global_statistics(self, classification_results: List[ClassificationResult]) -> Dict[str, Any]:
        """Get global statistics across all classification results."""
        global_stats = {
            'total_modules': len(classification_results),
            'total_actors': 0,
            'high_confidence_actors': 0,
            'ambiguous_actors': 0,
            'total_conflicts': 0,
            'actor_type_totals': {},
            'library_totals': {},
            'avg_confidence': 0.0,
            'consolidation_efficiency': 0.0
        }
        
        total_confidence = 0.0
        total_original_detections = 0
        total_consolidated = 0
        
        for result in classification_results:
            # Aggregate counts
            global_stats['total_actors'] += len(result.consolidated_actors)
            global_stats['high_confidence_actors'] += len(result.high_confidence_actors)
            global_stats['ambiguous_actors'] += len(result.ambiguous_actors)
            global_stats['total_conflicts'] += len(result.conflicting_detections)
            
            # Sum confidences for average calculation
            for actor in result.consolidated_actors:
                total_confidence += actor.confidence
            
            # Consolidation efficiency
            total_original_detections += result.statistics.get('total_detections', 0)
            total_consolidated += len(result.consolidated_actors)
            
            # Aggregate distributions
            for actor_type, count in result.statistics.get('actor_type_distribution', {}).items():
                global_stats['actor_type_totals'][actor_type] = global_stats['actor_type_totals'].get(actor_type, 0) + count
            
            for library, count in result.statistics.get('library_distribution', {}).items():
                global_stats['library_totals'][library] = global_stats['library_totals'].get(library, 0) + count
        
        # Calculate averages
        if global_stats['total_actors'] > 0:
            global_stats['avg_confidence'] = total_confidence / global_stats['total_actors']
        
        if total_original_detections > 0:
            global_stats['consolidation_efficiency'] = total_consolidated / total_original_detections
        
        return global_stats