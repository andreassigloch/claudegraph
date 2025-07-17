#!/usr/bin/env python3
"""
Analysis Result Validation Framework - Phase 3.2 Implementation

Validates analysis results against known ground truth data and provides
comprehensive accuracy metrics for dead code detection and flow analysis.

Key Features:
- Ground truth comparison against known correct results
- Statistical accuracy metrics (precision, recall, F1-score)
- False positive/negative detection and reporting
- Detailed validation reports with actionable insights
- Support for both dead code and flow relationship validation
"""

import json
import logging
from pathlib import Path
from typing import Dict, Set, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Statistical validation metrics"""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    

@dataclass
class DeadCodeValidationResult:
    """Results from dead code detection validation"""
    total_dead_detected: int = 0
    false_positives: List[str] = field(default_factory=list)
    false_negatives: List[str] = field(default_factory=list)
    true_positives: List[str] = field(default_factory=list)
    accuracy_metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    category_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class FlowValidationResult:
    """Results from flow relationship validation"""
    total_flows_detected: int = 0
    missing_flows: List[Tuple[str, str]] = field(default_factory=list)
    spurious_flows: List[Tuple[str, str]] = field(default_factory=list)
    correct_flows: List[Tuple[str, str]] = field(default_factory=list)
    accuracy_metrics: ValidationMetrics = field(default_factory=ValidationMetrics)


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    analysis_file: str
    ground_truth_file: Optional[str] = None
    validation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dead_code_validation: Optional[DeadCodeValidationResult] = None
    flow_validation: Optional[FlowValidationResult] = None
    overall_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    

class AnalysisResultValidator:
    """Validate analysis results against known ground truth."""
    
    def __init__(self, ground_truth_file: Optional[str] = None):
        self.ground_truth = self._load_ground_truth(ground_truth_file)
        self.ground_truth_file = ground_truth_file
        
    def _load_ground_truth(self, ground_truth_file: Optional[str]) -> Optional[Dict]:
        """Load ground truth data from file."""
        if not ground_truth_file:
            return None
            
        try:
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load ground truth file {ground_truth_file}: {e}")
            return None
    
    def validate_analysis_result(self, analysis_file: str) -> ValidationReport:
        """Validate complete analysis result."""
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis_result = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load analysis file {analysis_file}: {e}")
            return ValidationReport(
                analysis_file=analysis_file,
                recommendations=[f"Error loading analysis file: {e}"]
            )
        
        report = ValidationReport(
            analysis_file=analysis_file,
            ground_truth_file=self.ground_truth_file
        )
        
        # Validate dead code detection
        report.dead_code_validation = self.validate_dead_code_detection(analysis_result)
        
        # Validate flow relationships
        report.flow_validation = self.validate_flow_relationships(analysis_result)
        
        # Calculate overall score and recommendations
        self._generate_overall_assessment(report)
        
        return report
    
    def validate_dead_code_detection(self, analysis_result: Dict) -> DeadCodeValidationResult:
        """Validate dead code detection accuracy."""
        validation_result = DeadCodeValidationResult()
        
        # Extract dead functions from analysis
        dead_functions = self._extract_dead_functions(analysis_result)
        validation_result.total_dead_detected = len(dead_functions)
        
        if not self.ground_truth:
            logger.warning("No ground truth available - performing heuristic validation")
            return self._heuristic_dead_code_validation(analysis_result, dead_functions)
        
        # Compare against known ground truth
        known_dead = set(self.ground_truth.get('dead_functions', []))
        detected_dead = set(dead_functions)
        known_live = set(self.ground_truth.get('live_functions', []))
        
        # Calculate validation metrics
        validation_result.false_positives = list(detected_dead - known_dead)
        validation_result.false_negatives = list(known_dead - detected_dead)
        validation_result.true_positives = list(known_dead & detected_dead)
        
        # Calculate true negatives (live functions correctly identified as live)
        all_functions = known_dead | known_live
        detected_live = all_functions - detected_dead
        true_negatives = list(known_live & detected_live)
        
        # Calculate accuracy metrics
        tp = len(validation_result.true_positives)
        fp = len(validation_result.false_positives)
        fn = len(validation_result.false_negatives)
        tn = len(true_negatives)
        
        metrics = ValidationMetrics(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn
        )
        
        # Calculate derived metrics
        metrics.precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics.recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics.f1_score = (2 * metrics.precision * metrics.recall) / (metrics.precision + metrics.recall) if (metrics.precision + metrics.recall) > 0 else 0
        metrics.accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        
        validation_result.accuracy_metrics = metrics
        
        # Analyze confidence distribution and categories
        validation_result.confidence_distribution = self._analyze_confidence_distribution(analysis_result, dead_functions)
        validation_result.category_breakdown = self._analyze_dead_code_categories(analysis_result)
        
        logger.info(f"Dead code validation: Precision={metrics.precision:.3f}, Recall={metrics.recall:.3f}, F1={metrics.f1_score:.3f}")
        
        return validation_result
    
    def validate_flow_relationships(self, analysis_result: Dict) -> FlowValidationResult:
        """Validate flow relationship detection accuracy."""
        validation_result = FlowValidationResult()
        
        # Extract flow relationships from analysis
        detected_flows = self._extract_flow_relationships(analysis_result)
        validation_result.total_flows_detected = len(detected_flows)
        
        if not self.ground_truth:
            logger.warning("No ground truth available for flow validation")
            return validation_result
        
        # Compare against known ground truth flows
        known_flows = set()
        for flow in self.ground_truth.get('flow_relationships', []):
            if isinstance(flow, dict):
                source = flow.get('source', '')
                target = flow.get('target', '')
            else:
                # Assume it's a tuple/list
                source, target = flow[0], flow[1]
            known_flows.add((source, target))
        
        detected_flows_set = set(detected_flows)
        
        # Calculate validation metrics
        validation_result.missing_flows = list(known_flows - detected_flows_set)
        validation_result.spurious_flows = list(detected_flows_set - known_flows)
        validation_result.correct_flows = list(known_flows & detected_flows_set)
        
        # Calculate accuracy metrics
        tp = len(validation_result.correct_flows)
        fp = len(validation_result.spurious_flows)
        fn = len(validation_result.missing_flows)
        
        metrics = ValidationMetrics(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn
        )
        
        metrics.precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics.recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics.f1_score = (2 * metrics.precision * metrics.recall) / (metrics.precision + metrics.recall) if (metrics.precision + metrics.recall) > 0 else 0
        
        validation_result.accuracy_metrics = metrics
        
        logger.info(f"Flow validation: Precision={metrics.precision:.3f}, Recall={metrics.recall:.3f}, F1={metrics.f1_score:.3f}")
        
        return validation_result
    
    def _extract_dead_functions(self, analysis_result: Dict) -> List[str]:
        """Extract dead function names from analysis result."""
        dead_functions = []
        
        # Try to get from enhanced metadata first
        metadata = analysis_result.get('metadata', {})
        dead_code_analysis = metadata.get('dead_code_analysis', {})
        
        if dead_code_analysis:
            # Extract from detailed dead code analysis
            for category in ['duplicates', 'orphaned', 'unreachable']:
                category_data = dead_code_analysis.get('by_type', {}).get(category, [])
                for func_info in category_data:
                    if isinstance(func_info, dict):
                        func_name = f"{func_info.get('module', '')}.{func_info.get('name', '')}"
                        dead_functions.append(func_name)
                    else:
                        dead_functions.append(str(func_info))
        else:
            # Fallback: extract from node analysis
            nodes = analysis_result.get('nodes', [])
            relationships = analysis_result.get('relationships', [])
            
            # Find functions with no relationships
            func_nodes = [n for n in nodes if n.get('type') == 'FUNC']
            connected_funcs = set()
            
            for rel in relationships:
                if rel.get('type') == 'flow':
                    # Get function names from node lookups
                    source_node = next((n for n in nodes if n.get('uuid') == rel.get('source')), None)
                    target_node = next((n for n in nodes if n.get('uuid') == rel.get('target')), None)
                    
                    if source_node and source_node.get('type') == 'FUNC':
                        connected_funcs.add(source_node.get('Name', ''))
                    if target_node and target_node.get('type') == 'FUNC':
                        connected_funcs.add(target_node.get('Name', ''))
            
            # Functions without connections are potentially dead
            for func_node in func_nodes:
                func_name = func_node.get('Name', '')
                if func_name not in connected_funcs:
                    dead_functions.append(func_name)
        
        return dead_functions
    
    def _extract_flow_relationships(self, analysis_result: Dict) -> List[Tuple[str, str]]:
        """Extract flow relationships from analysis result."""
        flows = []
        
        nodes = analysis_result.get('nodes', [])
        relationships = analysis_result.get('relationships', [])
        
        # Build node lookup
        node_lookup = {node.get('uuid'): node.get('Name', '') for node in nodes}
        
        for rel in relationships:
            if rel.get('type') == 'flow':
                source_name = node_lookup.get(rel.get('source'), '')
                target_name = node_lookup.get(rel.get('target'), '')
                
                if source_name and target_name:
                    flows.append((source_name, target_name))
        
        return flows
    
    def _heuristic_dead_code_validation(self, analysis_result: Dict, dead_functions: List[str]) -> DeadCodeValidationResult:
        """Perform heuristic validation when no ground truth is available."""
        validation_result = DeadCodeValidationResult()
        validation_result.total_dead_detected = len(dead_functions)
        
        # Apply heuristic rules to identify likely false positives
        likely_false_positives = []
        likely_true_positives = []
        
        for func_name in dead_functions:
            # Functions that are likely false positives
            if any(pattern in func_name.lower() for pattern in ['main', 'cli', 'test_', 'setup', 'init']):
                likely_false_positives.append(func_name)
            # Functions that are likely legitimate dead code
            elif any(pattern in func_name.lower() for pattern in ['unused', 'deprecated', 'legacy', 'old_']):
                likely_true_positives.append(func_name)
            else:
                # Default to true positive if no clear indicators
                likely_true_positives.append(func_name)
        
        validation_result.false_positives = likely_false_positives
        validation_result.true_positives = likely_true_positives
        
        # Estimate metrics based on heuristics
        tp = len(likely_true_positives)
        fp = len(likely_false_positives)
        
        metrics = ValidationMetrics(
            true_positives=tp,
            false_positives=fp,
            precision=tp / (tp + fp) if (tp + fp) > 0 else 0
        )
        
        validation_result.accuracy_metrics = metrics
        
        return validation_result
    
    def _analyze_confidence_distribution(self, analysis_result: Dict, dead_functions: List[str]) -> Dict[str, int]:
        """Analyze confidence score distribution for dead functions."""
        distribution = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
        
        # Try to get confidence scores from hybrid analysis
        metadata = analysis_result.get('metadata', {})
        confidence_scores = metadata.get('hybrid_confidence_scores', {})
        
        for func_name in dead_functions:
            confidence = confidence_scores.get(func_name, 0.5)  # Default to medium
            
            if confidence >= 0.8:
                distribution['high'] += 1
            elif confidence >= 0.6:
                distribution['medium'] += 1
            elif confidence > 0:
                distribution['low'] += 1
            else:
                distribution['unknown'] += 1
        
        return distribution
    
    def _analyze_dead_code_categories(self, analysis_result: Dict) -> Dict[str, Dict[str, int]]:
        """Analyze dead code by category (duplicates, orphaned, unreachable)."""
        categories = {}
        
        metadata = analysis_result.get('metadata', {})
        dead_code_analysis = metadata.get('dead_code_analysis', {})
        
        if dead_code_analysis:
            for category in ['duplicates', 'orphaned', 'unreachable']:
                category_data = dead_code_analysis.get('by_type', {}).get(category, [])
                categories[category] = {
                    'count': len(category_data),
                    'examples': [item.get('name', '') for item in category_data[:3] if isinstance(item, dict)]
                }
        
        return categories
    
    def _generate_overall_assessment(self, report: ValidationReport):
        """Generate overall assessment and recommendations."""
        scores = []
        recommendations = []
        
        # Dead code validation assessment
        if report.dead_code_validation:
            dc_metrics = report.dead_code_validation.accuracy_metrics
            dead_code_score = (dc_metrics.precision + dc_metrics.recall + dc_metrics.f1_score) / 3
            scores.append(dead_code_score)
            
            if dc_metrics.precision < 0.8:
                recommendations.append(f"Dead code precision is low ({dc_metrics.precision:.2f}). Consider improving entry point detection.")
            
            if dc_metrics.recall < 0.8:
                recommendations.append(f"Dead code recall is low ({dc_metrics.recall:.2f}). May be missing actual dead code.")
            
            if len(report.dead_code_validation.false_positives) > 0:
                recommendations.append(f"Found {len(report.dead_code_validation.false_positives)} false positives in dead code detection.")
        
        # Flow validation assessment
        if report.flow_validation:
            flow_metrics = report.flow_validation.accuracy_metrics
            flow_score = (flow_metrics.precision + flow_metrics.recall + flow_metrics.f1_score) / 3
            scores.append(flow_score)
            
            if flow_metrics.precision < 0.7:
                recommendations.append(f"Flow detection precision is low ({flow_metrics.precision:.2f}). May have spurious flow relationships.")
            
            if flow_metrics.recall < 0.7:
                recommendations.append(f"Flow detection recall is low ({flow_metrics.recall:.2f}). May be missing actual flow relationships.")
        
        # Calculate overall score
        report.overall_score = sum(scores) / len(scores) if scores else 0.0
        
        # General recommendations
        if report.overall_score > 0.9:
            recommendations.append("Excellent analysis quality! Results are highly reliable.")
        elif report.overall_score > 0.7:
            recommendations.append("Good analysis quality with minor issues to address.")
        elif report.overall_score > 0.5:
            recommendations.append("Moderate analysis quality. Consider running hybrid analysis for better accuracy.")
        else:
            recommendations.append("Low analysis quality. Recommend investigating configuration and validation settings.")
        
        report.recommendations = recommendations
    
    def generate_validation_report(self, report: ValidationReport) -> str:
        """Generate human-readable validation report."""
        lines = []
        
        lines.append("=" * 60)
        lines.append("ANALYSIS VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Analysis File: {report.analysis_file}")
        lines.append(f"Ground Truth: {report.ground_truth_file or 'Not provided'}")
        lines.append(f"Validation Time: {report.validation_timestamp}")
        lines.append(f"Overall Score: {report.overall_score:.3f}")
        lines.append("")
        
        # Dead code validation
        if report.dead_code_validation:
            dc = report.dead_code_validation
            lines.append("📋 DEAD CODE VALIDATION:")
            lines.append(f"   Total detected: {dc.total_dead_detected}")
            lines.append(f"   True positives: {dc.accuracy_metrics.true_positives}")
            lines.append(f"   False positives: {dc.accuracy_metrics.false_positives}")
            lines.append(f"   False negatives: {dc.accuracy_metrics.false_negatives}")
            lines.append(f"   Precision: {dc.accuracy_metrics.precision:.3f}")
            lines.append(f"   Recall: {dc.accuracy_metrics.recall:.3f}")
            lines.append(f"   F1-Score: {dc.accuracy_metrics.f1_score:.3f}")
            
            if dc.false_positives:
                lines.append(f"   False Positives ({len(dc.false_positives)}):")
                for fp in dc.false_positives[:5]:
                    lines.append(f"     - {fp}")
                if len(dc.false_positives) > 5:
                    lines.append(f"     ... and {len(dc.false_positives) - 5} more")
            
            lines.append("")
        
        # Flow validation
        if report.flow_validation:
            fv = report.flow_validation
            lines.append("🔗 FLOW VALIDATION:")
            lines.append(f"   Total detected: {fv.total_flows_detected}")
            lines.append(f"   Correct flows: {fv.accuracy_metrics.true_positives}")
            lines.append(f"   Spurious flows: {fv.accuracy_metrics.false_positives}")
            lines.append(f"   Missing flows: {fv.accuracy_metrics.false_negatives}")
            lines.append(f"   Precision: {fv.accuracy_metrics.precision:.3f}")
            lines.append(f"   Recall: {fv.accuracy_metrics.recall:.3f}")
            lines.append(f"   F1-Score: {fv.accuracy_metrics.f1_score:.3f}")
            lines.append("")
        
        # Recommendations
        if report.recommendations:
            lines.append("💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"   {i}. {rec}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def create_sample_ground_truth(output_file: str, analysis_result_file: str):
    """Create a sample ground truth file based on analysis results."""
    try:
        with open(analysis_result_file, 'r') as f:
            analysis_result = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load analysis result: {e}")
        return
    
    # Extract functions and create sample ground truth
    nodes = analysis_result.get('nodes', [])
    func_nodes = [n for n in nodes if n.get('type') == 'FUNC']
    
    # Sample classification (this would need manual review in real usage)
    sample_ground_truth = {
        "description": "Sample ground truth for validation testing",
        "created_from": analysis_result_file,
        "created_at": datetime.now().isoformat(),
        "dead_functions": [
            # Add known dead functions here
        ],
        "live_functions": [
            # Add known live functions here
            node.get('Name', '') for node in func_nodes[:10]  # Sample first 10 as live
        ],
        "flow_relationships": [
            # Add known correct flow relationships here
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(sample_ground_truth, f, indent=2)
    
    logger.info(f"Sample ground truth created at {output_file}")