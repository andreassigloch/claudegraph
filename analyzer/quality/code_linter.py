#!/usr/bin/env python3
"""
Advanced Code Linter for Code Architecture Analyzer

Provides comprehensive code quality checking beyond basic linting,
including architecture-specific rules and best practices.
"""

import ast
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LintResult:
    """Result of linting analysis."""
    
    file_path: str
    issues: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def issue_count(self) -> int:
        """Get total number of issues."""
        return len(self.issues)
    
    @property
    def warning_count(self) -> int:
        """Get total number of warnings."""
        return len(self.warnings)
    
    @property
    def suggestion_count(self) -> int:
        """Get total number of suggestions."""
        return len(self.suggestions)
    
    def get_severity_counts(self) -> Dict[str, int]:
        """Get counts by severity level."""
        counts = {'error': 0, 'warning': 0, 'suggestion': 0}
        
        for issue in self.issues:
            severity = issue.get('severity', 'error')
            counts[severity] = counts.get(severity, 0) + 1
        
        for warning in self.warnings:
            counts['warning'] += 1
        
        for suggestion in self.suggestions:
            counts['suggestion'] += 1
        
        return counts


class QualityRule(ABC):
    """Abstract base class for quality rules."""
    
    @abstractmethod
    def check(self, tree: ast.AST, source_lines: List[str], file_path: str) -> List[Dict[str, Any]]:
        """Check rule against AST and source."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get rule name."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get rule description."""
        pass


class FunctionComplexityRule(QualityRule):
    """Rule to check function complexity."""
    
    def __init__(self, max_complexity: int = 10, max_length: int = 50):
        self.max_complexity = max_complexity
        self.max_length = max_length
    
    def check(self, tree: ast.AST, source_lines: List[str], file_path: str) -> List[Dict[str, Any]]:
        """Check function complexity."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Calculate cyclomatic complexity
                complexity = self._calculate_complexity(node)
                
                # Calculate function length
                func_start = node.lineno - 1
                func_end = max((n.lineno - 1 for n in ast.walk(node) if hasattr(n, 'lineno')), default=func_start)
                length = func_end - func_start + 1
                
                if complexity > self.max_complexity:
                    issues.append({
                        'type': 'complexity',
                        'severity': 'warning',
                        'line': node.lineno,
                        'column': node.col_offset,
                        'message': f"Function '{node.name}' has high complexity ({complexity})",
                        'rule': self.get_name(),
                        'details': {
                            'function_name': node.name,
                            'complexity': complexity,
                            'max_complexity': self.max_complexity
                        }
                    })
                
                if length > self.max_length:
                    issues.append({
                        'type': 'length',
                        'severity': 'suggestion',
                        'line': node.lineno,
                        'column': node.col_offset,
                        'message': f"Function '{node.name}' is long ({length} lines)",
                        'rule': self.get_name(),
                        'details': {
                            'function_name': node.name,
                            'length': length,
                            'max_length': self.max_length
                        }
                    })
        
        return issues
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # Add for each additional condition in boolean operations
                complexity += len(child.values) - 1
        
        return complexity
    
    def get_name(self) -> str:
        return "FunctionComplexity"
    
    def get_description(self) -> str:
        return f"Check function complexity (max: {self.max_complexity}) and length (max: {self.max_length})"


class NamingConventionRule(QualityRule):
    """Rule to check naming conventions."""
    
    def check(self, tree: ast.AST, source_lines: List[str], file_path: str) -> List[Dict[str, Any]]:
        """Check naming conventions."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not self._is_snake_case(node.name) and not node.name.startswith('__'):
                    issues.append({
                        'type': 'naming',
                        'severity': 'suggestion',
                        'line': node.lineno,
                        'column': node.col_offset,
                        'message': f"Function '{node.name}' should use snake_case",
                        'rule': self.get_name(),
                        'details': {'name': node.name, 'type': 'function'}
                    })
            
            elif isinstance(node, ast.ClassDef):
                if not self._is_pascal_case(node.name):
                    issues.append({
                        'type': 'naming',
                        'severity': 'suggestion',
                        'line': node.lineno,
                        'column': node.col_offset,
                        'message': f"Class '{node.name}' should use PascalCase",
                        'rule': self.get_name(),
                        'details': {'name': node.name, 'type': 'class'}
                    })
            
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                # Check variable names
                if (not self._is_snake_case(node.id) and 
                    not node.id.isupper() and  # Allow CONSTANTS
                    not node.id.startswith('_')):  # Allow private vars
                    issues.append({
                        'type': 'naming',
                        'severity': 'suggestion',
                        'line': node.lineno,
                        'column': node.col_offset,
                        'message': f"Variable '{node.id}' should use snake_case",
                        'rule': self.get_name(),
                        'details': {'name': node.id, 'type': 'variable'}
                    })
        
        return issues
    
    def _is_snake_case(self, name: str) -> bool:
        """Check if name follows snake_case convention."""
        return name.islower() and '_' in name or name.islower()
    
    def _is_pascal_case(self, name: str) -> bool:
        """Check if name follows PascalCase convention."""
        return name[0].isupper() and not '_' in name
    
    def get_name(self) -> str:
        return "NamingConvention"
    
    def get_description(self) -> str:
        return "Check naming conventions (snake_case for functions/variables, PascalCase for classes)"


class DocumentationRule(QualityRule):
    """Rule to check documentation quality."""
    
    def check(self, tree: ast.AST, source_lines: List[str], file_path: str) -> List[Dict[str, Any]]:
        """Check documentation quality."""
        issues = []
        
        # Check module docstring
        if not self._has_module_docstring(tree):
            issues.append({
                'type': 'documentation',
                'severity': 'warning',
                'line': 1,
                'column': 0,
                'message': "Module missing docstring",
                'rule': self.get_name(),
                'details': {'type': 'module'}
            })
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not self._has_docstring(node):
                    issues.append({
                        'type': 'documentation',
                        'severity': 'warning',
                        'line': node.lineno,
                        'column': node.col_offset,
                        'message': f"Function '{node.name}' missing docstring",
                        'rule': self.get_name(),
                        'details': {'name': node.name, 'type': 'function'}
                    })
                elif len(node.args.args) > 2 and not self._has_param_docs(node):
                    issues.append({
                        'type': 'documentation',
                        'severity': 'suggestion',
                        'line': node.lineno,
                        'column': node.col_offset,
                        'message': f"Function '{node.name}' should document parameters",
                        'rule': self.get_name(),
                        'details': {'name': node.name, 'type': 'function_params'}
                    })
            
            elif isinstance(node, ast.ClassDef):
                if not self._has_docstring(node):
                    issues.append({
                        'type': 'documentation',
                        'severity': 'warning',
                        'line': node.lineno,
                        'column': node.col_offset,
                        'message': f"Class '{node.name}' missing docstring",
                        'rule': self.get_name(),
                        'details': {'name': node.name, 'type': 'class'}
                    })
        
        return issues
    
    def _has_module_docstring(self, tree: ast.AST) -> bool:
        """Check if module has docstring."""
        return (tree.body and 
                isinstance(tree.body[0], ast.Expr) and 
                isinstance(tree.body[0].value, (ast.Str, ast.Constant)))
    
    def _has_docstring(self, node: ast.AST) -> bool:
        """Check if node has docstring."""
        return (hasattr(node, 'body') and node.body and
                isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, (ast.Str, ast.Constant)))
    
    def _has_param_docs(self, node: ast.FunctionDef) -> bool:
        """Check if function documents parameters."""
        if not self._has_docstring(node):
            return False
        
        docstring_node = node.body[0].value
        if isinstance(docstring_node, ast.Str):
            docstring = docstring_node.s
        elif isinstance(docstring_node, ast.Constant):
            docstring = str(docstring_node.value)
        else:
            return False
        
        # Simple check for parameter documentation
        return 'Args:' in docstring or 'Parameters:' in docstring or 'param ' in docstring.lower()
    
    def get_name(self) -> str:
        return "Documentation"
    
    def get_description(self) -> str:
        return "Check documentation completeness and quality"


class ArchitectureRule(QualityRule):
    """Rule to check architecture-specific patterns."""
    
    def check(self, tree: ast.AST, source_lines: List[str], file_path: str) -> List[Dict[str, Any]]:
        """Check architecture-specific rules."""
        issues = []
        
        # Check for god classes (too many methods)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                
                if method_count > 20:
                    issues.append({
                        'type': 'architecture',
                        'severity': 'warning',
                        'line': node.lineno,
                        'column': node.col_offset,
                        'message': f"Class '{node.name}' has too many methods ({method_count})",
                        'rule': self.get_name(),
                        'details': {'name': node.name, 'method_count': method_count}
                    })
        
        # Check for excessive imports
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        if len(imports) > 20:
            issues.append({
                'type': 'architecture',
                'severity': 'suggestion',
                'line': imports[20].lineno if len(imports) > 20 else 1,
                'column': 0,
                'message': f"File has many imports ({len(imports)}), consider refactoring",
                'rule': self.get_name(),
                'details': {'import_count': len(imports)}
            })
        
        # Check for deep nesting
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                max_depth = self._calculate_nesting_depth(node)
                if max_depth > 4:
                    issues.append({
                        'type': 'architecture',
                        'severity': 'suggestion',
                        'line': node.lineno,
                        'column': node.col_offset,
                        'message': f"Function '{node.name}' has deep nesting (depth {max_depth})",
                        'rule': self.get_name(),
                        'details': {'name': node.name, 'nesting_depth': max_depth}
                    })
        
        return issues
    
    def _calculate_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                child_depth = self._calculate_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calculate_nesting_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def get_name(self) -> str:
        return "Architecture"
    
    def get_description(self) -> str:
        return "Check architecture-specific patterns and anti-patterns"


class CodeLinter:
    """
    Advanced code linter with architecture-specific rules.
    """
    
    def __init__(self, rules: Optional[List[QualityRule]] = None):
        """
        Initialize code linter.
        
        Args:
            rules: List of quality rules to apply
        """
        self.rules = rules or self._get_default_rules()
        self.results: Dict[str, LintResult] = {}
        
        logger.info(f"Code linter initialized with {len(self.rules)} rules")
    
    def _get_default_rules(self) -> List[QualityRule]:
        """Get default set of quality rules."""
        return [
            FunctionComplexityRule(),
            NamingConventionRule(),
            DocumentationRule(),
            ArchitectureRule()
        ]
    
    def lint_file(self, file_path: str) -> LintResult:
        """
        Lint a single file.
        
        Args:
            file_path: Path to file to lint
            
        Returns:
            LintResult with findings
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                source_lines = content.splitlines()
            
            tree = ast.parse(content)
            
            result = LintResult(file_path=file_path)
            
            # Apply all rules
            for rule in self.rules:
                try:
                    rule_issues = rule.check(tree, source_lines, file_path)
                    
                    # Categorize issues
                    for issue in rule_issues:
                        severity = issue.get('severity', 'error')
                        if severity == 'error':
                            result.issues.append(issue)
                        elif severity == 'warning':
                            result.warnings.append(issue)
                        else:
                            result.suggestions.append(issue)
                
                except Exception as e:
                    logger.warning(f"Rule {rule.get_name()} failed on {file_path}: {e}")
            
            # Calculate metrics
            result.metrics = self._calculate_file_metrics(tree, source_lines)
            
            self.results[file_path] = result
            return result
            
        except Exception as e:
            logger.error(f"Failed to lint {file_path}: {e}")
            result = LintResult(file_path=file_path)
            result.issues.append({
                'type': 'system',
                'severity': 'error',
                'line': 1,
                'column': 0,
                'message': f"Failed to parse file: {e}",
                'rule': 'System',
                'details': {'error': str(e)}
            })
            return result
    
    def lint_directory(self, directory: str, pattern: str = "*.py") -> Dict[str, LintResult]:
        """
        Lint all files in a directory.
        
        Args:
            directory: Directory to lint
            pattern: File pattern to match
            
        Returns:
            Dictionary mapping file paths to LintResults
        """
        results = {}
        
        for file_path in Path(directory).rglob(pattern):
            if '__pycache__' in str(file_path):
                continue
                
            result = self.lint_file(str(file_path))
            results[str(file_path)] = result
        
        self.results.update(results)
        return results
    
    def _calculate_file_metrics(self, tree: ast.AST, source_lines: List[str]) -> Dict[str, Any]:
        """Calculate file-level metrics."""
        metrics = {
            'lines_of_code': len(source_lines),
            'blank_lines': len([line for line in source_lines if not line.strip()]),
            'comment_lines': len([line for line in source_lines if line.strip().startswith('#')]),
            'function_count': 0,
            'class_count': 0,
            'import_count': 0,
            'docstring_coverage': 0.0
        }
        
        functions_with_docstrings = 0
        total_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics['function_count'] += 1
                total_functions += 1
                
                if self._has_docstring(node):
                    functions_with_docstrings += 1
            
            elif isinstance(node, ast.ClassDef):
                metrics['class_count'] += 1
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics['import_count'] += 1
        
        if total_functions > 0:
            metrics['docstring_coverage'] = (functions_with_docstrings / total_functions) * 100
        
        return metrics
    
    def _has_docstring(self, node: ast.AST) -> bool:
        """Check if node has docstring."""
        return (hasattr(node, 'body') and node.body and
                isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, (ast.Str, ast.Constant)))
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get summary report of all linting results."""
        if not self.results:
            return {}
        
        summary = {
            'files_analyzed': len(self.results),
            'total_issues': 0,
            'total_warnings': 0,
            'total_suggestions': 0,
            'issues_by_type': defaultdict(int),
            'issues_by_rule': defaultdict(int),
            'worst_files': [],
            'metrics_summary': {
                'total_lines': 0,
                'total_functions': 0,
                'total_classes': 0,
                'average_docstring_coverage': 0.0
            }
        }
        
        docstring_coverages = []
        
        for file_path, result in self.results.items():
            summary['total_issues'] += result.issue_count
            summary['total_warnings'] += result.warning_count
            summary['total_suggestions'] += result.suggestion_count
            
            # Count by type and rule
            for issue in result.issues + result.warnings + result.suggestions:
                summary['issues_by_type'][issue.get('type', 'unknown')] += 1
                summary['issues_by_rule'][issue.get('rule', 'unknown')] += 1
            
            # Aggregate metrics
            metrics = result.metrics
            summary['metrics_summary']['total_lines'] += metrics.get('lines_of_code', 0)
            summary['metrics_summary']['total_functions'] += metrics.get('function_count', 0)
            summary['metrics_summary']['total_classes'] += metrics.get('class_count', 0)
            
            if metrics.get('docstring_coverage', 0) > 0:
                docstring_coverages.append(metrics['docstring_coverage'])
        
        # Calculate averages
        if docstring_coverages:
            summary['metrics_summary']['average_docstring_coverage'] = sum(docstring_coverages) / len(docstring_coverages)
        
        # Find worst files
        file_scores = []
        for file_path, result in self.results.items():
            score = (result.issue_count * 3) + (result.warning_count * 2) + result.suggestion_count
            file_scores.append((score, file_path, result))
        
        file_scores.sort(reverse=True)
        summary['worst_files'] = [
            {
                'file_path': file_path,
                'score': score,
                'issues': result.issue_count,
                'warnings': result.warning_count,
                'suggestions': result.suggestion_count
            }
            for score, file_path, result in file_scores[:10]
        ]
        
        return summary
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate human-readable linting report.
        
        Args:
            output_file: Optional file to write report to
            
        Returns:
            Report text
        """
        summary = self.get_summary_report()
        
        if not summary:
            return "No files analyzed."
        
        report_lines = []
        report_lines.append("Code Quality Analysis Report")
        report_lines.append("=" * 40)
        report_lines.append(f"Files analyzed: {summary['files_analyzed']}")
        report_lines.append(f"Total issues: {summary['total_issues']}")
        report_lines.append(f"Total warnings: {summary['total_warnings']}")
        report_lines.append(f"Total suggestions: {summary['total_suggestions']}")
        report_lines.append("")
        
        # Issues by type
        report_lines.append("Issues by Type:")
        for issue_type, count in sorted(summary['issues_by_type'].items()):
            report_lines.append(f"  {issue_type}: {count}")
        report_lines.append("")
        
        # Issues by rule
        report_lines.append("Issues by Rule:")
        for rule, count in sorted(summary['issues_by_rule'].items()):
            report_lines.append(f"  {rule}: {count}")
        report_lines.append("")
        
        # Metrics summary
        metrics = summary['metrics_summary']
        report_lines.append("Code Metrics:")
        report_lines.append(f"  Total lines of code: {metrics['total_lines']}")
        report_lines.append(f"  Total functions: {metrics['total_functions']}")
        report_lines.append(f"  Total classes: {metrics['total_classes']}")
        report_lines.append(f"  Average docstring coverage: {metrics['average_docstring_coverage']:.1f}%")
        report_lines.append("")
        
        # Worst files
        if summary['worst_files']:
            report_lines.append("Files needing attention:")
            for file_info in summary['worst_files'][:5]:
                report_lines.append(f"  {file_info['file_path']} (score: {file_info['score']})")
                report_lines.append(f"    Issues: {file_info['issues']}, "
                                  f"Warnings: {file_info['warnings']}, "
                                  f"Suggestions: {file_info['suggestions']}")
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                logger.info(f"Report written to {output_file}")
            except Exception as e:
                logger.error(f"Failed to write report to {output_file}: {e}")
        
        return report_text
    
    def clear_results(self):
        """Clear all linting results."""
        self.results.clear()
        logger.info("Linting results cleared")


# Convenience functions
def lint_file(file_path: str, rules: Optional[List[QualityRule]] = None) -> LintResult:
    """Lint a single file with default or custom rules."""
    linter = CodeLinter(rules)
    return linter.lint_file(file_path)


def lint_directory(directory: str, 
                  rules: Optional[List[QualityRule]] = None,
                  pattern: str = "*.py") -> Dict[str, LintResult]:
    """Lint all files in a directory."""
    linter = CodeLinter(rules)
    return linter.lint_directory(directory, pattern)