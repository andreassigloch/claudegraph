#!/usr/bin/env python3
"""
Code Quality Enhancement for Code Architecture Analyzer

This module provides tools and utilities for improving code quality,
including linting, formatting, testing, and best practice enforcement.

Components included:
- CodeLinter: Advanced linting and quality checking
- DocumentationChecker: Documentation completeness verification
- TestRunner: Enhanced testing infrastructure
- QualityMetrics: Code quality measurement and tracking
- CodeFormatter: Automatic code formatting and style enforcement
"""

from .code_linter import CodeLinter, LintResult, QualityRule

__all__ = [
    'CodeLinter',
    'LintResult', 
    'QualityRule'
]