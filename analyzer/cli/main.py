#!/usr/bin/env python3
"""
Simplified CLI for Code Architecture Analyzer

Usage: analyze <source> <target> [--llm <provider>]
"""

import sys
import os
import json
import logging
import click
from pathlib import Path
from typing import Optional
from datetime import datetime

# Import analyzer components
try:
    from analyzer.core.flow_based_analyzer import FlowBasedAnalyzer
    from analyzer.core.logical_analyzer import LogicalAnalyzer
    from analyzer.core.hybrid_analyzer import HybridAnalyzer
    from analyzer.validation.result_validator import AnalysisResultValidator
    from .config import load_unified_config
    from analyzer.utils.analysis_comparator import compare_analysis_files
except ImportError as e:
    print(f"Error importing analyzer components: {e}")
    sys.exit(1)


def setup_logging(verbose: bool = False):
    """Setup basic logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


@click.command()
@click.argument('source', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('target', type=click.Path())
@click.option('--llm', type=click.Choice(['none', 'local', 'openai', 'anthropic']), 
              default='none', help='LLM provider for enhanced analysis (default: none)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', type=click.Path(exists=True), 
              help='Custom config file (default: config/llm_config.yaml)')
def analyze(source, target, llm, verbose, config):
    """
    Analyze Python project architecture.
    
    SOURCE: Path to Python project directory
    TARGET: Output JSON file path
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        if config:
            analyzer_config = load_unified_config(config, llm_provider=llm)
        else:
            # Use default config path
            default_config = Path(__file__).parent.parent.parent / 'config' / 'llm_config.yaml'
            analyzer_config = load_unified_config(str(default_config), llm_provider=llm)
        
        click.echo(f"🔍 Analyzing: {source}")
        click.echo(f"📁 Output: {target}")
        click.echo(f"🤖 LLM Provider: {llm}")
        
        # Initialize analyzer
        analyzer = FlowBasedAnalyzer(analyzer_config)
        
        # Validate project
        click.echo("🔍 Validating project...")
        is_valid, errors = analyzer.validate_project(source)
        
        if not is_valid:
            click.echo("❌ Project validation failed:")
            for error in errors:
                click.echo(f"  • {error}")
            sys.exit(1)
        
        click.echo("✅ Project validation passed")
        
        # Perform analysis
        click.echo("🚀 Starting analysis...")
        
        with click.progressbar(length=100, label='Analyzing') as bar:
            result = analyzer.analyze(source)
            bar.update(100)
        
        # Check results
        if not result.is_successful():
            click.echo("❌ Analysis failed:")
            for error in result.errors[:3]:  # Show first 3 errors
                click.echo(f"  • {error}")
            if len(result.errors) > 3:
                click.echo(f"  ... and {len(result.errors) - 3} more errors")
            sys.exit(1)
        
        # Generate output
        click.echo(f"💾 Generating output...")
        
        # Convert analysis result to JSON format
        output_data = analyzer.format_for_export(result)
        
        # Ensure target directory exists
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write output file
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Show results
        click.echo("\n" + "="*50)
        click.echo("✅ ANALYSIS COMPLETED")
        click.echo("="*50)
        click.echo(f"📄 Files analyzed: {result.stats.files_parsed}")
        click.echo(f"🔧 Functions found: {result.stats.functions_found}")
        click.echo(f"📦 Classes found: {result.stats.classes_found}")
        click.echo(f"🎭 Trigger actors: {result.stats.trigger_actors}")
        click.echo(f"📡 Receiver actors: {result.stats.receiver_actors}")
        click.echo(f"🔗 Flow chains: {result.stats.flow_chains}")
        click.echo(f"🎯 Total nodes: {result.stats.sys_nodes + result.stats.mod_nodes + result.stats.func_nodes + result.stats.actor_nodes}")
        click.echo(f"⏱️  Analysis time: {result.stats.analysis_time_seconds:.2f}s")
        
        if result.stats.llm_calls_made > 0:
            click.echo(f"🤖 LLM calls made: {result.stats.llm_calls_made}")
            click.echo(f"✨ Actors enhanced: {result.stats.actors_enhanced}")
        
        # Show file info
        file_size = os.path.getsize(target)
        if file_size > 1024 * 1024:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size} bytes"
        
        click.echo(f"📊 Output size: {size_str}")
        click.echo(f"📁 Saved to: {target}")
        click.echo("="*50)
        
    except KeyboardInterrupt:
        click.echo("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Analysis failed")
        click.echo(f"❌ Analysis failed: {e}")
        sys.exit(1)


@click.command()
@click.argument('source', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--config', type=click.Path(exists=True), 
              help='Custom config file (default: config/llm_config.yaml)')
def validate(source, config):
    """Validate a Python project for analysis."""
    setup_logging()
    
    try:
        # Load minimal config for validation
        if config:
            analyzer_config = load_unified_config(config, llm_provider='none')
        else:
            default_config = Path(__file__).parent.parent.parent / 'config' / 'llm_config.yaml'
            analyzer_config = load_unified_config(str(default_config), llm_provider='none')
        
        analyzer = FlowBasedAnalyzer(analyzer_config)
        is_valid, errors = analyzer.validate_project(source)
        
        if is_valid:
            click.echo(f"✅ Project '{source}' is valid for analysis")
        else:
            click.echo(f"❌ Project '{source}' validation failed:")
            for error in errors:
                click.echo(f"  • {error}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}")
        sys.exit(1)


@click.command()
@click.argument('analysis1_file', type=click.Path(exists=True))
@click.argument('analysis2_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Save comparison report to file')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed comparison report')
@click.option('--json-output', is_flag=True, help='Output comparison as JSON')
def compare(analysis1_file, analysis2_file, output, verbose, json_output):
    """
    Compare two analysis results by content.
    
    ANALYSIS1_FILE: First analysis JSON file
    ANALYSIS2_FILE: Second analysis JSON file
    """
    try:
        # Perform comparison
        comparison = compare_analysis_files(
            analysis1_file, 
            analysis2_file, 
            output_report=verbose and not json_output
        )
        
        # Handle errors
        if 'error' in comparison:
            click.echo(f"❌ Comparison failed: {comparison['error']}")
            sys.exit(1)
        
        # Generate output
        if json_output:
            # JSON output
            output_data = json.dumps(comparison, indent=2)
            if output:
                with open(output, 'w') as f:
                    f.write(output_data)
                click.echo(f"📄 Comparison saved to: {output}")
            else:
                click.echo(output_data)
        else:
            # Human-readable output
            from analyzer.utils.analysis_comparator import AnalysisComparator
            comparator = AnalysisComparator()
            report = comparator.generate_comparison_report(comparison)
            
            if output:
                with open(output, 'w') as f:
                    f.write(report)
                click.echo(f"📄 Comparison report saved to: {output}")
            elif not verbose:  # Only show summary if not already shown
                summary = comparison.get('summary', {})
                if summary.get('identical', False):
                    click.echo("✅ Analysis results are functionally identical!")
                    click.echo("   (UUIDs may differ but content is the same)")
                else:
                    click.echo(f"❌ Found {summary.get('differences_count', 0)} differences")
                    
                    # Show brief stats
                    click.echo(f"   Nodes: {summary.get('total_nodes', {}).get('first', 0)} vs {summary.get('total_nodes', {}).get('second', 0)}")
                    click.echo(f"   Relationships: {summary.get('total_relationships', {}).get('first', 0)} vs {summary.get('total_relationships', {}).get('second', 0)}")
                    click.echo("   Use --verbose for detailed differences")
    
    except Exception as e:
        click.echo(f"❌ Comparison error: {e}")
        sys.exit(1)


@click.command()
@click.argument('source', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('target', type=click.Path())
@click.option('--llm', type=click.Choice(['none', 'local', 'openai', 'anthropic']), 
              default='none', help='LLM provider for enhanced analysis (default: none)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', type=click.Path(exists=True), 
              help='Custom config file (default: config/llm_config.yaml)')
@click.option('--enable-execution', is_flag=True, 
              help='Enable execution tracing (experimental)')
@click.option('--execution-timeout', type=int, default=30,
              help='Execution timeout in seconds (default: 30)')
def hybrid_analyze(source, target, llm, verbose, config, enable_execution, execution_timeout):
    """
    Hybrid analysis combining static and dynamic analysis.
    
    SOURCE: Path to Python project directory
    TARGET: Output JSON file path
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        if config:
            analyzer_config = load_unified_config(config, llm_provider=llm)
        else:
            default_config = Path(__file__).parent.parent.parent / 'config' / 'llm_config.yaml'
            analyzer_config = load_unified_config(str(default_config), llm_provider=llm)
        
        click.echo(f"🔍 Hybrid Analysis: {source}")
        click.echo(f"📁 Output: {target}")
        click.echo(f"🤖 LLM Provider: {llm}")
        click.echo(f"⚡ Execution Tracing: {'Enabled' if enable_execution else 'Disabled'}")
        
        # Initialize hybrid analyzer
        hybrid_analyzer = HybridAnalyzer(
            config=analyzer_config,
            enable_tracing=enable_execution
        )
        
        # Validate project
        click.echo("🔍 Validating project...")
        is_valid, errors = hybrid_analyzer.static_analyzer.validate_project(source)
        
        if not is_valid:
            click.echo("❌ Project validation failed:")
            for error in errors:
                click.echo(f"  • {error}")
            sys.exit(1)
        
        click.echo("✅ Project validation passed")
        
        # Perform hybrid analysis
        click.echo("🚀 Starting hybrid analysis...")
        
        with click.progressbar(length=100, label='Analyzing') as bar:
            result = hybrid_analyzer.analyze_with_tracing(source)
            bar.update(100)
        
        # Check results
        if not result.static_result.is_successful():
            click.echo("❌ Analysis failed:")
            for error in result.static_result.errors[:3]:
                click.echo(f"  • {error}")
            if len(result.static_result.errors) > 3:
                click.echo(f"  ... and {len(result.static_result.errors) - 3} more errors")
            sys.exit(1)
        
        # Generate output
        click.echo(f"💾 Generating output...")
        
        # Convert analysis result to JSON format
        output_data = hybrid_analyzer.format_hybrid_results(result)
        
        # Ensure target directory exists
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write output file
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Show results
        click.echo("\n" + "="*50)
        click.echo("✅ HYBRID ANALYSIS COMPLETED")
        click.echo("="*50)
        click.echo(f"📄 Files analyzed: {result.static_result.stats.files_parsed}")
        click.echo(f"🔧 Functions found: {result.static_result.stats.functions_found}")
        click.echo(f"📦 Classes found: {result.static_result.stats.classes_found}")
        click.echo(f"🎭 Trigger actors: {result.static_result.stats.trigger_actors}")
        click.echo(f"📡 Receiver actors: {result.static_result.stats.receiver_actors}")
        click.echo(f"🔗 Flow chains: {result.static_result.stats.flow_chains}")
        click.echo(f"🎯 Total nodes: {result.static_result.stats.sys_nodes + result.static_result.stats.mod_nodes + result.static_result.stats.func_nodes + result.static_result.stats.actor_nodes}")
        click.echo(f"⏱️  Analysis time: {result.static_result.stats.analysis_time_seconds:.2f}s")
        
        # Show hybrid-specific stats
        if result.trace_result:
            click.echo(f"⚡ Functions traced: {len(result.trace_result.traced_functions)}")
            click.echo(f"📊 Modules traced: {len(result.trace_result.modules_traced)}")
            if result.trace_result.trace_errors:
                click.echo(f"⚠️  Trace errors: {len(result.trace_result.trace_errors)}")
        
        if len(result.hybrid_confidence_scores) > 0:
            avg_confidence = sum(result.hybrid_confidence_scores.values()) / len(result.hybrid_confidence_scores)
            click.echo(f"🎯 Average confidence: {avg_confidence:.3f}")
        
        if result.static_result.stats.llm_calls_made > 0:
            click.echo(f"🤖 LLM calls made: {result.static_result.stats.llm_calls_made}")
            click.echo(f"✨ Actors enhanced: {result.static_result.stats.actors_enhanced}")
        
        # Show file info
        file_size = os.path.getsize(target)
        if file_size > 1024 * 1024:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size} bytes"
        
        click.echo(f"📊 Output size: {size_str}")
        click.echo(f"📁 Saved to: {target}")
        click.echo("="*50)
        
    except KeyboardInterrupt:
        click.echo("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Hybrid analysis failed")
        click.echo(f"❌ Hybrid analysis failed: {e}")
        sys.exit(1)


@click.command()
@click.argument('analysis_file', type=click.Path(exists=True))
@click.option('--ground-truth', type=click.Path(exists=True),
              help='Ground truth file for validation')
@click.option('--output', '-o', type=click.Path(), 
              help='Save validation report to file')
@click.option('--json-output', is_flag=True, 
              help='Output validation as JSON')
@click.option('--verbose', '-v', is_flag=True, 
              help='Show detailed validation report')
def validate_results(analysis_file, ground_truth, output, json_output, verbose):
    """
    Validate analysis results against ground truth.
    
    ANALYSIS_FILE: Analysis JSON file to validate
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        click.echo(f"🔍 Validating: {analysis_file}")
        if ground_truth:
            click.echo(f"📋 Ground truth: {ground_truth}")
        else:
            click.echo("📋 Ground truth: Not provided (heuristic validation)")
        
        # Initialize validator
        validator = AnalysisResultValidator(ground_truth)
        
        # Perform validation
        click.echo("🚀 Starting validation...")
        validation_report = validator.validate_analysis_result(analysis_file)
        
        # Generate output
        if json_output:
            # JSON output
            import dataclasses
            report_dict = dataclasses.asdict(validation_report)
            output_data = json.dumps(report_dict, indent=2, default=str)
            
            if output:
                with open(output, 'w') as f:
                    f.write(output_data)
                click.echo(f"📄 Validation results saved to: {output}")
            else:
                click.echo(output_data)
        else:
            # Human-readable report
            report_text = validator.generate_validation_report(validation_report)
            
            if output:
                with open(output, 'w') as f:
                    f.write(report_text)
                click.echo(f"📄 Validation report saved to: {output}")
            else:
                click.echo(report_text)
        
        # Show summary
        if not json_output and not output:
            click.echo("\n" + "="*50)
            click.echo("📊 VALIDATION SUMMARY")
            click.echo("="*50)
            click.echo(f"Overall Score: {validation_report.overall_score:.3f}")
            
            if validation_report.dead_code_validation:
                dc = validation_report.dead_code_validation
                click.echo(f"Dead Code F1-Score: {dc.accuracy_metrics.f1_score:.3f}")
            
            if validation_report.flow_validation:
                fv = validation_report.flow_validation
                click.echo(f"Flow Analysis F1-Score: {fv.accuracy_metrics.f1_score:.3f}")
            
            if validation_report.recommendations:
                click.echo(f"Recommendations: {len(validation_report.recommendations)}")
            
            click.echo("="*50)
        
    except Exception as e:
        logger.exception("Validation failed")
        click.echo(f"❌ Validation failed: {e}")
        sys.exit(1)


@click.command()
@click.argument('source', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('target', type=click.Path())
@click.option('--llm', type=click.Choice(['none', 'local', 'openai', 'anthropic']), 
              default='none', help='LLM provider for enhanced analysis (default: none)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', type=click.Path(exists=True), 
              help='Custom config file (default: config/llm_config.yaml)')
def logical_analyze(source, target, llm, verbose, config):
    """
    Logical dependency analysis generating abstract business flows.
    
    SOURCE: Path to Python project directory
    TARGET: Output JSON file path
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        if config:
            analyzer_config = load_unified_config(config, llm_provider=llm)
        else:
            default_config = Path(__file__).parent.parent.parent / 'config' / 'llm_config.yaml'
            analyzer_config = load_unified_config(str(default_config), llm_provider=llm)
        
        click.echo(f"🧠 Logical Analysis: {source}")
        click.echo(f"📁 Output: {target}")
        click.echo(f"🤖 LLM Provider: {llm}")
        click.echo(f"🔍 Analysis Type: Abstract Logical Dependencies")
        
        # Initialize logical analyzer
        analyzer = LogicalAnalyzer(analyzer_config)
        
        # Validate project
        click.echo("🔍 Validating project...")
        is_valid, errors = analyzer.validate_project(source)
        
        if not is_valid:
            click.echo("❌ Project validation failed:")
            for error in errors:
                click.echo(f"  • {error}")
            sys.exit(1)
        
        click.echo("✅ Project validation passed")
        
        # Perform logical analysis
        click.echo("🚀 Starting logical analysis...")
        
        with click.progressbar(length=100, label='Analyzing') as bar:
            result = analyzer.analyze(source)
            bar.update(100)
        
        # Check results
        if not result.is_successful():
            click.echo("❌ Analysis failed:")
            for error in result.errors[:3]:  # Show first 3 errors
                click.echo(f"  • {error}")
            if len(result.errors) > 3:
                click.echo(f"  ... and {len(result.errors) - 3} more errors")
            sys.exit(1)
        
        # Generate output
        click.echo(f"💾 Generating output...")
        
        # Convert analysis result to JSON format
        output_data = analyzer.format_for_export(result)
        
        # Ensure target directory exists
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write output file
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Show results
        click.echo("\n" + "="*50)
        click.echo("✅ LOGICAL ANALYSIS COMPLETED")
        click.echo("="*50)
        click.echo(f"📄 Files analyzed: {result.stats.files_parsed}")
        click.echo(f"🔧 Functions found: {result.stats.functions_found}")
        click.echo(f"📦 Classes found: {result.stats.classes_found}")
        click.echo(f"🏛️  Logical actors: {result.stats.logical_actors}")
        click.echo(f"🌐 Business domains: {result.stats.business_domains}")
        click.echo(f"🔄 Logical flows: {result.stats.logical_flows}")
        click.echo(f"⚡ Data transformations: {result.stats.data_transformations}")
        click.echo(f"🎯 Total nodes: {result.stats.sys_nodes + result.stats.mod_nodes + result.stats.func_nodes + result.stats.actor_nodes}")
        click.echo(f"⏱️  Analysis time: {result.stats.analysis_time_seconds:.2f}s")
        
        # Show logical breakdown
        if result.stats.domain_services > 0:
            click.echo(f"🏢 Domain services: {result.stats.domain_services}")
        if result.stats.business_entities > 0:
            click.echo(f"📋 Business entities: {result.stats.business_entities}")
        if result.stats.integration_points > 0:
            click.echo(f"🔗 Integration points: {result.stats.integration_points}")
        if result.stats.workflow_orchestrators > 0:
            click.echo(f"🎭 Workflow orchestrators: {result.stats.workflow_orchestrators}")
        
        if result.stats.llm_calls_made > 0:
            click.echo(f"🤖 LLM calls made: {result.stats.llm_calls_made}")
            click.echo(f"✨ Actors enhanced: {result.stats.actors_enhanced}")
        
        # Show file info
        file_size = os.path.getsize(target)
        if file_size > 1024 * 1024:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size} bytes"
        
        click.echo(f"📊 Output size: {size_str}")
        click.echo(f"📁 Saved to: {target}")
        click.echo("="*50)
        
    except KeyboardInterrupt:
        click.echo("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Logical analysis failed")
        click.echo(f"❌ Logical analysis failed: {e}")
        sys.exit(1)


@click.group()
@click.version_option(version='2.0.0', prog_name='Code Architecture Analyzer')
def cli():
    """Code Architecture Analyzer - Simple CLI for Python project analysis."""
    pass


# Register commands
cli.add_command(analyze)
cli.add_command(logical_analyze)
cli.add_command(validate)
cli.add_command(compare)
cli.add_command(hybrid_analyze)
cli.add_command(validate_results)


def main():
    """Main entry point."""
    try:
        cli()
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()