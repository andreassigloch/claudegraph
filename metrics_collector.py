#!/usr/bin/env python3
"""
Metrics Collector for ClaudeGraph v3 Validation
Parses Claude Code output to extract real metrics
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ToolCall:
    tool_name: str
    parameters: Dict[str, any]
    timestamp: float
    
@dataclass
class ClaudeMetrics:
    test_name: str
    group: str
    total_tokens: int
    tool_calls: List[ToolCall]
    files_searched: List[str]
    files_modified: List[str]
    errors: List[str]
    time_seconds: float
    search_attempts: int
    grep_count: int
    read_count: int
    glob_count: int

class MetricsCollector:
    """Collects and parses metrics from Claude Code execution"""
    
    def __init__(self):
        self.tool_patterns = {
            'Read': re.compile(r'Called the Read tool.*?file_path["\']:\s*["\'](.*?)["\']', re.DOTALL),
            'Grep': re.compile(r'Called the Grep tool.*?pattern["\']:\s*["\'](.*?)["\']', re.DOTALL),
            'Glob': re.compile(r'Called the Glob tool.*?pattern["\']:\s*["\'](.*?)["\']', re.DOTALL),
            'Edit': re.compile(r'Called the Edit tool.*?file_path["\']:\s*["\'](.*?)["\']', re.DOTALL),
            'Write': re.compile(r'Called the Write tool.*?file_path["\']:\s*["\'](.*?)["\']', re.DOTALL),
            'Task': re.compile(r'Called the Task tool.*?description["\']:\s*["\'](.*?)["\']', re.DOTALL),
        }
        
        # Token usage patterns (from Claude's output)
        self.token_pattern = re.compile(r'(\d+(?:,\d+)?)\s*tokens')
        self.time_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(?:seconds?|s)')
        
    def parse_claude_output(self, output_file: Path) -> ClaudeMetrics:
        """Parse Claude Code output file to extract metrics"""
        
        if not output_file.exists():
            raise FileNotFoundError(f"Output file not found: {output_file}")
            
        content = output_file.read_text()
        
        # Extract test name and group from filename
        match = re.match(r'output_(\w+)_group_(\w+)\.txt', output_file.name)
        test_name = match.group(1) if match else "Unknown"
        group = match.group(2) if match else "Unknown"
        
        # Count tool calls
        tool_calls = []
        files_searched = set()
        files_modified = set()
        
        # Parse Read tool calls
        for match in self.tool_patterns['Read'].finditer(content):
            file_path = match.group(1)
            files_searched.add(file_path)
            tool_calls.append(ToolCall('Read', {'file_path': file_path}, 0))
            
        # Parse Grep tool calls
        grep_matches = list(self.tool_patterns['Grep'].finditer(content))
        grep_count = len(grep_matches)
        for match in grep_matches:
            pattern = match.group(1)
            tool_calls.append(ToolCall('Grep', {'pattern': pattern}, 0))
            
        # Parse Glob tool calls
        glob_matches = list(self.tool_patterns['Glob'].finditer(content))
        glob_count = len(glob_matches)
        for match in glob_matches:
            pattern = match.group(1)
            tool_calls.append(ToolCall('Glob', {'pattern': pattern}, 0))
            
        # Parse Edit/Write tool calls
        for tool in ['Edit', 'Write']:
            for match in self.tool_patterns[tool].finditer(content):
                file_path = match.group(1)
                files_modified.add(file_path)
                tool_calls.append(ToolCall(tool, {'file_path': file_path}, 0))
        
        # Parse Task tool calls (search operations)
        task_matches = list(self.tool_patterns['Task'].finditer(content))
        search_attempts = len(task_matches)
        
        # Extract token usage
        token_matches = self.token_pattern.findall(content)
        total_tokens = 0
        if token_matches:
            # Sum all token mentions (Claude often reports incremental usage)
            for token_str in token_matches:
                tokens = int(token_str.replace(',', ''))
                total_tokens += tokens
                
        # Extract time
        time_matches = self.time_pattern.findall(content)
        total_time = 0
        if time_matches:
            for time_str in time_matches:
                total_time += float(time_str)
                
        # Extract errors
        errors = []
        error_patterns = [
            r'Error:.*',
            r'Failed to.*',
            r'Could not find.*',
            r'Schema mismatch.*'
        ]
        for pattern in error_patterns:
            errors.extend(re.findall(pattern, content, re.IGNORECASE))
            
        return ClaudeMetrics(
            test_name=test_name,
            group=group,
            total_tokens=total_tokens,
            tool_calls=tool_calls,
            files_searched=list(files_searched),
            files_modified=list(files_modified),
            errors=errors,
            time_seconds=total_time,
            search_attempts=search_attempts,
            grep_count=grep_count,
            read_count=len([t for t in tool_calls if t.tool_name == 'Read']),
            glob_count=glob_count
        )
    
    def compare_metrics(self, group_a: ClaudeMetrics, group_b: ClaudeMetrics) -> Dict:
        """Compare metrics between Group A and Group B"""
        
        comparison = {
            "test_name": group_a.test_name,
            "timestamp": datetime.now().isoformat(),
            "group_a": asdict(group_a),
            "group_b": asdict(group_b),
            "improvements": {
                "token_reduction": self._calculate_reduction(group_a.total_tokens, group_b.total_tokens),
                "time_reduction": self._calculate_reduction(group_a.time_seconds, group_b.time_seconds),
                "search_reduction": self._calculate_reduction(group_a.search_attempts, group_b.search_attempts),
                "grep_reduction": self._calculate_reduction(group_a.grep_count, group_b.grep_count),
                "files_searched_reduction": self._calculate_reduction(
                    len(group_a.files_searched), len(group_b.files_searched)
                )
            },
            "analysis": {
                "error_comparison": {
                    "group_a_errors": len(group_a.errors),
                    "group_b_errors": len(group_b.errors),
                    "error_reduction": len(group_a.errors) - len(group_b.errors)
                },
                "efficiency": {
                    "group_a_tools_per_file": len(group_a.tool_calls) / max(len(group_a.files_modified), 1),
                    "group_b_tools_per_file": len(group_b.tool_calls) / max(len(group_b.files_modified), 1)
                }
            }
        }
        
        return comparison
    
    def _calculate_reduction(self, before: float, after: float) -> float:
        """Calculate percentage reduction"""
        if before == 0:
            return 0
        return ((before - after) / before) * 100
    
    def generate_report(self, comparisons: List[Dict]) -> str:
        """Generate a comprehensive report from all comparisons"""
        
        report = ["# ClaudeGraph v3 Real Validation Report", ""]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        report.append("## Summary Statistics")
        report.append("")
        
        avg_token_reduction = sum(c["improvements"]["token_reduction"] for c in comparisons) / len(comparisons)
        avg_time_reduction = sum(c["improvements"]["time_reduction"] for c in comparisons) / len(comparisons)
        avg_search_reduction = sum(c["improvements"]["search_reduction"] for c in comparisons) / len(comparisons)
        
        report.append(f"- **Average Token Reduction**: {avg_token_reduction:.1f}%")
        report.append(f"- **Average Time Reduction**: {avg_time_reduction:.1f}%")
        report.append(f"- **Average Search Reduction**: {avg_search_reduction:.1f}%")
        report.append("")
        
        # Detailed results by test
        report.append("## Detailed Results by Test")
        report.append("")
        
        for comp in comparisons:
            report.append(f"### {comp['test_name']}")
            report.append("")
            
            # Metrics table
            report.append("| Metric | Group A (No Flow) | Group B (With Flow) | Reduction |")
            report.append("|--------|-------------------|---------------------|-----------|")
            
            a = comp["group_a"]
            b = comp["group_b"]
            imp = comp["improvements"]
            
            report.append(f"| Tokens | {a['total_tokens']:,} | {b['total_tokens']:,} | {imp['token_reduction']:.1f}% |")
            report.append(f"| Time | {a['time_seconds']:.1f}s | {b['time_seconds']:.1f}s | {imp['time_reduction']:.1f}% |")
            report.append(f"| Searches | {a['search_attempts']} | {b['search_attempts']} | {imp['search_reduction']:.1f}% |")
            report.append(f"| Grep Calls | {a['grep_count']} | {b['grep_count']} | {imp['grep_reduction']:.1f}% |")
            report.append(f"| Files Searched | {len(a['files_searched'])} | {len(b['files_searched'])} | {imp['files_searched_reduction']:.1f}% |")
            report.append(f"| Errors | {comp['analysis']['error_comparison']['group_a_errors']} | {comp['analysis']['error_comparison']['group_b_errors']} | -{comp['analysis']['error_comparison']['error_reduction']} |")
            report.append("")
            
        return "\n".join(report)
    
    def save_metrics(self, metrics: ClaudeMetrics, output_dir: Path = Path("metrics")):
        """Save metrics to JSON file"""
        output_dir.mkdir(exist_ok=True)
        
        filename = f"metrics_{metrics.test_name}_group_{metrics.group}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file = output_dir / filename
        
        output_file.write_text(json.dumps(asdict(metrics), indent=2))
        return output_file

def main():
    """Example usage of MetricsCollector"""
    collector = MetricsCollector()
    
    # Example: Parse output files
    output_dir = Path("claude_outputs")
    if output_dir.exists():
        comparisons = []
        
        # Process each test
        for test in ["UC1", "UC2", "UC3", "UC4"]:
            group_a_file = output_dir / f"output_{test}_group_A.txt"
            group_b_file = output_dir / f"output_{test}_group_B.txt"
            
            if group_a_file.exists() and group_b_file.exists():
                metrics_a = collector.parse_claude_output(group_a_file)
                metrics_b = collector.parse_claude_output(group_b_file)
                
                # Save individual metrics
                collector.save_metrics(metrics_a)
                collector.save_metrics(metrics_b)
                
                # Compare
                comparison = collector.compare_metrics(metrics_a, metrics_b)
                comparisons.append(comparison)
        
        # Generate report
        if comparisons:
            report = collector.generate_report(comparisons)
            report_file = Path("validation_report.md")
            report_file.write_text(report)
            print(f"Report saved to: {report_file}")
            
            # Also save JSON comparisons
            json_file = Path("validation_comparisons.json")
            json_file.write_text(json.dumps(comparisons, indent=2))
            print(f"Comparisons saved to: {json_file}")

if __name__ == "__main__":
    main()