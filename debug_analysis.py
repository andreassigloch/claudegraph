#!/usr/bin/env python3
"""
Debug analysis to understand result structure
"""

import sys
sys.path.insert(0, '.')

from analyzer.core.analyzer import DeterministicAnalyzer

def debug_analysis():
    """Debug the analysis result structure"""
    
    print("🔍 Debug Analysis")
    print("=" * 40)
    
    try:
        # Initialize analyzer
        analyzer = DeterministicAnalyzer()
        
        # Analyze project
        result = analyzer.analyze("/Users/andreas/Documents/Tools/Eclipse_workspace/RealPyTest")
        
        print(f"Result type: {type(result)}")
        print(f"Has graph_data: {hasattr(result, 'graph_data')}")
        print(f"Graph_data type: {type(result.graph_data) if result.graph_data else 'None'}")
        print(f"Graph_data content keys: {result.graph_data.keys() if result.graph_data and hasattr(result.graph_data, 'keys') else 'N/A'}")
        print(f"Is successful: {result.is_successful()}")
        print(f"Errors: {len(result.errors)}")
        if result.errors:
            print("First few errors:")
            for error in result.errors[:3]:
                print(f"  - {error}")
        
        if result.graph_data:
            if hasattr(result.graph_data, 'nodes'):
                print(f"Nodes count: {len(result.graph_data.nodes)}")
                print(f"Relationships count: {len(result.graph_data.relationships)}")
            else:
                print(f"Graph data attributes: {dir(result.graph_data)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_analysis()