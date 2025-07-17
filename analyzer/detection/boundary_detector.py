#!/usr/bin/env python3
"""
Improved ACTOR Generation with Boundary Detection

This module implements a sophisticated boundary detection system that only creates
ACTORs for real external interfaces that are actually used in the code.

Flow Direction: Trigger → Functions → Receiver
- Trigger ACTORs: External sources that initiate actions (User, API calls, etc.)
- Functions: Internal business logic that processes requests
- Receiver ACTORs: External destinations that receive results (DB, API responses, etc.)
"""

import ast
import re
import uuid
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class BoundaryType(Enum):
    """Types of system boundaries that generate ACTORs"""
    HTTP_INBOUND = "http_inbound"      # Triggers: @app.post, @router.get
    HTTP_OUTBOUND = "http_outbound"    # Receivers: requests.post, httpx.get
    DATABASE = "database"              # Receivers: db.execute, cursor.query
    FILESYSTEM = "filesystem"          # Receivers: open(), file operations
    SUBPROCESS = "subprocess"          # Receivers: subprocess.run
    MESSAGE_QUEUE = "message_queue"    # Both: publish, subscribe
    USER_INTERFACE = "user_interface"  # Triggers: CLI input, GUI events


class ActorRole(Enum):
    """Role of ACTOR in the flow"""
    TRIGGER = "trigger"     # Initiates action (external → system)
    RECEIVER = "receiver"   # Receives result (system → external)


@dataclass
class BoundaryCall:
    """A detected function call that crosses system boundary"""
    function_name: str
    call_code: str
    call_type: BoundaryType
    role: ActorRole
    source_function: str
    line_number: int
    extracted_info: Dict[str, Any]


@dataclass
class ActorCandidate:
    """A potential ACTOR before validation"""
    name: str
    description: str
    actor_type: str
    role: ActorRole
    boundary_type: BoundaryType
    source_functions: List[str]
    metadata: Dict[str, Any]


class BaseBoundaryDetector:
    """Base class for boundary detectors"""
    
    def __init__(self):
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> List[Dict[str, Any]]:
        """Load detection patterns for this boundary type"""
        raise NotImplementedError
    
    def detect_boundaries(self, ast_tree: ast.AST, source_code: str) -> List[BoundaryCall]:
        """Detect boundary calls in AST"""
        raise NotImplementedError
    
    def create_actor_from_boundary(self, boundary_call: BoundaryCall) -> ActorCandidate:
        """Create ACTOR candidate from boundary call"""
        raise NotImplementedError


class HttpBoundaryDetector(BaseBoundaryDetector):
    """Detects HTTP boundary crossings"""
    
    def _load_patterns(self) -> List[Dict[str, Any]]:
        return [
            # Inbound HTTP (Triggers)
            {"pattern": r"@app\.(get|post|put|delete|patch)", "role": ActorRole.TRIGGER, "type": "fastapi"},
            {"pattern": r"@router\.(get|post|put|delete|patch)", "role": ActorRole.TRIGGER, "type": "fastapi"},
            {"pattern": r"@flask_app\.route", "role": ActorRole.TRIGGER, "type": "flask"},
            {"pattern": r"def (get|post|put|delete|patch)_", "role": ActorRole.TRIGGER, "type": "rest_method"},
            
            # Outbound HTTP (Receivers)
            {"pattern": r"requests\.(get|post|put|delete|patch)", "role": ActorRole.RECEIVER, "type": "requests"},
            {"pattern": r"httpx\.(get|post|put|delete|patch)", "role": ActorRole.RECEIVER, "type": "httpx"},
            {"pattern": r"aiohttp\.ClientSession", "role": ActorRole.RECEIVER, "type": "aiohttp"},
            {"pattern": r"urllib\.request\.urlopen", "role": ActorRole.RECEIVER, "type": "urllib"},
        ]
    
    def detect_boundaries(self, ast_tree: ast.AST, source_code: str) -> List[BoundaryCall]:
        boundaries = []
        
        class HttpVisitor(ast.NodeVisitor):
            def __init__(self, detector):
                self.detector = detector
                self.current_function = None
                self.source_lines = source_code.split('\n')
            
            def visit_FunctionDef(self, node):
                old_function = self.current_function
                self.current_function = node.name
                
                # Check decorators for inbound HTTP
                for decorator in node.decorator_list:
                    decorator_code = ast.get_source_segment(source_code, decorator) or ""
                    for pattern_info in self.detector.patterns:
                        if pattern_info["role"] == ActorRole.TRIGGER:
                            if re.search(pattern_info["pattern"], decorator_code):
                                boundaries.append(BoundaryCall(
                                    function_name=node.name,
                                    call_code=decorator_code,
                                    call_type=BoundaryType.HTTP_INBOUND,
                                    role=ActorRole.TRIGGER,
                                    source_function=node.name,
                                    line_number=node.lineno,
                                    extracted_info=self.detector._extract_http_info(decorator_code, pattern_info)
                                ))
                
                self.generic_visit(node)
                self.current_function = old_function
            
            def visit_Call(self, node):
                if self.current_function:
                    call_code = ast.get_source_segment(source_code, node) or ""
                    for pattern_info in self.detector.patterns:
                        if pattern_info["role"] == ActorRole.RECEIVER:
                            if re.search(pattern_info["pattern"], call_code):
                                boundaries.append(BoundaryCall(
                                    function_name=self.current_function,
                                    call_code=call_code,
                                    call_type=BoundaryType.HTTP_OUTBOUND,
                                    role=ActorRole.RECEIVER,
                                    source_function=self.current_function,
                                    line_number=node.lineno,
                                    extracted_info=self.detector._extract_http_info(call_code, pattern_info)
                                ))
                
                self.generic_visit(node)
        
        visitor = HttpVisitor(self)
        visitor.visit(ast_tree)
        return boundaries
    
    def _extract_http_info(self, code: str, pattern_info: Dict) -> Dict[str, Any]:
        """Extract HTTP-specific information from code"""
        info = {"library": pattern_info.get("type", "unknown")}
        
        # Extract HTTP method
        method_match = re.search(r'\.(get|post|put|delete|patch)', code, re.IGNORECASE)
        if method_match:
            info["method"] = method_match.group(1).upper()
        
        # Extract URL/endpoint
        url_match = re.search(r'["\']([^"\']+)["\']', code)
        if url_match:
            info["endpoint"] = url_match.group(1)
        
        return info
    
    def create_actor_from_boundary(self, boundary_call: BoundaryCall) -> ActorCandidate:
        info = boundary_call.extracted_info
        
        if boundary_call.role == ActorRole.TRIGGER:
            # Inbound HTTP endpoint
            endpoint = info.get("endpoint", "unknown")
            method = info.get("method", "GET")
            return ActorCandidate(
                name=f"WebEndpoint{method}{endpoint.replace('/', '_').replace('-', '_')}",
                description=f"Web endpoint serving {method} requests to {endpoint}",
                actor_type="WebEndpoint",
                role=ActorRole.TRIGGER,
                boundary_type=BoundaryType.HTTP_INBOUND,
                source_functions=[boundary_call.source_function],
                metadata={
                    "HttpMethod": method,
                    "Endpoint": endpoint,
                    "Framework": info.get("library", "unknown")
                }
            )
        else:
            # Outbound HTTP call
            endpoint = info.get("endpoint", "external_api")
            method = info.get("method", "GET")
            return ActorCandidate(
                name=f"HttpClient{method}{endpoint.replace('/', '_').replace('-', '_')}",
                description=f"External HTTP service called via {method} {endpoint}",
                actor_type="HttpClient",
                role=ActorRole.RECEIVER,
                boundary_type=BoundaryType.HTTP_OUTBOUND,
                source_functions=[boundary_call.source_function],
                metadata={
                    "HttpMethod": method,
                    "Endpoint": endpoint,
                    "Library": info.get("library", "unknown")
                }
            )


class DatabaseBoundaryDetector(BaseBoundaryDetector):
    """Detects database boundary crossings"""
    
    def _load_patterns(self) -> List[Dict[str, Any]]:
        return [
            {"pattern": r"\.execute\(", "role": ActorRole.RECEIVER, "type": "sql"},
            {"pattern": r"\.query\(", "role": ActorRole.RECEIVER, "type": "sql"},
            {"pattern": r"\.find\(", "role": ActorRole.RECEIVER, "type": "mongodb"},
            {"pattern": r"\.insert\(", "role": ActorRole.RECEIVER, "type": "sql"},
            {"pattern": r"\.update\(", "role": ActorRole.RECEIVER, "type": "sql"},
            {"pattern": r"\.delete\(", "role": ActorRole.RECEIVER, "type": "sql"},
            {"pattern": r"connection\.", "role": ActorRole.RECEIVER, "type": "connection"},
            {"pattern": r"cursor\.", "role": ActorRole.RECEIVER, "type": "cursor"},
            {"pattern": r"Neo4jDriver", "role": ActorRole.RECEIVER, "type": "neo4j"},
            {"pattern": r"driver\.session", "role": ActorRole.RECEIVER, "type": "neo4j"},
        ]
    
    def detect_boundaries(self, ast_tree: ast.AST, source_code: str) -> List[BoundaryCall]:
        boundaries = []
        
        class DatabaseVisitor(ast.NodeVisitor):
            def __init__(self, detector):
                self.detector = detector
                self.current_function = None
            
            def visit_FunctionDef(self, node):
                old_function = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = old_function
            
            def visit_Call(self, node):
                if self.current_function:
                    call_code = ast.get_source_segment(source_code, node) or ""
                    for pattern_info in self.detector.patterns:
                        if re.search(pattern_info["pattern"], call_code):
                            boundaries.append(BoundaryCall(
                                function_name=self.current_function,
                                call_code=call_code,
                                call_type=BoundaryType.DATABASE,
                                role=ActorRole.RECEIVER,
                                source_function=self.current_function,
                                line_number=node.lineno,
                                extracted_info=self.detector._extract_db_info(call_code, pattern_info)
                            ))
                
                self.generic_visit(node)
        
        visitor = DatabaseVisitor(self)
        visitor.visit(ast_tree)
        return boundaries
    
    def _extract_db_info(self, code: str, pattern_info: Dict) -> Dict[str, Any]:
        """Extract database-specific information"""
        info = {"db_type": pattern_info.get("type", "unknown")}
        
        # Extract operation type
        if ".execute(" in code or ".query(" in code:
            info["operation"] = "query"
        elif ".insert(" in code:
            info["operation"] = "insert"
        elif ".update(" in code:
            info["operation"] = "update"
        elif ".delete(" in code:
            info["operation"] = "delete"
        else:
            info["operation"] = "access"
        
        return info
    
    def create_actor_from_boundary(self, boundary_call: BoundaryCall) -> ActorCandidate:
        info = boundary_call.extracted_info
        db_type = info.get("db_type", "Database")
        operation = info.get("operation", "access")
        
        return ActorCandidate(
            name=f"Database{db_type.title()}{operation.title()}",
            description=f"{db_type.title()} database for {operation} operations",
            actor_type="Database",
            role=ActorRole.RECEIVER,
            boundary_type=BoundaryType.DATABASE,
            source_functions=[boundary_call.source_function],
            metadata={
                "DatabaseType": db_type,
                "Operation": operation,
                "Library": info.get("library", "unknown")
            }
        )


class FileSystemBoundaryDetector(BaseBoundaryDetector):
    """Detects filesystem boundary crossings"""
    
    def _load_patterns(self) -> List[Dict[str, Any]]:
        return [
            {"pattern": r"open\(", "role": ActorRole.RECEIVER, "type": "file_io"},
            {"pattern": r"with open\(", "role": ActorRole.RECEIVER, "type": "file_io"},
            {"pattern": r"pathlib\.Path", "role": ActorRole.RECEIVER, "type": "pathlib"},
            {"pattern": r"os\.path\.", "role": ActorRole.RECEIVER, "type": "os_path"},
            {"pattern": r"\.read\(\)", "role": ActorRole.RECEIVER, "type": "file_read"},
            {"pattern": r"\.write\(", "role": ActorRole.RECEIVER, "type": "file_write"},
            {"pattern": r"json\.load\(", "role": ActorRole.RECEIVER, "type": "json_io"},
            {"pattern": r"json\.dump\(", "role": ActorRole.RECEIVER, "type": "json_io"},
            {"pattern": r"csv\.reader\(", "role": ActorRole.RECEIVER, "type": "csv_io"},
            {"pattern": r"csv\.writer\(", "role": ActorRole.RECEIVER, "type": "csv_io"},
        ]
    
    def detect_boundaries(self, ast_tree: ast.AST, source_code: str) -> List[BoundaryCall]:
        boundaries = []
        
        class FileSystemVisitor(ast.NodeVisitor):
            def __init__(self, detector):
                self.detector = detector
                self.current_function = None
            
            def visit_FunctionDef(self, node):
                old_function = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = old_function
            
            def visit_Call(self, node):
                if self.current_function:
                    call_code = ast.get_source_segment(source_code, node) or ""
                    for pattern_info in self.detector.patterns:
                        if re.search(pattern_info["pattern"], call_code):
                            boundaries.append(BoundaryCall(
                                function_name=self.current_function,
                                call_code=call_code,
                                call_type=BoundaryType.FILESYSTEM,
                                role=ActorRole.RECEIVER,
                                source_function=self.current_function,
                                line_number=node.lineno,
                                extracted_info=self.detector._extract_fs_info(call_code, pattern_info)
                            ))
                
                self.generic_visit(node)
        
        visitor = FileSystemVisitor(self)
        visitor.visit(ast_tree)
        return boundaries
    
    def _extract_fs_info(self, code: str, pattern_info: Dict) -> Dict[str, Any]:
        """Extract filesystem-specific information"""
        info = {"fs_type": pattern_info.get("type", "filesystem")}
        
        # Determine operation type
        if "read" in code.lower():
            info["operation"] = "read"
        elif "write" in code.lower() or "dump" in code.lower():
            info["operation"] = "write"
        else:
            info["operation"] = "access"
        
        return info
    
    def create_actor_from_boundary(self, boundary_call: BoundaryCall) -> ActorCandidate:
        info = boundary_call.extracted_info
        operation = info.get("operation", "access")
        fs_type = info.get("fs_type", "filesystem")
        
        return ActorCandidate(
            name=f"FileSystem{operation.title()}",
            description=f"File system {operation} operations via {fs_type}",
            actor_type="FileSystem",
            role=ActorRole.RECEIVER,
            boundary_type=BoundaryType.FILESYSTEM,
            source_functions=[boundary_call.source_function],
            metadata={
                "Operation": operation,
                "Type": fs_type
            }
        )


class BoundaryDetectionEngine:
    """Main engine that orchestrates boundary detection and ACTOR generation"""
    
    def __init__(self):
        self.detectors = [
            HttpBoundaryDetector(),
            DatabaseBoundaryDetector(),
            FileSystemBoundaryDetector(),
        ]
        self.detected_boundaries = []
        self.actor_candidates = []
        self.validated_actors = []
    
    def analyze_boundaries(self, ast_tree: ast.AST, source_code: str, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Main entry point for boundary analysis
        Returns: (actors, flows)
        """
        # Step 1: Detect all boundary crossings
        self.detected_boundaries = self._detect_all_boundaries(ast_tree, source_code)
        
        # Step 2: Create ACTOR candidates
        self.actor_candidates = self._create_actor_candidates()
        
        # Step 3: Validate and consolidate ACTORs
        self.validated_actors = self._validate_and_consolidate_actors()
        
        # Step 4: Generate flows following Trigger → Functions → Receiver
        flows = self._generate_bidirectional_flows()
        
        # Step 5: Convert to output format
        actors_output = self._format_actors_for_output()
        flows_output = self._format_flows_for_output(flows)
        
        return actors_output, flows_output
    
    def _detect_all_boundaries(self, ast_tree: ast.AST, source_code: str) -> List[BoundaryCall]:
        """Run all boundary detectors"""
        all_boundaries = []
        
        for detector in self.detectors:
            boundaries = detector.detect_boundaries(ast_tree, source_code)
            all_boundaries.extend(boundaries)
        
        return all_boundaries
    
    def _create_actor_candidates(self) -> List[ActorCandidate]:
        """Create ACTOR candidates from detected boundaries"""
        candidates = []
        
        for boundary in self.detected_boundaries:
            detector = self._get_detector_for_boundary(boundary.call_type)
            if detector:
                candidate = detector.create_actor_from_boundary(boundary)
                candidates.append(candidate)
        
        return candidates
    
    def _get_detector_for_boundary(self, boundary_type: BoundaryType) -> Optional[BaseBoundaryDetector]:
        """Get appropriate detector for boundary type"""
        detector_map = {
            BoundaryType.HTTP_INBOUND: HttpBoundaryDetector,
            BoundaryType.HTTP_OUTBOUND: HttpBoundaryDetector,
            BoundaryType.DATABASE: DatabaseBoundaryDetector,
            BoundaryType.FILESYSTEM: FileSystemBoundaryDetector,
        }
        
        detector_class = detector_map.get(boundary_type)
        if detector_class:
            # Find existing detector instance
            for detector in self.detectors:
                if isinstance(detector, detector_class):
                    return detector
        
        return None
    
    def _validate_and_consolidate_actors(self) -> List[ActorCandidate]:
        """Validate ACTORs and consolidate duplicates"""
        validated = []
        seen_actors = {}
        
        for candidate in self.actor_candidates:
            # Create unique key for deduplication
            key = f"{candidate.actor_type}_{candidate.name}_{candidate.role.value}"
            
            if key in seen_actors:
                # Merge with existing actor
                existing = seen_actors[key]
                existing.source_functions.extend(candidate.source_functions)
                existing.source_functions = list(set(existing.source_functions))  # Remove duplicates
            else:
                # New actor
                seen_actors[key] = candidate
                validated.append(candidate)
        
        return validated
    
    def _generate_bidirectional_flows(self) -> List[Dict]:
        """Generate flows following Trigger → Functions → Receiver pattern"""
        flows = []
        
        for actor in self.validated_actors:
            for source_function in actor.source_functions:
                if actor.role == ActorRole.TRIGGER:
                    # Trigger → Function flow
                    flows.append({
                        "uuid": str(uuid.uuid4()),
                        "type": "flow",
                        "source": actor.name,  # Will be replaced with UUID later
                        "target": source_function,
                        "FlowDescr": f"{actor.name} triggers {source_function} execution",
                        "FlowDef": f"external_trigger → {source_function}()",
                        "Confidence": 1.0,
                        "source_name": actor.name,
                        "target_name": source_function,
                        "flow_direction": "inbound"
                    })
                    
                elif actor.role == ActorRole.RECEIVER:
                    # Function → Receiver flow
                    flows.append({
                        "uuid": str(uuid.uuid4()),
                        "type": "flow",
                        "source": source_function,
                        "target": actor.name,  # Will be replaced with UUID later
                        "FlowDescr": f"{source_function} sends data to {actor.name}",
                        "FlowDef": f"{source_function}() → external_receiver",
                        "Confidence": 1.0,
                        "source_name": source_function,
                        "target_name": actor.name,
                        "flow_direction": "outbound"
                    })
        
        return flows
    
    def _format_actors_for_output(self) -> List[Dict]:
        """Format ACTORs for JSON output"""
        formatted_actors = []
        
        for actor in self.validated_actors:
            actor_dict = {
                "uuid": str(uuid.uuid4()),
                "type": "ACTOR",
                "Name": actor.name,
                "Descr": actor.description,
                "ActorType": actor.actor_type,
                "Role": actor.role.value,
                "BoundaryType": actor.boundary_type.value,
                "SourceFunctions": actor.source_functions
            }
            
            # Add type-specific metadata
            actor_dict.update(actor.metadata)
            
            formatted_actors.append(actor_dict)
        
        return formatted_actors
    
    def _format_flows_for_output(self, flows: List[Dict]) -> List[Dict]:
        """Format flows for JSON output with proper UUID references"""
        # Create UUID mapping for actors
        actor_uuid_map = {}
        for actor_dict in self._format_actors_for_output():
            actor_uuid_map[actor_dict["Name"]] = actor_dict["uuid"]
        
        formatted_flows = []
        for flow in flows:
            # Replace names with UUIDs where possible
            if flow["source"] in actor_uuid_map:
                flow["source"] = actor_uuid_map[flow["source"]]
            if flow["target"] in actor_uuid_map:
                flow["target"] = actor_uuid_map[flow["target"]]
            
            formatted_flows.append(flow)
        
        return formatted_flows
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get statistics about boundary detection"""
        stats = {
            "total_boundaries_detected": len(self.detected_boundaries),
            "actor_candidates_created": len(self.actor_candidates),
            "validated_actors": len(self.validated_actors),
            "boundary_types": {},
            "actor_roles": {"trigger": 0, "receiver": 0}
        }
        
        # Count by boundary type
        for boundary in self.detected_boundaries:
            boundary_type = boundary.call_type.value
            stats["boundary_types"][boundary_type] = stats["boundary_types"].get(boundary_type, 0) + 1
        
        # Count by actor role
        for actor in self.validated_actors:
            stats["actor_roles"][actor.role.value] += 1
        
        return stats


def create_boundary_detector() -> BoundaryDetectionEngine:
    """Factory function to create configured boundary detection engine"""
    return BoundaryDetectionEngine()


# Example usage and testing
if __name__ == "__main__":
    # Example code for testing
    test_code = '''
import requests
from fastapi import FastAPI
import sqlite3

app = FastAPI()

@app.post("/users")
async def create_user(user_data: dict):
    # Outbound HTTP call (Receiver)
    response = requests.post("https://api.external.com/validate", json=user_data)
    
    # Database operation (Receiver)
    connection = sqlite3.connect("users.db")
    cursor = connection.cursor()
    cursor.execute("INSERT INTO users VALUES (?)", (user_data["name"],))
    connection.commit()
    
    return {"status": "created"}

def process_file():
    # File system operation (Receiver)
    with open("data.json", "r") as f:
        data = json.load(f)
    return data
'''
    
    # Parse and analyze
    ast_tree = ast.parse(test_code)
    engine = create_boundary_detector()
    actors, flows = engine.analyze_boundaries(ast_tree, test_code, "test.py")
    
    print("=== DETECTED ACTORS ===")
    for actor in actors:
        print(f"- {actor['Name']}: {actor['Descr']} ({actor['Role']})")
    
    print("\n=== GENERATED FLOWS ===")
    for flow in flows:
        print(f"- {flow['source_name']} → {flow['target_name']}: {flow['FlowDescr']}")
    
    print(f"\n=== STATISTICS ===")
    stats = engine.get_detection_statistics()
    for key, value in stats.items():
        print(f"- {key}: {value}")