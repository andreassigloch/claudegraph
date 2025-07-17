#!/usr/bin/env python3
"""
Logical Flow Detector for Code Architecture Analyzer

Transforms physical dependency detection into abstract logical dependency analysis:
- Identifies business domains and logical services
- Detects data transformation patterns
- Maps technical components to logical capabilities
- Recognizes architectural patterns and design principles
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class LogicalDependencyType(Enum):
    """Types of logical dependencies"""
    DOMAIN_SERVICE = "domain_service"          # Business domain services
    DATA_TRANSFORMER = "data_transformer"      # Data processing/transformation
    BUSINESS_ENTITY = "business_entity"        # Domain entities/models
    INTEGRATION_POINT = "integration_point"    # External system integrations
    WORKFLOW_ORCHESTRATOR = "workflow_orchestrator"  # Business process coordination
    VALIDATION_SERVICE = "validation_service"  # Business rule validation
    PERSISTENCE_LAYER = "persistence_layer"    # Data persistence abstraction
    PRESENTATION_LAYER = "presentation_layer"  # User interaction abstraction


@dataclass
class LogicalActor:
    """Represents an abstract logical actor in the system"""
    name: str
    logical_type: LogicalDependencyType
    business_domain: str
    responsibilities: List[str]
    data_entities: List[str]
    business_operations: List[str]
    abstraction_level: str  # "domain", "service", "component", "utility"
    confidence: float = 1.0
    evidence: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = {}


@dataclass
class LogicalFlow:
    """Represents logical data/control flow between business components"""
    source_actor: str
    target_actor: str
    flow_type: str  # "data_flow", "control_flow", "event_flow", "business_process"
    data_transformation: str
    business_meaning: str
    quality_attributes: List[str]  # ["consistency", "availability", "security"]
    confidence: float = 1.0


@dataclass
class BusinessDomain:
    """Represents a business domain with its components"""
    name: str
    description: str
    primary_entities: List[str]
    business_capabilities: List[str]
    services: List[str]
    boundaries: Dict[str, str]  # Bounded context information


class LogicalFlowDetector:
    """Detects logical dependencies and business flows"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.semantic_analyzer = SemanticAnalyzer()
        self.domain_classifier = DomainClassifier()
        self.pattern_recognizer = ArchitecturalPatternRecognizer()
        
        # Initialize domain patterns
        self.business_domains = self._initialize_business_domains()
        
    def analyze_logical_dependencies(self, ast_result, function_flows: List) -> Tuple[List[LogicalActor], List[LogicalFlow]]:
        """Main method to analyze logical dependencies"""
        logger.info("Starting logical dependency analysis...")
        
        # Step 1: Semantic analysis to identify business concepts
        semantic_info = self.semantic_analyzer.analyze_semantic_content(ast_result)
        
        # Step 2: Domain classification
        domain_mappings = self.domain_classifier.classify_functions_to_domains(
            ast_result.functions, semantic_info
        )
        
        # Step 3: Identify logical actors
        logical_actors = self._identify_logical_actors(ast_result, domain_mappings, semantic_info)
        
        # Step 4: Analyze logical flows
        logical_flows = self._analyze_logical_flows(logical_actors, function_flows, semantic_info)
        
        # Step 5: Enhance with architectural patterns
        logical_actors, logical_flows = self.pattern_recognizer.enhance_with_patterns(
            logical_actors, logical_flows, ast_result
        )
        
        logger.info(f"Identified {len(logical_actors)} logical actors and {len(logical_flows)} logical flows")
        return logical_actors, logical_flows
    
    def _identify_logical_actors(self, ast_result, domain_mappings: Dict, semantic_info: Dict) -> List[LogicalActor]:
        """Identify logical actors from semantic analysis"""
        logical_actors = []
        
        # Group functions by logical domains
        domain_functions = {}
        for func in ast_result.functions:
            domain = domain_mappings.get(func.name, "utility")
            if domain not in domain_functions:
                domain_functions[domain] = []
            domain_functions[domain].append(func)
        
        # Create logical actors for each domain
        for domain, functions in domain_functions.items():
            if len(functions) < 2:  # Skip single-function domains
                continue
                
            # Analyze the business capabilities of this domain
            capabilities = self._extract_business_capabilities(functions, semantic_info)
            entities = self._extract_data_entities(functions, semantic_info)
            operations = self._extract_business_operations(functions, semantic_info)
            
            # Determine logical type based on analysis
            logical_type = self._determine_logical_type(functions, capabilities, entities)
            
            # Create logical actor
            actor = LogicalActor(
                name=f"{domain.replace('_', ' ').title()}Service",
                logical_type=logical_type,
                business_domain=domain,
                responsibilities=capabilities,
                data_entities=entities,
                business_operations=operations,
                abstraction_level=self._determine_abstraction_level(functions, capabilities),
                confidence=self._calculate_confidence(functions, capabilities),
                evidence={
                    "function_count": len(functions),
                    "function_names": [f.name for f in functions],
                    "primary_patterns": self._extract_primary_patterns(functions)
                }
            )
            logical_actors.append(actor)
        
        return logical_actors
    
    def _analyze_logical_flows(self, logical_actors: List[LogicalActor], 
                             function_flows: List, semantic_info: Dict) -> List[LogicalFlow]:
        """Analyze logical flows between actors"""
        logical_flows = []
        
        # Create actor lookup by business domain
        domain_to_actor = {actor.business_domain: actor for actor in logical_actors}
        
        # Map function flows to logical flows
        for flow in function_flows:
            source_domain = self._get_function_domain(flow.source_name, semantic_info)
            target_domain = self._get_function_domain(flow.target_name, semantic_info)
            
            if source_domain in domain_to_actor and target_domain in domain_to_actor:
                source_actor = domain_to_actor[source_domain]
                target_actor = domain_to_actor[target_domain]
                
                # Skip internal flows within the same domain
                if source_actor == target_actor:
                    continue
                
                # Analyze the data transformation
                transformation = self._analyze_data_transformation(flow, semantic_info)
                business_meaning = self._extract_business_meaning(flow, source_actor, target_actor)
                quality_attributes = self._identify_quality_attributes(flow)
                
                logical_flow = LogicalFlow(
                    source_actor=source_actor.name,
                    target_actor=target_actor.name,
                    flow_type=self._determine_flow_type(flow, transformation),
                    data_transformation=transformation,
                    business_meaning=business_meaning,
                    quality_attributes=quality_attributes,
                    confidence=self._calculate_flow_confidence(flow, transformation)
                )
                logical_flows.append(logical_flow)
        
        return logical_flows
    
    def _extract_business_capabilities(self, functions: List, semantic_info: Dict) -> List[str]:
        """Extract business capabilities from function analysis"""
        capabilities = set()
        
        for func in functions:
            # Analyze function name for business verbs
            name_parts = re.findall(r'[A-Z][a-z]+|[a-z]+', func.name)
            
            for part in name_parts:
                if part.lower() in ['create', 'make', 'build', 'generate']:
                    capabilities.add("creation")
                elif part.lower() in ['update', 'modify', 'change', 'edit']:
                    capabilities.add("modification")
                elif part.lower() in ['delete', 'remove', 'destroy']:
                    capabilities.add("deletion")
                elif part.lower() in ['get', 'find', 'search', 'query', 'retrieve']:
                    capabilities.add("retrieval")
                elif part.lower() in ['validate', 'verify', 'check']:
                    capabilities.add("validation")
                elif part.lower() in ['process', 'handle', 'manage']:
                    capabilities.add("processing")
                elif part.lower() in ['notify', 'alert', 'send']:
                    capabilities.add("notification")
                elif part.lower() in ['calculate', 'compute', 'analyze']:
                    capabilities.add("computation")
        
        return list(capabilities)
    
    def _extract_data_entities(self, functions: List, semantic_info: Dict) -> List[str]:
        """Extract data entities from function analysis"""
        entities = set()
        
        for func in functions:
            # Extract from function name
            name_parts = re.findall(r'[A-Z][a-z]+|[a-z]+', func.name)
            
            # Look for nouns that represent business entities
            for part in name_parts:
                if part.lower() in ['user', 'customer', 'client', 'account']:
                    entities.add("User")
                elif part.lower() in ['product', 'item', 'goods']:
                    entities.add("Product")
                elif part.lower() in ['order', 'purchase', 'transaction']:
                    entities.add("Order")
                elif part.lower() in ['payment', 'billing', 'invoice']:
                    entities.add("Payment")
                elif part.lower() in ['inventory', 'stock', 'warehouse']:
                    entities.add("Inventory")
                elif part.lower() in ['report', 'analytics', 'metrics']:
                    entities.add("Report")
                elif part.lower() in ['message', 'notification', 'alert']:
                    entities.add("Message")
                elif part.lower() in ['config', 'settings', 'preferences']:
                    entities.add("Configuration")
            
            # Extract from parameter types and return types
            for param in func.args:
                if param.lower() in ['user_id', 'customer_id', 'client_id']:
                    entities.add("User")
                elif param.lower() in ['product_id', 'item_id']:
                    entities.add("Product")
                elif param.lower() in ['order_id', 'transaction_id']:
                    entities.add("Order")
        
        return list(entities)
    
    def _extract_business_operations(self, functions: List, semantic_info: Dict) -> List[str]:
        """Extract business operations from function analysis"""
        operations = []
        
        for func in functions:
            # Convert function name to business operation
            name_clean = re.sub(r'[_-]', ' ', func.name)
            name_clean = re.sub(r'([A-Z])', r' \1', name_clean).strip()
            
            # Clean up common technical prefixes
            name_clean = re.sub(r'^(get|set|create|update|delete|find|search|query|fetch|load|save|handle|process|manage|execute|run|perform|do)\s*', '', name_clean, flags=re.IGNORECASE)
            
            if name_clean:
                operations.append(name_clean.title())
        
        return operations
    
    def _determine_logical_type(self, functions: List, capabilities: List, entities: List) -> LogicalDependencyType:
        """Determine the logical type of an actor based on its characteristics"""
        
        # Domain service: multiple capabilities, multiple entities
        if len(capabilities) >= 3 and len(entities) >= 2:
            return LogicalDependencyType.DOMAIN_SERVICE
        
        # Data transformer: processing/computation capabilities
        if "processing" in capabilities or "computation" in capabilities:
            return LogicalDependencyType.DATA_TRANSFORMER
        
        # Business entity: single entity focus with CRUD operations
        if len(entities) == 1 and any(cap in capabilities for cap in ["creation", "modification", "deletion", "retrieval"]):
            return LogicalDependencyType.BUSINESS_ENTITY
        
        # Validation service: validation capabilities
        if "validation" in capabilities:
            return LogicalDependencyType.VALIDATION_SERVICE
        
        # Workflow orchestrator: multiple business operations
        if len(functions) >= 5 and "processing" in capabilities:
            return LogicalDependencyType.WORKFLOW_ORCHESTRATOR
        
        # Default to domain service
        return LogicalDependencyType.DOMAIN_SERVICE
    
    def _determine_abstraction_level(self, functions: List, capabilities: List) -> str:
        """Determine the abstraction level of the component"""
        
        if len(capabilities) >= 3:
            return "domain"
        elif len(capabilities) >= 2:
            return "service"
        elif len(functions) >= 3:
            return "component"
        else:
            return "utility"
    
    def _calculate_confidence(self, functions: List, capabilities: List) -> float:
        """Calculate confidence score for logical actor identification"""
        base_confidence = 0.5
        
        # More functions = higher confidence
        function_boost = min(len(functions) * 0.1, 0.3)
        
        # More capabilities = higher confidence
        capability_boost = min(len(capabilities) * 0.1, 0.2)
        
        return min(base_confidence + function_boost + capability_boost, 1.0)
    
    def _extract_primary_patterns(self, functions: List) -> List[str]:
        """Extract primary patterns from function analysis"""
        patterns = []
        
        function_names = [f.name.lower() for f in functions]
        
        # CRUD pattern
        if any(name.startswith(('create', 'add', 'insert')) for name in function_names):
            patterns.append("CRUD")
        
        # Service pattern
        if any(name.endswith(('service', 'handler', 'processor')) for name in function_names):
            patterns.append("Service")
        
        # Repository pattern
        if any(name.startswith(('get', 'find', 'query', 'search')) for name in function_names):
            patterns.append("Repository")
        
        # Factory pattern
        if any(name.startswith(('create', 'build', 'make')) for name in function_names):
            patterns.append("Factory")
        
        return patterns
    
    def _get_function_domain(self, function_name: str, semantic_info: Dict) -> str:
        """Get the business domain for a function"""
        # This would use the domain classifier
        # For now, simple heuristic based on function name
        name_lower = function_name.lower()
        
        if any(term in name_lower for term in ['user', 'customer', 'account', 'auth']):
            return "user_management"
        elif any(term in name_lower for term in ['product', 'item', 'catalog']):
            return "product_management"
        elif any(term in name_lower for term in ['order', 'purchase', 'cart']):
            return "order_management"
        elif any(term in name_lower for term in ['payment', 'billing', 'invoice']):
            return "payment_processing"
        elif any(term in name_lower for term in ['inventory', 'stock', 'warehouse']):
            return "inventory_management"
        elif any(term in name_lower for term in ['report', 'analytics', 'metrics']):
            return "reporting"
        elif any(term in name_lower for term in ['notification', 'email', 'message']):
            return "notification"
        else:
            return "utility"
    
    def _analyze_data_transformation(self, flow, semantic_info: Dict) -> str:
        """Analyze what data transformation occurs in the flow"""
        # Safely get source and target names as strings
        source_name = str(getattr(flow, 'source_name', '')).lower()
        target_name = str(getattr(flow, 'target_name', '')).lower()
        
        # Simple heuristic based on function names
        if 'validate' in source_name and 'process' in target_name:
            return "validation_to_processing"
        elif 'get' in source_name and 'create' in target_name:
            return "retrieval_to_creation"
        elif 'parse' in source_name and 'store' in target_name:
            return "parsing_to_storage"
        elif 'calculate' in source_name and 'format' in target_name:
            return "computation_to_formatting"
        else:
            return "data_transfer"
    
    def _extract_business_meaning(self, flow, source_actor: LogicalActor, target_actor: LogicalActor) -> str:
        """Extract business meaning from the flow"""
        return f"{source_actor.business_domain} provides data to {target_actor.business_domain}"
    
    def _identify_quality_attributes(self, flow) -> List[str]:
        """Identify quality attributes from flow analysis"""
        attributes = []
        
        # Simple heuristics
        if hasattr(flow, 'flow_def') and flow.flow_def:
            flow_def_lower = str(flow.flow_def).lower()
            
            if 'async' in flow_def_lower or 'await' in flow_def_lower:
                attributes.append("asynchronous")
            if 'validate' in flow_def_lower:
                attributes.append("validated")
            if 'cache' in flow_def_lower:
                attributes.append("cached")
            if 'retry' in flow_def_lower:
                attributes.append("resilient")
        
        return attributes
    
    def _determine_flow_type(self, flow, transformation: str) -> str:
        """Determine the type of logical flow"""
        if transformation.endswith("_to_processing"):
            return "control_flow"
        elif transformation.endswith("_to_storage"):
            return "data_flow"
        elif transformation.startswith("validation_"):
            return "validation_flow"
        else:
            return "data_flow"
    
    def _calculate_flow_confidence(self, flow, transformation: str) -> float:
        """Calculate confidence for logical flow identification"""
        base_confidence = 0.7
        
        if transformation != "data_transfer":
            base_confidence += 0.2
        
        if hasattr(flow, 'confidence'):
            base_confidence = (base_confidence + flow.confidence) / 2
        
        return min(base_confidence, 1.0)
    
    def _initialize_business_domains(self) -> Dict[str, BusinessDomain]:
        """Initialize common business domains"""
        domains = {
            "user_management": BusinessDomain(
                name="User Management",
                description="Handles user accounts, authentication, and authorization",
                primary_entities=["User", "Account", "Role", "Permission"],
                business_capabilities=["user_registration", "authentication", "authorization", "profile_management"],
                services=["UserService", "AuthService", "ProfileService"],
                boundaries={"input": "user_requests", "output": "user_data"}
            ),
            "product_management": BusinessDomain(
                name="Product Management",
                description="Manages product catalog and inventory",
                primary_entities=["Product", "Category", "Inventory", "Pricing"],
                business_capabilities=["product_creation", "catalog_management", "inventory_tracking", "pricing"],
                services=["ProductService", "CatalogService", "InventoryService"],
                boundaries={"input": "product_data", "output": "product_information"}
            ),
            "order_management": BusinessDomain(
                name="Order Management",
                description="Handles order processing and fulfillment",
                primary_entities=["Order", "OrderItem", "Cart", "Fulfillment"],
                business_capabilities=["order_creation", "order_processing", "fulfillment", "order_tracking"],
                services=["OrderService", "CartService", "FulfillmentService"],
                boundaries={"input": "order_requests", "output": "order_status"}
            )
        }
        return domains


class SemanticAnalyzer:
    """Analyzes semantic content of code to identify business concepts"""
    
    def analyze_semantic_content(self, ast_result) -> Dict[str, Any]:
        """Analyze semantic content from AST results"""
        semantic_info = {
            "business_terms": self._extract_business_terms(ast_result),
            "data_patterns": self._identify_data_patterns(ast_result),
            "interaction_patterns": self._analyze_interaction_patterns(ast_result)
        }
        return semantic_info
    
    def _extract_business_terms(self, ast_result) -> List[str]:
        """Extract business terms from code"""
        terms = set()
        
        for func in ast_result.functions:
            # Extract from function names
            name_parts = re.findall(r'[A-Z][a-z]+|[a-z]+', func.name)
            for part in name_parts:
                if len(part) > 3:  # Skip very short terms
                    terms.add(part.lower())
            
            # Extract from docstrings
            if func.docstring:
                words = re.findall(r'\b[a-zA-Z]{4,}\b', func.docstring)
                for word in words:
                    terms.add(word.lower())
        
        return list(terms)
    
    def _identify_data_patterns(self, ast_result) -> Dict[str, List[str]]:
        """Identify data patterns in the code"""
        patterns = {
            "crud_operations": [],
            "data_transformations": [],
            "validation_patterns": []
        }
        
        for func in ast_result.functions:
            name_lower = func.name.lower()
            
            # CRUD operations
            if any(op in name_lower for op in ['create', 'read', 'update', 'delete', 'get', 'set']):
                patterns["crud_operations"].append(func.name)
            
            # Data transformations
            if any(transform in name_lower for transform in ['convert', 'transform', 'parse', 'format', 'serialize']):
                patterns["data_transformations"].append(func.name)
            
            # Validation patterns
            if any(validate in name_lower for validate in ['validate', 'verify', 'check', 'ensure']):
                patterns["validation_patterns"].append(func.name)
        
        return patterns
    
    def _analyze_interaction_patterns(self, ast_result) -> Dict[str, List[str]]:
        """Analyze interaction patterns between components"""
        patterns = {
            "service_calls": [],
            "data_flows": [],
            "event_patterns": []
        }
        
        for func in ast_result.functions:
            for call in func.calls:
                call_str = str(call).lower()
                
                # Service calls
                if any(service in call_str for service in ['service', 'client', 'api', 'handler']):
                    patterns["service_calls"].append(call_str)
                
                # Data flows
                if any(data in call_str for data in ['process', 'transform', 'convert', 'parse']):
                    patterns["data_flows"].append(call_str)
                
                # Event patterns
                if any(event in call_str for event in ['emit', 'trigger', 'publish', 'subscribe']):
                    patterns["event_patterns"].append(call_str)
        
        return patterns


class DomainClassifier:
    """Classifies functions into business domains"""
    
    def classify_functions_to_domains(self, functions: List, semantic_info: Dict) -> Dict[str, str]:
        """Classify functions into business domains"""
        domain_mappings = {}
        
        for func in functions:
            domain = self._classify_single_function(func, semantic_info)
            domain_mappings[func.name] = domain
        
        return domain_mappings
    
    def _classify_single_function(self, func, semantic_info: Dict) -> str:
        """Classify a single function into a business domain"""
        name_lower = func.name.lower()
        
        # User/Account domain
        if any(term in name_lower for term in ['user', 'account', 'auth', 'login', 'register', 'profile']):
            return "user_management"
        
        # Product domain
        elif any(term in name_lower for term in ['product', 'item', 'catalog', 'inventory', 'stock']):
            return "product_management"
        
        # Order domain
        elif any(term in name_lower for term in ['order', 'cart', 'purchase', 'checkout', 'fulfillment']):
            return "order_management"
        
        # Payment domain
        elif any(term in name_lower for term in ['payment', 'billing', 'invoice', 'transaction', 'charge']):
            return "payment_processing"
        
        # Notification domain
        elif any(term in name_lower for term in ['notify', 'email', 'message', 'alert', 'notification']):
            return "notification"
        
        # Reporting domain
        elif any(term in name_lower for term in ['report', 'analytics', 'metrics', 'dashboard', 'stats']):
            return "reporting"
        
        # Configuration domain
        elif any(term in name_lower for term in ['config', 'settings', 'preferences', 'option']):
            return "configuration"
        
        # Default to utility
        else:
            return "utility"


class ArchitecturalPatternRecognizer:
    """Recognizes architectural patterns in the codebase"""
    
    def enhance_with_patterns(self, logical_actors: List[LogicalActor], logical_flows: List[LogicalFlow], 
                            ast_result) -> Tuple[List[LogicalActor], List[LogicalFlow]]:
        """Enhance logical actors and flows with architectural pattern recognition"""
        
        # Detect architectural patterns
        patterns = self._detect_architectural_patterns(ast_result)
        
        # Enhance actors with pattern information
        enhanced_actors = self._enhance_actors_with_patterns(logical_actors, patterns)
        
        # Enhance flows with pattern information
        enhanced_flows = self._enhance_flows_with_patterns(logical_flows, patterns)
        
        return enhanced_actors, enhanced_flows
    
    def _detect_architectural_patterns(self, ast_result) -> Dict[str, List[str]]:
        """Detect architectural patterns in the code"""
        patterns = {
            "mvc": [],
            "repository": [],
            "factory": [],
            "observer": [],
            "strategy": [],
            "decorator": []
        }
        
        for func in ast_result.functions:
            name_lower = func.name.lower()
            
            # MVC pattern
            if any(mvc in name_lower for mvc in ['controller', 'view', 'model']):
                patterns["mvc"].append(func.name)
            
            # Repository pattern
            if any(repo in name_lower for repo in ['repository', 'dao', 'data_access']):
                patterns["repository"].append(func.name)
            
            # Factory pattern
            if any(factory in name_lower for factory in ['factory', 'builder', 'creator']):
                patterns["factory"].append(func.name)
            
            # Observer pattern
            if any(observer in name_lower for observer in ['observer', 'listener', 'subscriber']):
                patterns["observer"].append(func.name)
            
            # Strategy pattern
            if any(strategy in name_lower for strategy in ['strategy', 'algorithm', 'policy']):
                patterns["strategy"].append(func.name)
            
            # Decorator pattern
            if func.decorators:
                patterns["decorator"].append(func.name)
        
        return patterns
    
    def _enhance_actors_with_patterns(self, logical_actors: List[LogicalActor], 
                                    patterns: Dict[str, List[str]]) -> List[LogicalActor]:
        """Enhance logical actors with architectural pattern information"""
        enhanced_actors = []
        
        for actor in logical_actors:
            enhanced_actor = LogicalActor(
                name=actor.name,
                logical_type=actor.logical_type,
                business_domain=actor.business_domain,
                responsibilities=actor.responsibilities,
                data_entities=actor.data_entities,
                business_operations=actor.business_operations,
                abstraction_level=actor.abstraction_level,
                confidence=actor.confidence,
                evidence=actor.evidence.copy()
            )
            
            # Add architectural patterns
            actor_patterns = []
            if actor.evidence and "function_names" in actor.evidence:
                for func_name in actor.evidence["function_names"]:
                    for pattern_name, pattern_functions in patterns.items():
                        if func_name in pattern_functions:
                            actor_patterns.append(pattern_name)
            
            enhanced_actor.evidence["architectural_patterns"] = list(set(actor_patterns))
            enhanced_actors.append(enhanced_actor)
        
        return enhanced_actors
    
    def _enhance_flows_with_patterns(self, logical_flows: List[LogicalFlow], 
                                   patterns: Dict[str, List[str]]) -> List[LogicalFlow]:
        """Enhance logical flows with architectural pattern information"""
        # For now, return flows as-is
        # Could be enhanced to detect pattern-specific flow types
        return logical_flows