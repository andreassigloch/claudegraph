#!/usr/bin/env python3
"""
Simple LLM Call Bundling for Code Architecture Analyzer

Simple approach to reduce LLM costs:
1. Bundle multiple actor enhancement requests into single calls
2. Buffer requests to batch process
3. No complex cost control - just efficient batching
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class EnhancementRequest:
    """Single actor enhancement request"""
    actor_name: str
    actor_type: str = ""
    code_context: str = ""
    function_context: str = ""
    context: Dict[str, Any] = None
    enhancement_type: str = "physical_actor"  # "physical_actor" or "logical_actor"
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class EnhancementResult:
    """Result of actor enhancement"""
    original_name: str
    enhanced_name: str
    description: str
    confidence: float = 0.0
    business_domain: str = ""


class SimpleLLMBundler:
    """Simple bundling service for LLM calls"""
    
    def __init__(self, llm_client, batch_size: int = 5, max_wait_time: float = 2.0):
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.last_batch_time = time.time()
    
    def enhance_actors_bundled(self, requests: List[EnhancementRequest]) -> List[EnhancementResult]:
        """Enhance multiple actors with bundled LLM calls"""
        
        if not requests:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
            
            # Small delay between batches to respect rate limits
            if i + self.batch_size < len(requests):
                time.sleep(0.1)
        
        logger.info(f"Enhanced {len(requests)} actors in {len(results) // self.batch_size + 1} batches")
        return results
    
    def enhance_logical_actors(self, requests: List[EnhancementRequest]) -> List[EnhancementResult]:
        """Enhance logical actors with business-focused prompts"""
        
        if not requests:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            batch_results = self._process_logical_batch(batch)
            results.extend(batch_results)
            
            # Small delay between batches to respect rate limits
            if i + self.batch_size < len(requests):
                time.sleep(0.1)
        
        logger.info(f"Enhanced {len(requests)} logical actors in {len(results) // self.batch_size + 1} batches")
        return results
    
    def _process_batch(self, batch: List[EnhancementRequest]) -> List[EnhancementResult]:
        """Process a batch of enhancement requests in single LLM call"""
        
        if len(batch) == 1:
            return [self._enhance_single(batch[0])]
        
        # Bundle multiple requests into single prompt
        prompt = self._create_batch_prompt(batch)
        
        try:
            response = self.llm_client.query(prompt)
            return self._parse_batch_response(response, batch)
        except Exception as e:
            logger.warning(f"Batch processing failed: {e}. Falling back to individual calls.")
            return [self._enhance_single(req) for req in batch]
    
    def _process_logical_batch(self, batch: List[EnhancementRequest]) -> List[EnhancementResult]:
        """Process a batch of logical actor enhancement requests"""
        
        if len(batch) == 1:
            return [self._enhance_logical_single(batch[0])]
        
        # Bundle multiple logical actor requests into single prompt
        prompt = self._create_logical_batch_prompt(batch)
        
        try:
            response = self.llm_client.query(prompt)
            return self._parse_logical_batch_response(response, batch)
        except Exception as e:
            logger.warning(f"Logical batch processing failed: {e}. Falling back to individual calls.")
            return [self._enhance_logical_single(req) for req in batch]
    
    def _create_batch_prompt(self, batch: List[EnhancementRequest]) -> str:
        """Create single prompt for multiple actor enhancements"""
        
        prompt = """Analyze these code actors and provide specific business names and descriptions.
For each actor, respond with: NAME|DESCRIPTION

Examples:
- HTTP call to api.stripe.com → StripePaymentAPI|Stripe payment processing service
- Database with 'user' table → UserDatabase|User data storage and management
- File operations with .json → ConfigurationFiles|Application configuration management

Actors to analyze:
"""
        
        for i, req in enumerate(batch, 1):
            prompt += f"\n{i}. Type: {req.actor_type}"
            prompt += f"\n   Code: {req.code_context}"
            if req.function_context:
                prompt += f"\n   Context: {req.function_context}"
            prompt += "\n"
        
        prompt += "\nRespond with numbered list (1. NAME|DESCRIPTION, 2. NAME|DESCRIPTION, etc.):"
        
        return prompt
    
    def _parse_batch_response(self, response: str, batch: List[EnhancementRequest]) -> List[EnhancementResult]:
        """Parse LLM response for batch enhancement"""
        
        results = []
        lines = response.strip().split('\n')
        
        for i, req in enumerate(batch):
            enhanced_name = req.actor_name  # Default to original
            description = f"{req.actor_type} interface"  # Default description
            
            # Look for matching line in response
            for line in lines:
                if line.strip().startswith(f"{i+1}."):
                    if '|' in line:
                        parts = line.split('|', 1)
                        name_part = parts[0].split('.', 1)[1].strip()  # Remove number
                        desc_part = parts[1].strip()
                        
                        enhanced_name = name_part
                        description = desc_part
                    break
            
            results.append(EnhancementResult(
                original_name=req.actor_name,
                enhanced_name=enhanced_name,
                description=description,
                confidence=0.8  # Assume good confidence for successful parsing
            ))
        
        return results
    
    def _create_logical_batch_prompt(self, batch: List[EnhancementRequest]) -> str:
        """Create single prompt for multiple logical actor enhancements"""
        
        prompt = """Analyze these business components and provide better business names and domains.
For each component, respond with: NAME|DOMAIN|DESCRIPTION

Examples:
- UserService in user_management → UserAccountService|User Management|User registration and account management
- ProductService in product_management → ProductCatalogService|Product Management|Product catalog and inventory management
- OrderService in order_management → OrderProcessingService|Order Management|Order processing and fulfillment

Components to analyze:
"""
        
        for i, req in enumerate(batch, 1):
            context = req.context
            prompt += f"\n{i}. Component: {req.actor_name}"
            prompt += f"   Domain: {context.get('business_domain', 'unknown')}"
            prompt += f"   Type: {context.get('logical_type', 'unknown')}"
            prompt += f"   Responsibilities: {', '.join(context.get('responsibilities', []))}"
            prompt += f"   Entities: {', '.join(context.get('data_entities', []))}"
        
        prompt += "\n\nProvide improved business names for each component:"
        return prompt
    
    def _parse_logical_batch_response(self, response: str, batch: List[EnhancementRequest]) -> List[EnhancementResult]:
        """Parse response from logical batch enhancement"""
        
        results = []
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        for i, req in enumerate(batch):
            if i < len(lines):
                line = lines[i]
                
                # Remove number prefix if present
                if line.startswith(f"{i+1}."):
                    line = line[len(f"{i+1}."):].strip()
                
                # Parse NAME|DOMAIN|DESCRIPTION format
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        name = parts[0].strip()
                        domain = parts[1].strip()
                        desc = parts[2].strip()
                    elif len(parts) == 2:
                        name = parts[0].strip()
                        domain = req.context.get('business_domain', 'unknown')
                        desc = parts[1].strip()
                    else:
                        name = parts[0].strip()
                        domain = req.context.get('business_domain', 'unknown')
                        desc = f"{req.context.get('logical_type', 'service')} component"
                else:
                    name = line
                    domain = req.context.get('business_domain', 'unknown')
                    desc = f"{req.context.get('logical_type', 'service')} component"
                
                results.append(EnhancementResult(
                    original_name=req.actor_name,
                    enhanced_name=name,
                    description=desc,
                    business_domain=domain,
                    confidence=0.8
                ))
            else:
                # Fallback for missing responses
                results.append(EnhancementResult(
                    original_name=req.actor_name,
                    enhanced_name=req.actor_name,
                    description=f"{req.context.get('logical_type', 'service')} component",
                    business_domain=req.context.get('business_domain', 'unknown'),
                    confidence=0.5
                ))
        
        return results
    
    def _enhance_logical_single(self, request: EnhancementRequest) -> EnhancementResult:
        """Enhance single logical actor"""
        
        context = request.context
        prompt = f"""Analyze this business component and provide a better business name and description:

Component: {request.actor_name}
Business Domain: {context.get('business_domain', 'unknown')}
Type: {context.get('logical_type', 'unknown')}
Responsibilities: {', '.join(context.get('responsibilities', []))}
Data Entities: {', '.join(context.get('data_entities', []))}
Operations: {', '.join(context.get('business_operations', []))}

What would be a better business name for this component? Respond with: NAME|DOMAIN|DESCRIPTION
Example: UserAccountService|User Management|User registration and account management"""
        
        try:
            response = self.llm_client.query(prompt)
            
            if '|' in response:
                parts = response.split('|')
                if len(parts) >= 3:
                    name = parts[0].strip()
                    domain = parts[1].strip()
                    desc = parts[2].strip()
                elif len(parts) == 2:
                    name = parts[0].strip()
                    domain = context.get('business_domain', 'unknown')
                    desc = parts[1].strip()
                else:
                    name = parts[0].strip()
                    domain = context.get('business_domain', 'unknown')
                    desc = f"{context.get('logical_type', 'service')} component"
                
                return EnhancementResult(
                    original_name=request.actor_name,
                    enhanced_name=name,
                    description=desc,
                    business_domain=domain,
                    confidence=0.8
                )
        except Exception as e:
            logger.warning(f"Logical single enhancement failed: {e}")
        
        # Fallback to original
        return EnhancementResult(
            original_name=request.actor_name,
            enhanced_name=request.actor_name,
            description=f"{context.get('logical_type', 'service')} component",
            business_domain=context.get('business_domain', 'unknown'),
            confidence=0.5
        )
    
    def _enhance_single(self, request: EnhancementRequest) -> EnhancementResult:
        """Enhance single actor (fallback method)"""
        
        prompt = f"""Analyze this code actor and provide a specific business name and description:

Type: {request.actor_type}
Code: {request.code_context}
Context: {request.function_context}

What specific service/system is this? Respond with just: NAME|DESCRIPTION
Example: StripePaymentAPI|Stripe payment processing service"""
        
        try:
            response = self.llm_client.query(prompt)
            
            if '|' in response:
                name, desc = response.split('|', 1)
                return EnhancementResult(
                    original_name=request.actor_name,
                    enhanced_name=name.strip(),
                    description=desc.strip(),
                    confidence=0.8
                )
        except Exception as e:
            logger.warning(f"Single enhancement failed: {e}")
        
        # Fallback to original
        return EnhancementResult(
            original_name=request.actor_name,
            enhanced_name=request.actor_name,
            description=f"{request.actor_type} interface",
            confidence=0.5
        )


class BufferedEnhancer:
    """Buffer enhancement requests and process when batch is full or timeout reached"""
    
    def __init__(self, bundler: SimpleLLMBundler):
        self.bundler = bundler
        self.buffer = []
        self.last_flush = time.time()
        self.buffer_timeout = 5.0  # seconds
    
    def add_request(self, request: EnhancementRequest) -> Optional[EnhancementResult]:
        """Add request to buffer, return result if buffer processed"""
        
        self.buffer.append(request)
        
        # Check if we should flush buffer
        should_flush = (
            len(self.buffer) >= self.bundler.batch_size or
            time.time() - self.last_flush > self.buffer_timeout
        )
        
        if should_flush:
            return self._flush_buffer()
        
        return None
    
    def flush_remaining(self) -> List[EnhancementResult]:
        """Flush any remaining requests in buffer"""
        if self.buffer:
            return self._flush_buffer()
        return []
    
    def _flush_buffer(self) -> List[EnhancementResult]:
        """Process all requests in buffer"""
        if not self.buffer:
            return []
        
        results = self.bundler.enhance_actors_bundled(self.buffer)
        self.buffer.clear()
        self.last_flush = time.time()
        
        return results


class SimpleEnhancementService:
    """Simple service that uses bundling for efficient LLM usage"""
    
    def __init__(self, llm_client):
        self.bundler = SimpleLLMBundler(llm_client, batch_size=5)
        self.stats = {
            'total_requests': 0,
            'batch_calls': 0,
            'single_calls': 0
        }
    
    def enhance_actor_list(self, actors_needing_enhancement: List) -> List[EnhancementResult]:
        """Enhance a list of actors efficiently"""
        
        if not actors_needing_enhancement:
            return []
        
        # Convert to enhancement requests
        requests = []
        for actor in actors_needing_enhancement:
            requests.append(EnhancementRequest(
                actor_name=actor.name,
                actor_type=actor.actor_type,
                code_context=actor.code_context,
                function_context=getattr(actor, 'function_name', '')
            ))
        
        # Process with bundling
        results = self.bundler.enhance_actors_bundled(requests)
        
        # Update stats
        self.stats['total_requests'] += len(requests)
        self.stats['batch_calls'] += len(requests) // self.bundler.batch_size + 1
        
        return results
    
    def enhance_logical_actors(self, requests: List[EnhancementRequest]) -> List[EnhancementResult]:
        """Enhance logical actors with business-focused enhancement"""
        
        if not requests:
            return []
        
        # Process with logical bundling
        results = self.bundler.enhance_logical_actors(requests)
        
        # Update stats
        self.stats['total_requests'] += len(requests)
        self.stats['batch_calls'] += len(requests) // self.bundler.batch_size + 1
        
        return results
    
    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get statistics about enhancement efficiency"""
        efficiency = 0
        if self.stats['total_requests'] > 0:
            efficiency = self.stats['batch_calls'] / self.stats['total_requests']
        
        return {
            **self.stats,
            'efficiency': efficiency,
            'average_batch_size': self.stats['total_requests'] / max(self.stats['batch_calls'], 1)
        }