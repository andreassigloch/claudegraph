#!/usr/bin/env python3
"""
Universal LLM Client Interface for Code Architecture Analyzer

Supports local LLM, OpenAI, and Anthropic APIs with unified interface.
"""

import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """Request structure for LLM calls."""
    prompt: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 30
    metadata: Dict[str, Any] = None


@dataclass
class LLMResponse:
    """Response structure from LLM calls."""
    content: str
    provider: str
    model: str
    usage: Dict[str, int] = None
    metadata: Dict[str, Any] = None
    raw_response: Dict[str, Any] = None


@dataclass
class ClassificationResult:
    """Structured result from code classification."""
    classification: str
    confidence: float
    reasoning: str
    item_id: Optional[str] = None


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMClientBase(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = self.__class__.__name__.lower().replace('client', '')
        self._setup_session()
    
    def _setup_session(self):
        """Setup HTTP session with retry strategy."""
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=self.config.get('max_retries', 3),
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    @abstractmethod
    def call(self, request: LLMRequest) -> LLMResponse:
        """Make a single LLM call."""
        pass
    
    @abstractmethod
    def call_batch(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Make batch LLM calls."""
        pass
    
    def parse_classification_response(self, response: LLMResponse) -> ClassificationResult:
        """Parse classification response into structured result."""
        try:
            # Try to parse as JSON
            data = json.loads(response.content.strip())
            
            return ClassificationResult(
                classification=data.get('classification', 'Unknown'),
                confidence=float(data.get('confidence', 0.0)),
                reasoning=data.get('reasoning', 'No reasoning provided'),
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse classification response: {e}")
            logger.debug(f"Raw response: {response.content}")
            
            # Fallback parsing
            content = response.content.lower()
            if 'httpclient' in content:
                classification = 'HttpClient'
                confidence = 0.5
            elif 'database' in content:
                classification = 'Database'
                confidence = 0.5
            elif 'filesystem' in content:
                classification = 'FileSystem'
                confidence = 0.5
            elif 'webendpoint' in content:
                classification = 'WebEndpoint'
                confidence = 0.5
            else:
                classification = 'Unknown'
                confidence = 0.1
            
            return ClassificationResult(
                classification=classification,
                confidence=confidence,
                reasoning="Fallback parsing due to invalid JSON response",
            )


class LocalLLMClient(LLMClientBase):
    """Client for local LLM server (e.g., text-generation-webui, ollama)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config['base_url'].rstrip('/')
        self.model = config.get('model', 'local-model')
        self.headers = config.get('headers', {})
        self.timeout = config.get('timeout', 30)
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to local LLM server."""
        try:
            response = self.session.get(
                f"{self.base_url}/v1/models",
                headers=self.headers,
                timeout=5
            )
            response.raise_for_status()
            logger.info(f"Successfully connected to local LLM at {self.base_url}")
        except Exception as e:
            logger.warning(f"Could not connect to local LLM at {self.base_url}: {e}")
    
    def call(self, request: LLMRequest) -> LLMResponse:
        """Make a single call to local LLM."""
        payload = {
            "model": request.model or self.model,
            "messages": [
                {"role": "user", "content": request.prompt}
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self.headers,
                timeout=request.timeout or self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            return LLMResponse(
                content=content,
                provider="local",
                model=request.model or self.model,
                usage=data.get('usage', {}),
                raw_response=data
            )
            
        except Exception as e:
            logger.error(f"Local LLM call failed: {e}")
            raise LLMClientError(f"Local LLM call failed: {e}")
    
    def call_batch(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Make batch calls to local LLM (sequential for now)."""
        responses = []
        for req in requests:
            try:
                response = self.call(req)
                responses.append(response)
            except Exception as e:
                logger.error(f"Batch item failed: {e}")
                # Create error response
                responses.append(LLMResponse(
                    content='{"classification": "Unknown", "confidence": 0.0, "reasoning": "API call failed"}',
                    provider="local",
                    model=req.model or self.model,
                    metadata={"error": str(e)}
                ))
        return responses


class OpenAIClient(LLMClientBase):
    """Client for OpenAI API."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if openai is None:
            raise LLMClientError("OpenAI library not installed. Run: pip install openai")
        
        self.api_key = config['api_key']
        self.model = config.get('model', 'gpt-4')
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=config.get('timeout', 30)
        )
    
    def call(self, request: LLMRequest) -> LLMResponse:
        """Make a single call to OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=request.model or self.model,
                messages=[
                    {"role": "user", "content": request.prompt}
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                timeout=request.timeout
            )
            
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return LLMResponse(
                content=content,
                provider="openai",
                model=response.model,
                usage=usage,
                raw_response=response.model_dump()
            )
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise LLMClientError(f"OpenAI API call failed: {e}")
    
    def call_batch(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Make batch calls to OpenAI API."""
        responses = []
        for req in requests:
            try:
                response = self.call(req)
                responses.append(response)
                # Rate limiting
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Batch item failed: {e}")
                responses.append(LLMResponse(
                    content='{"classification": "Unknown", "confidence": 0.0, "reasoning": "API call failed"}',
                    provider="openai",
                    model=req.model or self.model,
                    metadata={"error": str(e)}
                ))
        return responses


class AnthropicClient(LLMClientBase):
    """Client for Anthropic Claude API."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if anthropic is None:
            raise LLMClientError("Anthropic library not installed. Run: pip install anthropic")
        
        self.api_key = config['api_key']
        self.model = config.get('model', 'claude-3-sonnet-20240229')
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=config.get('timeout', 30)
        )
    
    def call(self, request: LLMRequest) -> LLMResponse:
        """Make a single call to Anthropic API."""
        try:
            response = self.client.messages.create(
                model=request.model or self.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=[
                    {"role": "user", "content": request.prompt}
                ]
            )
            
            content = response.content[0].text
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            
            return LLMResponse(
                content=content,
                provider="anthropic",
                model=response.model,
                usage=usage,
                raw_response=response.model_dump()
            )
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise LLMClientError(f"Anthropic API call failed: {e}")
    
    def call_batch(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Make batch calls to Anthropic API."""
        responses = []
        for req in requests:
            try:
                response = self.call(req)
                responses.append(response)
                # Rate limiting
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Batch item failed: {e}")
                responses.append(LLMResponse(
                    content='{"classification": "Unknown", "confidence": 0.0, "reasoning": "API call failed"}',
                    provider="anthropic",
                    model=req.model or self.model,
                    metadata={"error": str(e)}
                ))
        return responses


class LLMClientFactory:
    """Factory for creating LLM clients based on configuration."""
    
    _clients = {
        'local': LocalLLMClient,
        'openai': OpenAIClient,
        'anthropic': AnthropicClient,
    }
    
    @classmethod
    def create_client(cls, config: Dict[str, Any]) -> LLMClientBase:
        """Create LLM client based on configuration."""
        provider = config.get('provider', 'local').lower()
        
        if provider not in cls._clients:
            raise LLMClientError(f"Unsupported LLM provider: {provider}")
        
        provider_config = config.get(provider, {})
        
        # Merge general config with provider-specific config
        merged_config = {**config, **provider_config}
        
        client_class = cls._clients[provider]
        return client_class(merged_config)
    
    @classmethod
    def register_client(cls, name: str, client_class: type):
        """Register a new LLM client type."""
        cls._clients[name] = client_class


class LLMManager:
    """High-level manager for LLM operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Extract LLM-specific configuration
        llm_config = config.get('llm', {})
        self.client = LLMClientFactory.create_client(llm_config)
        self.confidence_threshold = llm_config.get('confidence_threshold', 0.7)
        self.max_batch_size = llm_config.get('max_batch_size', 10)
        
        # Initialize prompt templates
        self.prompts = llm_config.get('prompts', {})
        
        logger.info(f"Initialized LLM manager with provider: {llm_config.get('provider', 'local')}")
    
    def get_client(self) -> LLMClientBase:
        """Get the underlying LLM client."""
        return self.client
    
    def classify_code_pattern(self, code_snippet: str, context: str = "", imports: List[str] = None) -> ClassificationResult:
        """Classify a single code pattern."""
        imports = imports or []
        prompt = self.prompts.get('classification', '').format(
            code_snippet=code_snippet,
            context=context,
            imports=', '.join(imports)
        )
        
        request = LLMRequest(
            prompt=prompt,
            model=self.config.get('model', ''),
            temperature=self.config.get('temperature', 0.1),
            max_tokens=self.config.get('max_tokens', 1000),
            timeout=self.config.get('timeout', 30)
        )
        
        try:
            response = self.client.call(request)
            result = self.client.parse_classification_response(response)
            logger.debug(f"Classified pattern as {result.classification} with confidence {result.confidence}")
            return result
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ClassificationResult(
                classification="Unknown",
                confidence=0.0,
                reasoning=f"Classification failed: {e}"
            )
    
    def classify_batch(self, items: List[Dict[str, Any]]) -> List[ClassificationResult]:
        """Classify multiple code patterns in batch."""
        results = []
        
        # Process in batches
        for i in range(0, len(items), self.max_batch_size):
            batch = items[i:i + self.max_batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[ClassificationResult]:
        """Process a single batch of items."""
        # For now, process individually
        # TODO: Implement true batch processing for providers that support it
        results = []
        
        for item in batch:
            result = self.classify_code_pattern(
                code_snippet=item.get('code_snippet', ''),
                context=item.get('context', ''),
                imports=item.get('imports', [])
            )
            result.item_id = item.get('item_id')
            results.append(result)
        
        return results
    
    def is_high_confidence(self, result: ClassificationResult) -> bool:
        """Check if classification result has high confidence."""
        return result.confidence >= self.confidence_threshold
    
    def get_uncertain_items(self, results: List[ClassificationResult]) -> List[ClassificationResult]:
        """Filter results that need user review."""
        return [r for r in results if not self.is_high_confidence(r)]