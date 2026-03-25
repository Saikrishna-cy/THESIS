"""
OpenAI embedding provider using the OpenAI API.
Includes rate limiting, cost tracking, and error handling.
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
import numpy as np
import hashlib
import json
from datetime import datetime, timedelta

from .base import EmbeddingProvider

logger = logging.getLogger("openai-embedding-provider")

# Try to import OpenAI
try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Install with: pip install openai")


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_per_minute: int = 3000):
        self.max_per_minute = max_per_minute
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Wait if necessary to respect rate limits."""
        async with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.calls = [t for t in self.calls if now - t < 60]
            
            if len(self.calls) >= self.max_per_minute:
                # Wait until the oldest call is more than 1 minute old
                sleep_time = 60 - (now - self.calls[0]) + 0.1
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
            
            self.calls.append(time.time())


class CostTracker:
    """Track API usage costs."""
    
    def __init__(self, max_daily_cost: float = 10.0):
        self.max_daily_cost = max_daily_cost
        self.daily_costs: Dict[str, float] = {}
        self.total_tokens = 0
        self.total_requests = 0
        self.lock = asyncio.Lock()
    
    async def add_usage(self, tokens: int, cost: float) -> bool:
        """
        Add usage and check if within budget.
        
        Returns:
            True if within budget, False if exceeded
        """
        async with self.lock:
            today = datetime.now().strftime("%Y-%m-%d")
            current_cost = self.daily_costs.get(today, 0.0)
            
            if current_cost + cost > self.max_daily_cost:
                logger.warning(f"Daily cost limit would be exceeded: {current_cost + cost:.4f} > {self.max_daily_cost}")
                return False
            
            self.daily_costs[today] = current_cost + cost
            self.total_tokens += tokens
            self.total_requests += 1
            
            # Clean up old days
            cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            self.daily_costs = {k: v for k, v in self.daily_costs.items() if k >= cutoff}
            
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        today = datetime.now().strftime("%Y-%m-%d")
        return {
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "today_cost": self.daily_costs.get(today, 0.0),
            "daily_costs": self.daily_costs
        }


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider using the OpenAI API.
    """
    
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "text-embedding-3-small": 0.00002,
        "text-embedding-3-large": 0.00013,
        "text-embedding-ada-002": 0.0001
    }
    
    # Embedding dimensions
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        max_daily_cost: float = 10.0
    ):
        """
        Initialize the OpenAI embedding provider.
        
        Args:
            model_name: OpenAI model name (default: text-embedding-3-small)
            api_key: OpenAI API key
            max_retries: Maximum number of retries for failed requests
            max_daily_cost: Maximum daily cost in USD
        """
        super().__init__(model_name or "text-embedding-3-small")
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        self.api_key = api_key
        self.max_retries = max_retries
        self.client: Optional[AsyncOpenAI] = None
        self.rate_limiter = RateLimiter()
        self.cost_tracker = CostTracker(max_daily_cost)
        
        # Audit logging
        self.audit_log = []
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if self._initialized:
            return
        
        try:
            if not self.api_key:
                import os
                self.api_key = os.getenv("OPENAI_API_KEY")
            
            if not self.api_key:
                raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")
            
            self.client = AsyncOpenAI(api_key=self.api_key)
            self._initialized = True
            logger.info(f"Initialized OpenAI embedding provider with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embedding provider: {e}")
            raise
    
    async def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self.initialize()
        
        # Estimate tokens and cost
        total_tokens = sum(self.estimate_tokens(t) for t in texts)
        estimated_cost = (total_tokens / 1000) * self.PRICING.get(self.model_name, 0.0001)
        
        # Check cost limit
        if not await self.cost_tracker.add_usage(0, 0):  # Check without adding
            raise RuntimeError(f"Daily cost limit exceeded. Current stats: {self.cost_tracker.get_stats()}")
        
        # Rate limiting
        await self.rate_limiter.acquire()
        
        # Make API call with retries
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                    encoding_format="float"
                )
                
                # Track actual usage
                actual_tokens = response.usage.total_tokens
                actual_cost = (actual_tokens / 1000) * self.PRICING.get(self.model_name, 0.0001)
                await self.cost_tracker.add_usage(actual_tokens, actual_cost)
                
                # Audit logging
                self._log_api_call(
                    texts=texts,
                    tokens=actual_tokens,
                    cost=actual_cost,
                    latency=time.time() - start_time,
                    success=True
                )
                
                # Convert to numpy arrays
                embeddings = [np.array(item.embedding) for item in response.data]
                return embeddings
                
            except Exception as e:
                logger.warning(f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self._log_api_call(
                        texts=texts,
                        tokens=0,
                        cost=0,
                        latency=0,
                        success=False,
                        error=str(e)
                    )
                    raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.
        
        Returns:
            Embedding dimension
        """
        return self.DIMENSIONS.get(self.model_name, 1536)
    
    def get_provider_name(self) -> str:
        """
        Get the name of this provider.
        
        Returns:
            Provider name
        """
        return "openai"
    
    def supports_language(self, language_code: str) -> bool:
        """
        Check if this provider supports a given language.
        OpenAI models support most languages.
        
        Args:
            language_code: ISO language code
            
        Returns:
            True (OpenAI supports most languages)
        """
        return True
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # More accurate estimation for OpenAI tokenizer
        # Rough estimate: 1 token per 4 characters for English, 2-3 for other languages
        return len(text) // 3
    
    def _log_api_call(
        self,
        texts: List[str],
        tokens: int,
        cost: float,
        latency: float,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Log API call for auditing."""
        # Hash texts for privacy
        text_hashes = [hashlib.sha256(t.encode()).hexdigest()[:8] for t in texts]
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "text_hashes": text_hashes,
            "num_texts": len(texts),
            "tokens": tokens,
            "cost": cost,
            "latency": latency,
            "success": success,
            "error": error
        }
        
        self.audit_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
        
        # Log to file if configured
        if success:
            logger.info(f"OpenAI API call: {len(texts)} texts, {tokens} tokens, ${cost:.6f}, {latency:.2f}s")
        else:
            logger.error(f"OpenAI API call failed: {error}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "cost_tracker": self.cost_tracker.get_stats(),
            "recent_calls": len(self.audit_log),
            "model": self.model_name
        }
    
    async def close(self) -> None:
        """Clean up resources."""
        if self.client:
            await self.client.close()
        self.client = None
        self._initialized = False