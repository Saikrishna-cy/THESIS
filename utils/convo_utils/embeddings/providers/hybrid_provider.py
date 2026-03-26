"""
Hybrid embedding provider with automatic fallback.
Provides resilience by falling back to secondary provider on errors.
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime, timedelta
from enum import Enum

from .base import EmbeddingProvider

logger = logging.getLogger("hybrid-embedding-provider")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failures exceeded threshold, using fallback
    HALF_OPEN = "half_open"  # Testing if primary recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for handling provider failures.
    """
    
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        half_open_requests: int = 1
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            half_open_requests: Number of test requests in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_count = 0
        self.lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection.
        
        Raises:
            Exception: If circuit is open or function fails
        """
        async with self.lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self.state == CircuitState.OPEN:
                if (time.time() - self.last_failure_time) > self.recovery_timeout:
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_count = 0
                else:
                    raise RuntimeError("Circuit breaker is OPEN")
            
            # Handle HALF_OPEN state
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_count >= self.half_open_requests:
                    # Successful test requests, close circuit
                    logger.info("Circuit breaker closing after successful recovery")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
        
        # Execute the function
        try:
            result = await func(*args, **kwargs)
            
            async with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.half_open_count += 1
                elif self.state == CircuitState.CLOSED:
                    self.failure_count = 0  # Reset on success
            
            return result
            
        except Exception as e:
            async with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    logger.warning(f"Circuit breaker opening after {self.failure_count} failures")
                    self.state = CircuitState.OPEN
                
                if self.state == CircuitState.HALF_OPEN:
                    # Failed during recovery test, reopen
                    logger.warning("Circuit breaker reopening after failed recovery attempt")
                    self.state = CircuitState.OPEN
            
            raise
    
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == CircuitState.OPEN
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure": datetime.fromtimestamp(self.last_failure_time).isoformat() if self.last_failure_time else None
        }


class HybridEmbeddingProvider(EmbeddingProvider):
    """
    Hybrid provider with automatic fallback between primary and secondary providers.
    """
    
    def __init__(
        self,
        primary: EmbeddingProvider,
        secondary: EmbeddingProvider,
        circuit_breaker_enabled: bool = True,
        prefer_primary: bool = True
    ):
        """
        Initialize hybrid provider.
        
        Args:
            primary: Primary embedding provider
            secondary: Secondary (fallback) provider
            circuit_breaker_enabled: Enable circuit breaker pattern
            prefer_primary: Always try primary first if True
        """
        super().__init__(model_name=f"hybrid({primary.model_name},{secondary.model_name})")
        
        self.primary = primary
        self.secondary = secondary
        self.circuit_breaker_enabled = circuit_breaker_enabled
        self.prefer_primary = prefer_primary
        
        self.circuit_breaker = CircuitBreaker() if circuit_breaker_enabled else None
        
        # Statistics
        self.stats = {
            "primary_calls": 0,
            "secondary_calls": 0,
            "fallback_count": 0,
            "total_calls": 0
        }
    
    async def initialize(self) -> None:
        """Initialize both providers."""
        if self._initialized:
            return
        
        # Initialize both providers
        init_tasks = []
        
        try:
            init_tasks.append(self.primary.initialize())
        except Exception as e:
            logger.warning(f"Failed to initialize primary provider: {e}")
        
        try:
            init_tasks.append(self.secondary.initialize())
        except Exception as e:
            logger.warning(f"Failed to initialize secondary provider: {e}")
        
        if init_tasks:
            await asyncio.gather(*init_tasks, return_exceptions=True)
        
        self._initialized = True
        logger.info(f"Initialized hybrid provider with primary={self.primary.get_provider_name()}, secondary={self.secondary.get_provider_name()}")
    
    async def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding with automatic fallback.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings with automatic fallback.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self.initialize()
        
        self.stats["total_calls"] += 1
        
        # Try primary provider
        if self.prefer_primary and (not self.circuit_breaker or not self.circuit_breaker.is_open()):
            try:
                if self.circuit_breaker:
                    result = await self.circuit_breaker.call(self.primary.embed_batch, texts)
                else:
                    result = await self.primary.embed_batch(texts)
                
                self.stats["primary_calls"] += 1
                logger.debug(f"Successfully used primary provider for {len(texts)} texts")
                return result
                
            except Exception as e:
                logger.warning(f"Primary provider failed: {e}")
                self.stats["fallback_count"] += 1
        
        # Fallback to secondary
        try:
            result = await self.secondary.embed_batch(texts)
            self.stats["secondary_calls"] += 1
            logger.debug(f"Successfully used secondary provider for {len(texts)} texts")
            return result
            
        except Exception as e:
            logger.error(f"Both providers failed: {e}")
            
            # Last resort: try primary again if we haven't
            if not self.prefer_primary:
                try:
                    result = await self.primary.embed_batch(texts)
                    self.stats["primary_calls"] += 1
                    return result
                except:
                    pass
            
            raise RuntimeError(f"All embedding providers failed. Primary: {self.primary.get_provider_name()}, Secondary: {self.secondary.get_provider_name()}")
    
    def get_embedding_dimension(self) -> int:
        """
        Get embedding dimension (should be same for both providers).
        
        Returns:
            Embedding dimension
        """
        try:
            return self.primary.get_embedding_dimension()
        except:
            return self.secondary.get_embedding_dimension()
    
    def get_provider_name(self) -> str:
        """
        Get provider name.
        
        Returns:
            Provider name
        """
        return "hybrid"
    
    def supports_language(self, language_code: str) -> bool:
        """
        Check language support (true if either provider supports it).
        
        Args:
            language_code: ISO language code
            
        Returns:
            True if language is supported
        """
        return (self.primary.supports_language(language_code) or 
                self.secondary.supports_language(language_code))
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate tokens (use primary provider's estimation).
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        try:
            return self.primary.estimate_tokens(text)
        except:
            return self.secondary.estimate_tokens(text)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()
        
        if self.circuit_breaker:
            stats["circuit_breaker"] = self.circuit_breaker.get_state()
        
        # Add provider-specific stats if available
        if hasattr(self.primary, 'get_usage_stats'):
            stats["primary_stats"] = self.primary.get_usage_stats()
        
        if hasattr(self.secondary, 'get_usage_stats'):
            stats["secondary_stats"] = self.secondary.get_usage_stats()
        
        return stats
    
    async def close(self) -> None:
        """Clean up both providers."""
        await asyncio.gather(
            self.primary.close(),
            self.secondary.close(),
            return_exceptions=True
        )
        self._initialized = False