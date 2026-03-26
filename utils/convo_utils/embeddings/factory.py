"""
Factory for creating embedding providers based on configuration.
"""

import logging
from typing import Optional

from .providers.base import EmbeddingProvider
from .providers.local_provider import LocalEmbeddingProvider
from .providers.openai_provider import OpenAIEmbeddingProvider
from .providers.hybrid_provider import HybridEmbeddingProvider
from .config import EmbeddingConfig

logger = logging.getLogger("embedding-factory")


class EmbeddingProviderFactory:
    """Factory for creating embedding providers."""
    
    @staticmethod
    def create_provider(config: Optional[EmbeddingConfig] = None) -> EmbeddingProvider:
        """
        Create an embedding provider based on configuration.
        
        Args:
            config: Embedding configuration (uses env vars if not provided)
            
        Returns:
            Configured embedding provider
        """
        if config is None:
            config = EmbeddingConfig.from_env()
        
        logger.info(f"Creating embedding provider: {config.provider_type}")
        
        if config.provider_type == "local":
            return LocalEmbeddingProvider(
                model_name=config.local_model_name
            )
        
        elif config.provider_type == "openai":
            if not config.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI provider")
            
            return OpenAIEmbeddingProvider(
                model_name=config.openai_model_name,
                api_key=config.openai_api_key,
                max_retries=config.max_retries,
                max_daily_cost=config.max_cost_per_day
            )
        
        elif config.provider_type == "hybrid":
            # Create primary provider
            if config.fallback_provider == "openai":
                # Local primary, OpenAI fallback
                primary = LocalEmbeddingProvider(config.local_model_name)
                secondary = OpenAIEmbeddingProvider(
                    model_name=config.openai_model_name,
                    api_key=config.openai_api_key,
                    max_retries=config.max_retries,
                    max_daily_cost=config.max_cost_per_day
                )
            else:
                # OpenAI primary, local fallback (recommended)
                if not config.openai_api_key:
                    logger.warning("OpenAI API key not provided, using local-only provider")
                    return LocalEmbeddingProvider(config.local_model_name)
                
                primary = OpenAIEmbeddingProvider(
                    model_name=config.openai_model_name,
                    api_key=config.openai_api_key,
                    max_retries=config.max_retries,
                    max_daily_cost=config.max_cost_per_day
                )
                secondary = LocalEmbeddingProvider(config.local_model_name)
            
            return HybridEmbeddingProvider(
                primary=primary,
                secondary=secondary,
                circuit_breaker_enabled=config.fallback_on_error,
                prefer_primary=True
            )
        
        else:
            raise ValueError(f"Unknown provider type: {config.provider_type}")
    
    @staticmethod
    async def create_and_initialize(config: Optional[EmbeddingConfig] = None) -> EmbeddingProvider:
        """
        Create and initialize an embedding provider.
        
        Args:
            config: Embedding configuration
            
        Returns:
            Initialized embedding provider
        """
        provider = EmbeddingProviderFactory.create_provider(config)
        await provider.initialize()
        return provider