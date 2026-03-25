"""
Configuration for embedding providers.
"""

from dataclasses import dataclass
from typing import Optional, Literal
import os


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""
    
    # Provider selection
    provider_type: Literal["local", "openai", "hybrid"] = "local"
    
    # Model configuration
    local_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    openai_model_name: str = "text-embedding-3-small"
    
    # API configuration
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_org_id: Optional[str] = None
    
    # Fallback configuration
    fallback_provider: Optional[Literal["local", "openai"]] = "local"
    fallback_on_error: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Caching configuration
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 1000
    
    # Performance configuration
    batch_size: int = 100
    timeout: float = 30.0
    max_concurrent_requests: int = 10
    
    # Cost control
    max_tokens_per_minute: int = 100000
    max_cost_per_day: float = 10.0  # USD
    
    # Security
    enable_pii_detection: bool = True
    audit_logging: bool = True
    
    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Create configuration from environment variables."""
        return cls(
            provider_type=os.getenv("EMBEDDING_PROVIDER", "local"),
            local_model_name=os.getenv("LOCAL_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
            openai_model_name=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            openai_org_id=os.getenv("OPENAI_ORG_ID"),
            fallback_provider=os.getenv("EMBEDDING_FALLBACK", "local"),
            fallback_on_error=os.getenv("EMBEDDING_FALLBACK_ON_ERROR", "true").lower() == "true",
            max_retries=int(os.getenv("EMBEDDING_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("EMBEDDING_RETRY_DELAY", "1.0")),
            cache_enabled=os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true",
            cache_ttl=int(os.getenv("EMBEDDING_CACHE_TTL", "3600")),
            cache_max_size=int(os.getenv("EMBEDDING_CACHE_MAX_SIZE", "1000")),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            timeout=float(os.getenv("EMBEDDING_TIMEOUT", "30.0")),
            max_concurrent_requests=int(os.getenv("EMBEDDING_MAX_CONCURRENT", "10")),
            max_tokens_per_minute=int(os.getenv("EMBEDDING_MAX_TOKENS_PER_MINUTE", "100000")),
            max_cost_per_day=float(os.getenv("EMBEDDING_MAX_COST_PER_DAY", "10.0")),
            enable_pii_detection=os.getenv("EMBEDDING_PII_DETECTION", "true").lower() == "true",
            audit_logging=os.getenv("EMBEDDING_AUDIT_LOGGING", "true").lower() == "true",
        )