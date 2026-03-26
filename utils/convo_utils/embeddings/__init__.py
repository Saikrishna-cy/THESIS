"""
Embeddings module for semantic matching.
Provides abstraction over different embedding providers (local, OpenAI, etc.)
"""

from .providers.base import EmbeddingProvider
from .providers.local_provider import LocalEmbeddingProvider
from .providers.openai_provider import OpenAIEmbeddingProvider
from .providers.hybrid_provider import HybridEmbeddingProvider
from .config import EmbeddingConfig

__all__ = [
    'EmbeddingProvider',
    'LocalEmbeddingProvider', 
    'OpenAIEmbeddingProvider',
    'HybridEmbeddingProvider',
    'EmbeddingConfig'
]