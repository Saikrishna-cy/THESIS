"""
Abstract base class for embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np
import logging

logger = logging.getLogger("embedding-provider")


class EmbeddingProvider(ABC):
    """
    Abstract base class for all embedding providers.
    Defines the interface that all providers must implement.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding provider.
        
        Args:
            model_name: Name/identifier of the model to use
        """
        self.model_name = model_name
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the provider (load models, connect to APIs, etc).
        Should be called before first use.
        """
        pass
    
    @abstractmethod
    async def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.
        
        Returns:
            Embedding dimension
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of this provider.
        
        Returns:
            Provider name (e.g., "local", "openai")
        """
        pass
    
    @abstractmethod
    def supports_language(self, language_code: str) -> bool:
        """
        Check if this provider supports a given language.
        
        Args:
            language_code: ISO language code (e.g., "en", "fr", "de")
            
        Returns:
            True if language is supported
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the text.
        Used for cost estimation and rate limiting.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        pass
    
    async def close(self) -> None:
        """
        Clean up resources (close connections, free memory, etc).
        Override if provider needs cleanup.
        """
        pass
    
    def is_initialized(self) -> bool:
        """
        Check if the provider has been initialized.
        
        Returns:
            True if initialized
        """
        return self._initialized