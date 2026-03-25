"""
Local embedding provider using sentence-transformers.
Wraps the existing SentenceTransformer implementation.
"""

import asyncio
import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import threading

from .base import EmbeddingProvider

logger = logging.getLogger("local-embedding-provider")

# Shared model cache to prevent multiple loads
_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.
    This wraps the existing implementation to maintain compatibility.
    """
    
    SUPPORTED_LANGUAGES = {
        'en', 'fr', 'de', 'es', 'it', 'pt', 'nl', 'pl', 'ru', 'zh', 
        'ja', 'ko', 'ar', 'tr', 'th', 'hi', 'sv', 'da', 'no', 'fi'
    }
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the local embedding provider.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        super().__init__(model_name or "paraphrase-multilingual-MiniLM-L12-v2")
        self.model: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None
    
    async def initialize(self) -> None:
        """Load the sentence-transformer model."""
        if self._initialized:
            return
        
        try:
            # Load model in thread to avoid blocking
            await asyncio.to_thread(self._load_model)
            self._initialized = True
            logger.info(f"Initialized local embedding provider with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize local embedding provider: {e}")
            raise
    
    def _load_model(self) -> None:
        """Load model with caching to prevent multiple loads."""
        with _MODEL_LOCK:
            if self.model_name in _MODEL_CACHE:
                self.model = _MODEL_CACHE[self.model_name]
                logger.info(f"Reusing cached model: {self.model_name}")
            else:
                logger.info(f"Loading model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                _MODEL_CACHE[self.model_name] = self.model
            
            # Get embedding dimension
            dummy_embedding = self.model.encode("test", convert_to_numpy=True)
            self._embedding_dim = len(dummy_embedding)
    
    async def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not self._initialized:
            await self.initialize()
        
        # Run encoding in thread pool to avoid blocking
        embedding = await asyncio.to_thread(
            self.model.encode,
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embedding
    
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
        
        # Batch encoding is more efficient
        embeddings = await asyncio.to_thread(
            self.model.encode,
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )
        
        return [embedding for embedding in embeddings]
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.
        
        Returns:
            Embedding dimension
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
        return self._embedding_dim
    
    def get_provider_name(self) -> str:
        """
        Get the name of this provider.
        
        Returns:
            Provider name
        """
        return "local"
    
    def supports_language(self, language_code: str) -> bool:
        """
        Check if this provider supports a given language.
        
        Args:
            language_code: ISO language code (e.g., "en", "fr", "de")
            
        Returns:
            True if language is supported
        """
        # The multilingual model supports many languages
        return language_code.lower() in self.SUPPORTED_LANGUAGES
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count (rough approximation)
        """
        # Rough estimate: 1 token per 4 characters (for multilingual text)
        return len(text) // 4
    
    async def close(self) -> None:
        """
        Clean up resources.
        Note: We don't actually unload the model as it might be shared.
        """
        self.model = None
        self._initialized = False
    
    def encode_sync(self, text: str) -> np.ndarray:
        """
        Synchronous encoding for backward compatibility.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not self._initialized:
            self._load_model()
            self._initialized = True
        
        return self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )