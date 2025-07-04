# src/innovation_catalyst/tools/embeddings.py
"""
Semantic embeddings generation with advanced caching and optimization.
Implements robust embedding generation using SentenceTransformers with comprehensive error handling.
"""

import os
import time
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Core libraries
try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    torch = None

# SmolAgent integration
from smolagents import tool

from ..models.document import TextChunk
from ..utils.config import get_config
from ..utils.logging import get_logger, log_function_performance

logger = get_logger(__name__)
config = get_config()

@dataclass
class EmbeddingResult:
    """Result of embedding generation operation."""
    embeddings: List[List[float]]
    model_name: str
    dimension: int
    processing_time: float
    cache_hit: bool = False
    error_message: Optional[str] = None

class EmbeddingCache:
    """
    Advanced caching system for embeddings with persistence and memory management.
    
    Features:
        - Disk-based persistence
        - Memory-efficient storage
        - Hash-based key generation
        - Automatic cleanup
        - Thread-safe operations
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, max_memory_items: int = 1000):
        self.cache_dir = cache_dir or config.cache_dir / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_items = max_memory_items
        self.memory_cache: Dict[str, np.ndarray] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()
        
        logger.info(f"Embedding cache initialized: {self.cache_dir}")
    
    def _generate_cache_key(self, texts: List[str], model_name: str) -> str:
        """Generate unique cache key for text list and model."""
        text_hash = hashlib.sha256(
            '|'.join(sorted(texts)).encode('utf-8')
        ).hexdigest()
        return f"{model_name}_{text_hash}"
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, texts: List[str], model_name: str) -> Optional[np.ndarray]:
        """Retrieve embeddings from cache."""
        cache_key = self._generate_cache_key(texts, model_name)
        
        with self.lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                self.access_times[cache_key] = time.time()
                logger.debug(f"Cache hit (memory): {cache_key}")
                return self.memory_cache[cache_key]
            
            # Check disk cache
            cache_file = self._get_cache_file_path(cache_key)
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        embeddings = pickle.load(f)
                    
                    # Add to memory cache
                    self._add_to_memory_cache(cache_key, embeddings)
                    logger.debug(f"Cache hit (disk): {cache_key}")
                    return embeddings
                    
                except Exception as e:
                    logger.warning(f"Failed to load cache file {cache_file}: {e}")
                    cache_file.unlink(missing_ok=True)
        
        return None
    
    def put(self, texts: List[str], model_name: str, embeddings: np.ndarray) -> None:
        """Store embeddings in cache."""
        cache_key = self._generate_cache_key(texts, model_name)
        
        with self.lock:
            # Add to memory cache
            self._add_to_memory_cache(cache_key, embeddings)
            
            # Save to disk cache
            cache_file = self._get_cache_file_path(cache_key)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.debug(f"Cache stored: {cache_key}")
            except Exception as e:
                logger.warning(f"Failed to save cache file {cache_file}: {e}")
    
    def _add_to_memory_cache(self, cache_key: str, embeddings: np.ndarray) -> None:
        """Add embeddings to memory cache with LRU eviction."""
        # Remove oldest items if cache is full
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove 20% of oldest items
            items_to_remove = max(1, len(self.memory_cache) // 5)
            oldest_keys = sorted(
                self.access_times.keys(), 
                key=lambda k: self.access_times[k]
            )[:items_to_remove]
            
            for key in oldest_keys:
                self.memory_cache.pop(key, None)
                self.access_times.pop(key, None)
        
        self.memory_cache[cache_key] = embeddings
        self.access_times[cache_key] = time.time()
    
    def clear(self) -> None:
        """Clear all caches."""
        with self.lock:
            self.memory_cache.clear()
            self.access_times.clear()
            
            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        logger.info("Embedding cache cleared")

class EmbeddingGenerator:
    """
    Advanced embedding generation with multiple model support and optimization.
    
    Features:
        - Multiple model support with fallbacks
        - Batch processing optimization
        - GPU/CPU automatic selection
        - Memory management
        - Performance monitoring
        - Comprehensive error handling
    """
    
    def __init__(self):
        self.models: Dict[str, SentenceTransformer] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
        self.cache = EmbeddingCache()
        self.device = self._determine_device()
        
        # Load default model
        self.default_model_name = config.model.embedding_model
        self._load_model(self.default_model_name)
        
        logger.info(f"EmbeddingGenerator initialized with device: {self.device}")
    
    def _determine_device(self) -> str:
        """Determine optimal device for model inference."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return "cpu"
        
        device_config = config.model.device.lower()
        
        if device_config == "auto":
            if torch and torch.cuda.is_available():
                return "cuda"
            elif torch and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        elif device_config in ["cuda", "mps", "cpu"]:
            return device_config
        else:
            logger.warning(f"Unknown device config: {device_config}, defaulting to CPU")
            return "cpu"
    
    def _load_model(self, model_name: str) -> bool:
        """Load a SentenceTransformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers library not available")
            return False
        
        if model_name in self.models:
            return True
        
        try:
            logger.info(f"Loading embedding model: {model_name}")
            model = SentenceTransformer(model_name, device=self.device)
            
            # Get model information
            model_info = {
                'dimension': model.get_sentence_embedding_dimension(),
                'max_seq_length': model.max_seq_length,
                'device': str(model.device),
                'loaded_at': time.time()
            }
            
            self.models[model_name] = model
            self.model_info[model_name] = model_info
            
            logger.info(f"Model loaded successfully: {model_name} "
                       f"(dim: {model_info['dimension']}, device: {model_info['device']})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    @log_function_performance("generate_embeddings")
    def generate_embeddings(
        self, 
        texts: List[str], 
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: bool = True,
        show_progress: bool = False
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to embed
            model_name (Optional[str]): Model name to use
            batch_size (Optional[int]): Batch size for processing
            normalize (bool): Whether to normalize embeddings
            show_progress (bool): Whether to show progress bar
            
        Returns:
            EmbeddingResult: Result containing embeddings and metadata
        """
        start_time = time.time()
        
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model_name="",
                dimension=0,
                processing_time=0.0,
                error_message="No texts provided"
            )
        
        # Use default model if not specified
        model_name = model_name or self.default_model_name
        batch_size = batch_size or config.model.batch_size
        
        # Check cache first
        cached_embeddings = self.cache.get(texts, model_name)
        if cached_embeddings is not None:
            processing_time = time.time() - start_time
            return EmbeddingResult(
                embeddings=cached_embeddings.tolist(),
                model_name=model_name,
                dimension=cached_embeddings.shape[1],
                processing_time=processing_time,
                cache_hit=True
            )
        
        # Load model if needed
        if not self._load_model(model_name):
            return EmbeddingResult(
                embeddings=[],
                model_name=model_name,
                dimension=0,
                processing_time=time.time() - start_time,
                error_message=f"Failed to load model: {model_name}"
            )
        
        try:
            model = self.models[model_name]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} texts using {model_name}")
            
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # Cache the results
            self.cache.put(texts, model_name, embeddings)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Generated embeddings: {embeddings.shape} in {processing_time:.2f}s")
            
            return EmbeddingResult(
                embeddings=embeddings.tolist(),
                model_name=model_name,
                dimension=embeddings.shape[1],
                processing_time=processing_time,
                cache_hit=False
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Embedding generation failed: {str(e)}"
            logger.error(error_msg)
            
            return EmbeddingResult(
                embeddings=[],
                model_name=model_name,
                dimension=0,
                processing_time=processing_time,
                error_message=error_msg
            )
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a loaded model."""
        model_name = model_name or self.default_model_name
        return self.model_info.get(model_name, {})
    
    def get_available_models(self) -> List[str]:
        """Get list of loaded models."""
        return list(self.models.keys())
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self.cache.clear()

# Global embedding generator instance
embedding_generator = EmbeddingGenerator()

@tool
def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate semantic embeddings for text chunks using SentenceTransformers.
    
    Args:
        texts (List[str]): List of text strings to embed
        
    Returns:
        List[List[float]]: List of 384-dimensional embedding vectors
        
    Model: all-MiniLM-L6-v2 (fast, good quality, 384 dimensions)
    
    Features:
        - Batch processing for efficiency
        - Normalized vectors for cosine similarity
        - Caching to avoid re-computation
        - Error handling for invalid inputs
        
    Performance:
        - ~1000 texts/second on CPU
        - Memory efficient batch processing
        - Automatic GPU utilization if available
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("sentence-transformers not available, returning empty embeddings")
        return []
    
    result = embedding_generator.generate_embeddings(texts)
    
    if result.error_message:
        logger.error(f"Embedding generation failed: {result.error_message}")
        return []
    
    return result.embeddings

def get_embedding_model_info() -> Dict[str, Any]:
    """Get information about the current embedding model."""
    return embedding_generator.get_model_info()
