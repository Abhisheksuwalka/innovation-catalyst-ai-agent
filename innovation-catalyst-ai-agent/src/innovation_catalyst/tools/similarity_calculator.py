# src/innovation_catalyst/tools/similarity_calculator.py
"""
Advanced similarity calculation engine with multiple metrics and optimization.
Implements efficient similarity computation with various distance metrics.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from scipy.spatial.distance import cosine, euclidean, cityblock as manhattan
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize

from ..models.connections import SemanticConnection, ConnectionType
from ..models.document import TextChunk
from ..utils.config import get_config
from ..utils.logging import get_logger, log_function_performance

logger = get_logger(__name__)
config = get_config()

@dataclass
class SimilarityMetrics:
    """Container for various similarity metrics."""
    cosine_similarity: float
    euclidean_distance: float
    manhattan_distance: float
    pearson_correlation: float
    spearman_correlation: float
    jaccard_similarity: float
    
    def get_combined_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted combined similarity score."""
        if weights is None:
            weights = {
                'cosine': 0.4,
                'euclidean': 0.2,
                'manhattan': 0.1,
                'pearson': 0.1,
                'spearman': 0.1,
                'jaccard': 0.1
            }
        
        # Normalize euclidean and manhattan (lower is better)
        normalized_euclidean = 1.0 / (1.0 + self.euclidean_distance)
        normalized_manhattan = 1.0 / (1.0 + self.manhattan_distance)
        
        combined = (
            self.cosine_similarity * weights.get('cosine', 0.4) +
            normalized_euclidean * weights.get('euclidean', 0.2) +
            normalized_manhattan * weights.get('manhattan', 0.1) +
            abs(self.pearson_correlation) * weights.get('pearson', 0.1) +
            abs(self.spearman_correlation) * weights.get('spearman', 0.1) +
            self.jaccard_similarity * weights.get('jaccard', 0.1)
        )
        
        return min(1.0, max(0.0, combined))

class SimilarityCalculator:
    """
    Advanced similarity calculation with multiple metrics and optimization.
    
    Features:
        - Multiple similarity metrics
        - Batch processing optimization
        - Parallel computation
        - Memory-efficient processing
        - Comprehensive error handling
    """
    
    def __init__(self):
        self.similarity_threshold_min = config.connection.similarity_threshold_min
        self.similarity_threshold_max = config.connection.similarity_threshold_max
        self.batch_size = config.model.batch_size
        
        logger.info("SimilarityCalculator initialized")
    
    @log_function_performance("similarity_calculation")
    def calculate_pairwise_similarities(
        self, 
        embeddings: List[List[float]],
        chunks: List[TextChunk],
        use_parallel: bool = True,
        similarity_metrics: List[str] = None
    ) -> List[Tuple[int, int, SimilarityMetrics]]:
        """
        Calculate pairwise similarities between all embeddings.
        
        Args:
            embeddings (List[List[float]]): List of embedding vectors
            chunks (List[TextChunk]): Corresponding text chunks
            use_parallel (bool): Whether to use parallel processing
            similarity_metrics (List[str]): Metrics to calculate
            
        Returns:
            List[Tuple[int, int, SimilarityMetrics]]: List of (i, j, metrics) tuples
        """
        if not embeddings or len(embeddings) < 2:
            return []
        
        if similarity_metrics is None:
            similarity_metrics = ['cosine', 'euclidean', 'jaccard']
        
        # Convert to numpy array for efficient computation
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = normalize(embeddings_array, norm='l2')
        
        logger.info(f"Calculating similarities for {len(embeddings)} embeddings")
        
        if use_parallel and len(embeddings) > 50:
            return self._calculate_parallel(
                embeddings_array, normalized_embeddings, chunks, similarity_metrics
            )
        else:
            return self._calculate_sequential(
                embeddings_array, normalized_embeddings, chunks, similarity_metrics
            )
    
    def _calculate_sequential(
        self,
        embeddings: np.ndarray,
        normalized_embeddings: np.ndarray,
        chunks: List[TextChunk],
        metrics: List[str]
    ) -> List[Tuple[int, int, SimilarityMetrics]]:
        """Calculate similarities sequentially."""
        similarities = []
        n = len(embeddings)
        
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    similarity_metrics = self._calculate_similarity_metrics(
                        embeddings[i], embeddings[j],
                        normalized_embeddings[i], normalized_embeddings[j],
                        chunks[i], chunks[j],
                        metrics
                    )
                    
                    # Filter by similarity threshold
                    if (self.similarity_threshold_min <= 
                        similarity_metrics.cosine_similarity <= 
                        self.similarity_threshold_max):
                        similarities.append((i, j, similarity_metrics))
                        
                except Exception as e:
                    logger.warning(f"Failed to calculate similarity for pair ({i}, {j}): {e}")
                    continue
        
        return similarities
    
    def _calculate_parallel(
        self,
        embeddings: np.ndarray,
        normalized_embeddings: np.ndarray,
        chunks: List[TextChunk],
        metrics: List[str]
    ) -> List[Tuple[int, int, SimilarityMetrics]]:
        """Calculate similarities in parallel."""
        similarities = []
        n = len(embeddings)
        
        # Create pairs for parallel processing
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Process in batches to manage memory
        batch_size = min(1000, len(pairs))
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            for batch_start in range(0, len(pairs), batch_size):
                batch_end = min(batch_start + batch_size, len(pairs))
                batch_pairs = pairs[batch_start:batch_end]
                
                # Submit batch for processing
                future_to_pair = {
                    executor.submit(
                        self._calculate_similarity_metrics,
                        embeddings[i], embeddings[j],
                        normalized_embeddings[i], normalized_embeddings[j],
                        chunks[i], chunks[j],
                        metrics
                    ): (i, j) for i, j in batch_pairs
                }
                
                # Collect results
                for future in as_completed(future_to_pair):
                    i, j = future_to_pair[future]
                    try:
                        similarity_metrics = future.result()
                        
                        # Filter by similarity threshold
                        if (self.similarity_threshold_min <= 
                            similarity_metrics.cosine_similarity <= 
                            self.similarity_threshold_max):
                            similarities.append((i, j, similarity_metrics))
                            
                    except Exception as e:
                        logger.warning(f"Failed to calculate similarity for pair ({i}, {j}): {e}")
                        continue
        
        return similarities
    
    def _calculate_similarity_metrics(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        normalized_embedding1: np.ndarray,
        normalized_embedding2: np.ndarray,
        chunk1: TextChunk,
        chunk2: TextChunk,
        metrics: List[str]
    ) -> SimilarityMetrics:
        """Calculate all similarity metrics for a pair of embeddings."""
        
        # Cosine similarity (using normalized embeddings)
        cosine_sim = np.dot(normalized_embedding1, normalized_embedding2)
        cosine_sim = max(0.0, min(1.0, cosine_sim))  # Clamp to [0, 1]
        
        # Euclidean distance
        euclidean_dist = np.linalg.norm(embedding1 - embedding2)
        
        # Manhattan distance
        manhattan_dist = np.sum(np.abs(embedding1 - embedding2))
        
        # Pearson correlation
        try:
            pearson_corr, _ = pearsonr(embedding1, embedding2)
            if np.isnan(pearson_corr):
                pearson_corr = 0.0
        except:
            pearson_corr = 0.0
        
        # Spearman correlation
        try:
            spearman_corr, _ = spearmanr(embedding1, embedding2)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
        except:
            spearman_corr = 0.0
        
        # Jaccard similarity (based on keywords)
        jaccard_sim = self._calculate_jaccard_similarity(chunk1, chunk2)
        
        return SimilarityMetrics(
            cosine_similarity=cosine_sim,
            euclidean_distance=euclidean_dist,
            manhattan_distance=manhattan_dist,
            pearson_correlation=pearson_corr,
            spearman_correlation=spearman_corr,
            jaccard_similarity=jaccard_sim
        )
    
    def _calculate_jaccard_similarity(self, chunk1: TextChunk, chunk2: TextChunk) -> float:
        """Calculate Jaccard similarity based on keywords and entities."""
        # Combine keywords, entities, and concepts
        set1 = set(chunk1.keywords + chunk1.entities + chunk1.concepts)
        set2 = set(chunk2.keywords + chunk2.entities + chunk2.concepts)
        
        if not set1 and not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_similarity_matrix(
        self, 
        embeddings: List[List[float]]
    ) -> np.ndarray:
        """
        Calculate full similarity matrix using optimized computation.
        
        Args:
            embeddings (List[List[float]]): List of embedding vectors
            
        Returns:
            np.ndarray: Similarity matrix
        """
        if not embeddings:
            return np.array([])
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Use sklearn's optimized cosine similarity
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # Ensure diagonal is 1.0 and matrix is symmetric
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix

# Global similarity calculator instance
similarity_calculator = SimilarityCalculator()