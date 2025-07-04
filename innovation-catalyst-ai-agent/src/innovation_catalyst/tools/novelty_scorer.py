# src/innovation_catalyst/tools/novelty_scorer.py
"""
Advanced novelty and innovation potential scoring system.
Implements sophisticated algorithms to identify truly innovative connections.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import math

from ..models.connections import SemanticConnection, ConnectionType, InnovationPotential
from ..models.document import TextChunk
from ..utils.config import get_config
from ..utils.logging import get_logger, log_function_performance

logger = get_logger(__name__)
config = get_config()

@dataclass
class NoveltyMetrics:
    """Container for novelty scoring metrics."""
    entity_diversity: float
    topic_diversity: float
    concept_diversity: float
    semantic_distance: float
    contextual_novelty: float
    cross_domain_score: float
    
    def get_combined_novelty(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted combined novelty score."""
        if weights is None:
            weights = {
                'entity_diversity': 0.2,
                'topic_diversity': 0.2,
                'concept_diversity': 0.2,
                'semantic_distance': 0.15,
                'contextual_novelty': 0.15,
                'cross_domain_score': 0.1
            }
        
        combined = (
            self.entity_diversity * weights.get('entity_diversity', 0.2) +
            self.topic_diversity * weights.get('topic_diversity', 0.2) +
            self.concept_diversity * weights.get('concept_diversity', 0.2) +
            self.semantic_distance * weights.get('semantic_distance', 0.15) +
            self.contextual_novelty * weights.get('contextual_novelty', 0.15) +
            self.cross_domain_score * weights.get('cross_domain_score', 0.1)
        )
        
        return min(1.0, max(0.0, combined))

class NoveltyScorer:
    """
    Advanced novelty and innovation potential scoring system.
    
    Features:
        - Multi-dimensional novelty assessment
        - Cross-domain connection detection
        - Contextual novelty evaluation
        - Innovation potential prediction
        - Domain expertise integration
    """
    
    def __init__(self):
        self.novelty_weight = config.connection.novelty_weight
        self.similarity_weight = config.connection.similarity_weight
        
        # Domain knowledge for cross-domain detection
        self.domain_keywords = self._load_domain_keywords()
        
        logger.info("NoveltyScorer initialized")
    
    def _load_domain_keywords(self) -> Dict[str, Set[str]]:
        """Load domain-specific keywords for cross-domain detection."""
        return {
            'technology': {
                'ai', 'artificial intelligence', 'machine learning', 'algorithm',
                'software', 'hardware', 'computer', 'digital', 'automation',
                'robotics', 'blockchain', 'cloud', 'data', 'analytics'
            },
            'business': {
                'market', 'customer', 'revenue', 'profit', 'strategy', 'management',
                'leadership', 'innovation', 'competition', 'growth', 'investment',
                'finance', 'marketing', 'sales', 'operations'
            },
            'healthcare': {
                'medical', 'health', 'patient', 'treatment', 'diagnosis', 'therapy',
                'pharmaceutical', 'clinical', 'hospital', 'doctor', 'nurse',
                'medicine', 'disease', 'wellness', 'prevention'
            },
            'science': {
                'research', 'experiment', 'hypothesis', 'theory', 'analysis',
                'discovery', 'innovation', 'methodology', 'data', 'results',
                'publication', 'peer review', 'laboratory', 'study'
            },
            'education': {
                'learning', 'teaching', 'student', 'teacher', 'curriculum',
                'assessment', 'knowledge', 'skill', 'training', 'development',
                'academic', 'university', 'school', 'education'
            },
            'environment': {
                'sustainability', 'climate', 'environment', 'green', 'renewable',
                'carbon', 'emission', 'conservation', 'ecosystem', 'biodiversity',
                'pollution', 'waste', 'recycling', 'energy'
            }
        }
    
    @log_function_performance("novelty_scoring")
    def calculate_novelty_score(
        self, 
        chunk1: TextChunk, 
        chunk2: TextChunk,
        similarity_score: float,
        all_chunks: Optional[List[TextChunk]] = None
    ) -> NoveltyMetrics:
        """
        Calculate comprehensive novelty score for a connection.
        
        Args:
            chunk1 (TextChunk): First chunk
            chunk2 (TextChunk): Second chunk
            similarity_score (float): Similarity score between chunks
            all_chunks (Optional[List[TextChunk]]): All chunks for context
            
        Returns:
            NoveltyMetrics: Comprehensive novelty metrics
        """
        # Entity diversity
        entity_diversity = self._calculate_entity_diversity(chunk1, chunk2)
        
        # Topic diversity
        topic_diversity = self._calculate_topic_diversity(chunk1, chunk2)
        
        # Concept diversity
        concept_diversity = self._calculate_concept_diversity(chunk1, chunk2)
        
        # Semantic distance (inverse of similarity)
        semantic_distance = 1.0 - similarity_score
        
        # Contextual novelty
        contextual_novelty = self._calculate_contextual_novelty(
            chunk1, chunk2, all_chunks
        )
        
        # Cross-domain score
        cross_domain_score = self._calculate_cross_domain_score(chunk1, chunk2)
        
        return NoveltyMetrics(
            entity_diversity=entity_diversity,
            topic_diversity=topic_diversity,
            concept_diversity=concept_diversity,
            semantic_distance=semantic_distance,
            contextual_novelty=contextual_novelty,
            cross_domain_score=cross_domain_score
        )
    
    def _calculate_entity_diversity(self, chunk1: TextChunk, chunk2: TextChunk) -> float:
        """Calculate diversity based on named entities."""
        entities1 = set(chunk1.entities)
        entities2 = set(chunk2.entities)
        
        if not entities1 and not entities2:
            return 0.5  # Neutral score when no entities
        
        # Calculate Jaccard distance (1 - Jaccard similarity)
        intersection = len(entities1.intersection(entities2))
        union = len(entities1.union(entities2))
        
        if union == 0:
            return 0.5
        
        jaccard_similarity = intersection / union
        jaccard_distance = 1.0 - jaccard_similarity
        
        return jaccard_distance
    
    def _calculate_topic_diversity(self, chunk1: TextChunk, chunk2: TextChunk) -> float:
        """Calculate diversity based on topics."""
        topics1 = set(chunk1.topics)
        topics2 = set(chunk2.topics)
        
        if not topics1 and not topics2:
            return 0.5
        
        intersection = len(topics1.intersection(topics2))
        union = len(topics1.union(topics2))
        
        if union == 0:
            return 0.5
        
        jaccard_similarity = intersection / union
        jaccard_distance = 1.0 - jaccard_similarity
        
        return jaccard_distance
    
    def _calculate_concept_diversity(self, chunk1: TextChunk, chunk2: TextChunk) -> float:
        """Calculate diversity based on concepts."""
        concepts1 = set(chunk1.concepts)
        concepts2 = set(chunk2.concepts)
        
        if not concepts1 and not concepts2:
            return 0.5
        
        intersection = len(concepts1.intersection(concepts2))
        union = len(concepts1.union(concepts2))
        
        if union == 0:
            return 0.5
        
        jaccard_similarity = intersection / union
        jaccard_distance = 1.0 - jaccard_similarity
        
        return jaccard_distance
    
    def _calculate_contextual_novelty(
        self, 
        chunk1: TextChunk, 
        chunk2: TextChunk,
        all_chunks: Optional[List[TextChunk]]
    ) -> float:
        """Calculate novelty based on context within the document corpus."""
        if not all_chunks or len(all_chunks) < 3:
            return 0.5  # Neutral score for small corpus
        
        # Get all keywords from the corpus
        all_keywords = []
        for chunk in all_chunks:
            all_keywords.extend(chunk.keywords)
        
        keyword_freq = Counter(all_keywords)
        total_keywords = len(all_keywords)
        
        if total_keywords == 0:
            return 0.5
        
        # Calculate rarity score for keywords in both chunks
        chunk1_rarity = self._calculate_keyword_rarity(chunk1.keywords, keyword_freq, total_keywords)
        chunk2_rarity = self._calculate_keyword_rarity(chunk2.keywords, keyword_freq, total_keywords)
        
        # Average rarity indicates contextual novelty
        contextual_novelty = (chunk1_rarity + chunk2_rarity) / 2
        
        return contextual_novelty
    
    def _calculate_keyword_rarity(
        self, 
        keywords: List[str], 
        keyword_freq: Counter, 
        total_keywords: int
    ) -> float:
        """Calculate rarity score for a list of keywords."""
        if not keywords:
            return 0.5
        
        rarity_scores = []
        for keyword in keywords:
            freq = keyword_freq.get(keyword, 0)
            if freq > 0:
                # Use inverse frequency as rarity (rare keywords get higher scores)
                rarity = 1.0 - (freq / total_keywords)
                rarity_scores.append(rarity)
        
        return sum(rarity_scores) / len(rarity_scores) if rarity_scores else 0.5
    
    def _calculate_cross_domain_score(self, chunk1: TextChunk, chunk2: TextChunk) -> float:
        """Calculate score based on cross-domain connections."""
        # Get domains for each chunk
        domains1 = self._identify_domains(chunk1)
        domains2 = self._identify_domains(chunk2)
        
        if not domains1 or not domains2:
            return 0.0  # No clear domains identified
        
        # Check for cross-domain connections
        common_domains = domains1.intersection(domains2)
        different_domains = domains1.symmetric_difference(domains2)
        
        if not common_domains and different_domains:
            # Pure cross-domain connection
            return 1.0
        elif common_domains and different_domains:
            # Mixed domain connection
            cross_domain_ratio = len(different_domains) / (len(common_domains) + len(different_domains))
            return cross_domain_ratio
        else:
            # Same domain connection
            return 0.0
    
    def _identify_domains(self, chunk: TextChunk) -> Set[str]:
        """Identify domains based on chunk content."""
        domains = set()
        
        # Combine all text elements
        all_text = ' '.join([
            chunk.content.lower(),
            ' '.join(chunk.keywords),
            ' '.join(chunk.entities),
            ' '.join(chunk.concepts)
        ])
        
        # Check for domain keywords
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    domains.add(domain)
                    break  # Found at least one keyword for this domain
        
        return domains
    
    def calculate_innovation_potential(
        self, 
        similarity_score: float,
        novelty_metrics: NoveltyMetrics,
        chunk1: TextChunk,
        chunk2: TextChunk
    ) -> Tuple[float, InnovationPotential]:
        """
        Calculate innovation potential score and level.
        
        Args:
            similarity_score (float): Similarity score
            novelty_metrics (NoveltyMetrics): Novelty metrics
            chunk1 (TextChunk): First chunk
            chunk2 (TextChunk): Second chunk
            
        Returns:
            Tuple[float, InnovationPotential]: Innovation score and level
        """
        # Get combined novelty score
        novelty_score = novelty_metrics.get_combined_novelty()
        
        # Innovation potential combines similarity and novelty
        # Sweet spot: moderate to high similarity + high novelty
        innovation_score = self._calculate_innovation_score(similarity_score, novelty_score)
        
        # Determine innovation level
        innovation_level = self._determine_innovation_level(
            innovation_score, novelty_metrics, chunk1, chunk2
        )
        
        return innovation_score, innovation_level
    
    def _calculate_innovation_score(self, similarity_score: float, novelty_score: float) -> float:
        """Calculate innovation score using optimized formula."""
        # Innovation sweet spot: similarity between 0.3-0.7 with high novelty
        similarity_factor = self._similarity_factor(similarity_score)
        novelty_factor = novelty_score
        
        # Combine with weights
        innovation_score = (
            similarity_factor * self.similarity_weight +
            novelty_factor * self.novelty_weight
        )
        
        return min(1.0, max(0.0, innovation_score))
    
    def _similarity_factor(self, similarity_score: float) -> float:
        """Calculate similarity factor for innovation scoring."""
        # Optimal similarity range for innovation: 0.3 to 0.7
        if 0.3 <= similarity_score <= 0.7:
            # Peak at 0.5 similarity
            return 1.0 - 2 * abs(similarity_score - 0.5)
        elif similarity_score < 0.3:
            # Too dissimilar - lower innovation potential
            return similarity_score / 0.3
        else:
            # Too similar - lower innovation potential
            return (1.0 - similarity_score) / 0.3
    
    def _determine_innovation_level(
        self, 
        innovation_score: float,
        novelty_metrics: NoveltyMetrics,
        chunk1: TextChunk,
        chunk2: TextChunk
    ) -> InnovationPotential:
        """Determine innovation potential level."""
        # Base level on innovation score
        if innovation_score >= 0.8:
            base_level = InnovationPotential.BREAKTHROUGH
        elif innovation_score >= 0.6:
            base_level = InnovationPotential.HIGH
        elif innovation_score >= 0.4:
            base_level = InnovationPotential.MEDIUM
        else:
            base_level = InnovationPotential.LOW
        
        # Adjust based on specific criteria
        if novelty_metrics.cross_domain_score >= 0.8:
            # Strong cross-domain connection boosts innovation level
            if base_level == InnovationPotential.HIGH:
                return InnovationPotential.BREAKTHROUGH
            elif base_level == InnovationPotential.MEDIUM:
                return InnovationPotential.HIGH
        
        return base_level

# Global novelty scorer instance
novelty_scorer = NoveltyScorer()
