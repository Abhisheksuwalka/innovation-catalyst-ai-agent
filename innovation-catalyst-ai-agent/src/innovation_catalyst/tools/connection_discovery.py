# src/innovation_catalyst/tools/connection_discovery.py
"""
Main connection discovery tool that orchestrates the entire process.
Implements the core @tool function for SmolAgent integration.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# SmolAgent integration
from smolagents import tool

from ..models.connections import (
    SemanticConnection, ConnectionType, ConnectionStrength,
    ConnectionDiscoveryResult, ConnectionCluster
)
from ..models.document import TextChunk
from ..tools.similarity_calculator import similarity_calculator
from ..tools.novelty_scorer import novelty_scorer
from ..utils.config import get_config
from ..utils.logging import get_logger, log_function_performance

logger = get_logger(__name__)
config = get_config()

class ConnectionExplainer:
    """Generates human-readable explanations for connections."""
    
    def __init__(self):
        self.explanation_templates = {
            ConnectionType.SEMANTIC: [
                "These chunks share similar semantic concepts: {shared_concepts}",
                "Both chunks discuss related themes around {main_theme}",
                "Semantic similarity detected in topics: {shared_topics}"
            ],
            ConnectionType.ENTITY: [
                "Both chunks mention the same entities: {shared_entities}",
                "Common entities create a connection: {shared_entities}",
                "Entity overlap found: {shared_entities}"
            ],
            ConnectionType.TOPIC: [
                "Both chunks cover similar topics: {shared_topics}",
                "Topic alignment detected: {shared_topics}",
                "Related subject matter: {shared_topics}"
            ],
            ConnectionType.CONCEPT: [
                "Conceptual similarity in: {shared_concepts}",
                "Both chunks explore related concepts: {shared_concepts}",
                "Conceptual connection through: {shared_concepts}"
            ],
            ConnectionType.HYBRID: [
                "Multi-dimensional connection combining {connection_aspects}",
                "Complex relationship involving {connection_aspects}",
                "Hybrid connection across {connection_aspects}"
            ]
        }
    
    def generate_explanation(self, connection: SemanticConnection, chunk1: TextChunk, chunk2: TextChunk) -> str:
        """Generate human-readable explanation for a connection."""
        try:
            templates = self.explanation_templates.get(connection.connection_type, [])
            if not templates:
                return "Connection detected between chunks."
            
            # Choose template based on available shared elements
            template = templates[0]  # Default to first template
            
            # Prepare context variables
            context = {
                'shared_entities': ', '.join(connection.shared_entities[:3]),
                'shared_topics': ', '.join(connection.shared_topics[:3]),
                'shared_concepts': ', '.join(connection.shared_concepts[:3]),
                'shared_keywords': ', '.join(connection.shared_keywords[:3]),
                'main_theme': connection.shared_topics[0] if connection.shared_topics else 'related concepts',
                'connection_aspects': self._get_connection_aspects(connection)
            }
            
            # Format explanation
            explanation = template.format(**context)
            
            # Add innovation rationale if high potential
            if connection.innovation_potential > 0.7:
                innovation_reason = self._generate_innovation_rationale(connection, chunk1, chunk2)
                explanation += f" {innovation_reason}"
            
            return explanation
            
        except Exception as e:
            logger.warning(f"Failed to generate explanation: {e}")
            return "Connection detected between chunks."
    
    def _get_connection_aspects(self, connection: SemanticConnection) -> str:
        """Get aspects involved in a hybrid connection."""
        aspects = []
        if connection.shared_entities:
            aspects.append("entities")
        if connection.shared_topics:
            aspects.append("topics")
        if connection.shared_concepts:
            aspects.append("concepts")
        if connection.shared_keywords:
            aspects.append("keywords")
        
        return ', '.join(aspects) if aspects else "multiple dimensions"
    
    def _generate_innovation_rationale(self, connection: SemanticConnection, chunk1: TextChunk, chunk2: TextChunk) -> str:
        """Generate rationale for why this connection is innovative."""
        rationales = []
        
        if connection.novelty_score > 0.8:
            rationales.append("This represents a novel combination of ideas")
        
        if len(connection.shared_entities) > 0 and len(connection.shared_topics) > 0:
            rationales.append("The connection bridges different domains")
        
        if connection.similarity_score > 0.6 and connection.novelty_score > 0.6:
            rationales.append("This combines familiar concepts in an innovative way")
        
        return ". ".join(rationales) + "." if rationales else "This connection shows innovation potential."

class ConnectionDiscoveryEngine:
    """
    Main connection discovery engine that orchestrates the entire process.
    
    Features:
        - Comprehensive similarity calculation
        - Advanced novelty scoring
        - Connection clustering
        - Explanation generation
        - Performance optimization
    """
    
    def __init__(self):
        self.max_connections = config.connection.max_connections
        self.explainer = ConnectionExplainer()
        
        logger.info("ConnectionDiscoveryEngine initialized")
    
    @log_function_performance("connection_discovery")
    def discover_connections(
        self,
        chunks_data: List[Dict[str, Any]],
        max_connections: Optional[int] = None
    ) -> ConnectionDiscoveryResult:
        """
        Discover semantic connections between chunks.
        
        Args:
            chunks_data (List[Dict[str, Any]]): Processed chunks with embeddings
            max_connections (Optional[int]): Maximum connections to return
            
        Returns:
            ConnectionDiscoveryResult: Complete discovery results
        """
        start_time = time.time()
        max_connections = max_connections or self.max_connections
        
        try:
            # Validate input
            if not chunks_data or len(chunks_data) < 2:
                return ConnectionDiscoveryResult(
                    success=False,
                    error_message="Need at least 2 chunks to discover connections"
                )
            
            # Convert to TextChunk objects and extract embeddings
            chunks, embeddings = self._prepare_data(chunks_data)
            
            if not embeddings:
                return ConnectionDiscoveryResult(
                    success=False,
                    error_message="No embeddings found in chunk data"
                )
            
            logger.info(f"Discovering connections for {len(chunks)} chunks")
            
            # Calculate pairwise similarities
            similarities = similarity_calculator.calculate_pairwise_similarities(
                embeddings, chunks, use_parallel=len(chunks) > 50
            )
            
            if not similarities:
                return ConnectionDiscoveryResult(
                    success=True,
                    connections=[],
                    processing_time=time.time() - start_time,
                    error_message="No connections found within similarity thresholds"
                )
            
            # Create connections with novelty scoring
            connections = self._create_connections(similarities, chunks)
            
            # Rank and filter connections
            top_connections = self._rank_and_filter_connections(connections, max_connections)
            
            # Create clusters
            clusters = self._create_connection_clusters(top_connections)
            
            # Create result
            result = ConnectionDiscoveryResult(
                success=True,
                connections=top_connections,
                clusters=clusters,
                processing_time=time.time() - start_time
            )
            
            result.update_statistics()
            
            logger.info(f"Discovered {len(top_connections)} connections in {result.processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Connection discovery failed: {str(e)}"
            logger.error(error_msg)
            
            return ConnectionDiscoveryResult(
                success=False,
                error_message=error_msg,
                processing_time=processing_time
            )
    
    def _prepare_data(self, chunks_data: List[Dict[str, Any]]) -> Tuple[List[TextChunk], List[List[float]]]:
        """Prepare chunks and embeddings from input data."""
        chunks = []
        embeddings = []
        
        for chunk_dict in chunks_data:
            try:
                # Create TextChunk object
                chunk = TextChunk(
                    chunk_id=chunk_dict.get('chunk_id', ''),
                    content=chunk_dict.get('content', ''),
                    chunk_index=chunk_dict.get('chunk_index', 0),
                    word_count=chunk_dict.get('word_count', 0),
                    sentence_count=chunk_dict.get('sentence_count', 0),
                    start_position=chunk_dict.get('start_position', 0),
                    end_position=chunk_dict.get('end_position', 0),
                    source_document_id=chunk_dict.get('source_document_id', ''),
                    entities=chunk_dict.get('entities', []),
                    topics=chunk_dict.get('topics', []),
                    keywords=chunk_dict.get('keywords', []),
                    concepts=chunk_dict.get('concepts', [])
                )
                
                # Extract embedding
                embedding = chunk_dict.get('embedding', [])
                if embedding:
                    chunks.append(chunk)
                    embeddings.append(embedding)
                    
            except Exception as e:
                logger.warning(f"Failed to process chunk data: {e}")
                continue
        
        return chunks, embeddings
    
    def _create_connections(
        self, 
        similarities: List[Tuple[int, int, Any]], 
        chunks: List[TextChunk]
    ) -> List[SemanticConnection]:
        """Create SemanticConnection objects from similarity data."""
        connections = []
        
        for i, j, similarity_metrics in similarities:
            try:
                chunk1 = chunks[i]
                chunk2 = chunks[j]
                
                # Calculate novelty metrics
                novelty_metrics = novelty_scorer.calculate_novelty_score(
                    chunk1, chunk2, similarity_metrics.cosine_similarity, chunks
                )
                
                # Calculate innovation potential
                innovation_score, innovation_level = novelty_scorer.calculate_innovation_potential(
                    similarity_metrics.cosine_similarity, novelty_metrics, chunk1, chunk2
                )
                
                # Determine connection type and strength
                connection_type = self._determine_connection_type(chunk1, chunk2)
                connection_strength = self._determine_connection_strength(
                    similarity_metrics.cosine_similarity, novelty_metrics.get_combined_novelty()
                )
                
                # Find shared elements
                shared_elements = self._find_shared_elements(chunk1, chunk2)
                
                # Create connection
                connection = SemanticConnection(
                    chunk_a_id=chunk1.chunk_id,
                    chunk_b_id=chunk2.chunk_id,
                    chunk_a_index=i,
                    chunk_b_index=j,
                    similarity_score=similarity_metrics.cosine_similarity,
                    novelty_score=novelty_metrics.get_combined_novelty(),
                    innovation_potential=innovation_score,
                    connection_type=connection_type,
                    connection_strength=connection_strength,
                    innovation_level=innovation_level,
                    shared_entities=shared_elements['entities'],
                    shared_topics=shared_elements['topics'],
                    shared_keywords=shared_elements['keywords'],
                    shared_concepts=shared_elements['concepts'],
                    explanation="",  # Will be filled later
                    confidence_score=similarity_metrics.get_combined_score(),
                    quality_score=self._calculate_quality_score(similarity_metrics, novelty_metrics)
                )
                
                # Generate explanation
                connection.explanation = self.explainer.generate_explanation(connection, chunk1, chunk2)
                
                connections.append(connection)
                
            except Exception as e:
                logger.warning(f"Failed to create connection for pair ({i}, {j}): {e}")
                continue
        
        return connections
    
    def _determine_connection_type(self, chunk1: TextChunk, chunk2: TextChunk) -> ConnectionType:
        """Determine the primary type of connection."""
        shared_entities = len(set(chunk1.entities).intersection(set(chunk2.entities)))
        shared_topics = len(set(chunk1.topics).intersection(set(chunk2.topics)))
        shared_concepts = len(set(chunk1.concepts).intersection(set(chunk2.concepts)))
        shared_keywords = len(set(chunk1.keywords).intersection(set(chunk2.keywords)))
        
        # Count non-zero overlaps
        overlap_types = sum([
            shared_entities > 0,
            shared_topics > 0,
            shared_concepts > 0,
            shared_keywords > 0
        ])
        
        if overlap_types >= 3:
            return ConnectionType.HYBRID
        elif shared_entities > 0 and shared_entities >= max(shared_topics, shared_concepts):
            return ConnectionType.ENTITY
        elif shared_topics > 0 and shared_topics >= max(shared_entities, shared_concepts):
            return ConnectionType.TOPIC
        elif shared_concepts > 0:
            return ConnectionType.CONCEPT
        else:
            return ConnectionType.SEMANTIC
    
    def _determine_connection_strength(self, similarity_score: float, novelty_score: float) -> ConnectionStrength:
        """Determine connection strength based on similarity and novelty."""
        combined_score = (similarity_score + novelty_score) / 2
        
        if combined_score >= 0.8:
            return ConnectionStrength.VERY_STRONG
        elif combined_score >= 0.6:
            return ConnectionStrength.STRONG
        elif combined_score >= 0.4:
            return ConnectionStrength.MODERATE
        else:
            return ConnectionStrength.WEAK
    
    def _find_shared_elements(self, chunk1: TextChunk, chunk2: TextChunk) -> Dict[str, List[str]]:
        """Find shared elements between two chunks."""
        return {
            'entities': list(set(chunk1.entities).intersection(set(chunk2.entities))),
            'topics': list(set(chunk1.topics).intersection(set(chunk2.topics))),
            'keywords': list(set(chunk1.keywords).intersection(set(chunk2.keywords))),
            'concepts': list(set(chunk1.concepts).intersection(set(chunk2.concepts)))
        }
    
    def _calculate_quality_score(self, similarity_metrics: Any, novelty_metrics: Any) -> float:
        """Calculate overall quality score for a connection."""
        # Combine multiple factors for quality assessment
        similarity_quality = similarity_metrics.get_combined_score()
        novelty_quality = novelty_metrics.get_combined_novelty()
        
        # Quality is high when both similarity and novelty are balanced
        quality_score = (similarity_quality + novelty_quality) / 2
        
        return quality_score
    
    def _rank_and_filter_connections(
        self, 
        connections: List[SemanticConnection], 
        max_connections: int
    ) -> List[SemanticConnection]:
        """Rank connections and return top ones."""
        # Sort by combined score (innovation potential + quality)
        connections.sort(
            key=lambda conn: (conn.innovation_potential + conn.quality_score) / 2,
            reverse=True
        )
        
        return connections[:max_connections]
    
    def _create_connection_clusters(self, connections: List[SemanticConnection]) -> List[ConnectionCluster]:
        """Create clusters of related connections."""
        if not connections:
            return []
        
        # Simple clustering based on shared keywords
        clusters = []
        used_connections = set()
        
        for connection in connections:
            if connection.connection_id in used_connections:
                continue
            
            # Create new cluster
            cluster = ConnectionCluster()
            cluster.add_connection(connection)
            used_connections.add(connection.connection_id)
            
            # Find related connections
            for other_connection in connections:
                if (other_connection.connection_id in used_connections or 
                    other_connection.connection_id == connection.connection_id):
                    continue
                
                # Check if connections are related (share keywords/topics)
                if self._are_connections_related(connection, other_connection):
                    cluster.add_connection(other_connection)
                    used_connections.add(other_connection.connection_id)
            
            clusters.append(cluster)
        
        return clusters
    
    def _are_connections_related(self, conn1: SemanticConnection, conn2: SemanticConnection) -> bool:
        """Check if two connections are related enough to be in the same cluster."""
        # Check for shared keywords or topics
        shared_keywords = set(conn1.shared_keywords).intersection(set(conn2.shared_keywords))
        shared_topics = set(conn1.shared_topics).intersection(set(conn2.shared_topics))
        
        return len(shared_keywords) > 0 or len(shared_topics) > 0

# Global connection discovery engine
connection_discovery_engine = ConnectionDiscoveryEngine()

@tool
def discover_semantic_connections(chunks_data: List[Dict[str, Any]], max_connections: int = 20) -> List[Dict[str, Any]]:
    """
    Find semantic connections between knowledge chunks using vector similarity.
    
    Args:
        chunks_data (List[Dict]): Processed chunks with embeddings
        max_connections (int): Maximum connections to return
        
    Returns:
        List[Dict[str, Any]]: Connection objects with structure:
        {
            "chunk_a_index": int,
            "chunk_b_index": int,
            "similarity_score": float,      # 0.0 to 1.0
            "novelty_score": float,         # 0.0 to 1.0
            "connection_type": str,         # "semantic", "entity", "topic"
            "explanation": str,             # Human-readable explanation
            "shared_entities": List[str],   # Common entities
            "shared_topics": List[str],     # Common topics
            "innovation_potential": float   # 0.0 to 1.0
        }
        
    Algorithm:
        1. Compute pairwise cosine similarity between all chunks
        2. Filter connections with similarity between 0.3 and 0.9
        3. Calculate novelty based on entity/topic diversity
        4. Rank by innovation potential (similarity + novelty)
        5. Generate explanations for top connections
        
    Innovation Potential Scoring:
        - High similarity + High novelty = High innovation potential
        - Same concepts in different contexts = Most valuable
        - Pure similarity without novelty = Less valuable
    """
    result = connection_discovery_engine.discover_connections(chunks_data, max_connections)
    
    if not result.success:
        logger.error(f"Connection discovery failed: {result.error_message}")
        return []
    
    # Convert SemanticConnection objects to dictionaries for SmolAgent compatibility
    connections_dict = []
    for connection in result.connections:
        connections_dict.append({
            "connection_id": connection.connection_id,
            "chunk_a_index": connection.chunk_a_index,
            "chunk_b_index": connection.chunk_b_index,
            "similarity_score": connection.similarity_score,
            "novelty_score": connection.novelty_score,
            "innovation_potential": connection.innovation_potential,
            "connection_type": connection.connection_type.value,
            "connection_strength": connection.connection_strength.value,
            "innovation_level": connection.innovation_level.value,
            "explanation": connection.explanation,
            "shared_entities": connection.shared_entities,
            "shared_topics": connection.shared_topics,
            "shared_keywords": connection.shared_keywords,
            "shared_concepts": connection.shared_concepts,
            "confidence_score": connection.confidence_score,
            "quality_score": connection.quality_score,
            "created_at": connection.created_at.isoformat()
        })
    
    return connections_dict

def get_connection_discovery_info() -> Dict[str, Any]:
    """Get information about connection discovery capabilities."""
    return {
        "max_connections": config.connection.max_connections,
        "similarity_threshold_min": config.connection.similarity_threshold_min,
        "similarity_threshold_max": config.connection.similarity_threshold_max,
        "novelty_weight": config.connection.novelty_weight,
        "similarity_weight": config.connection.similarity_weight,
        "connection_types": [ct.value for ct in ConnectionType],
        "innovation_levels": [il.value for il in InnovationPotential],
        "connection_strengths": [cs.value for cs in ConnectionStrength]
    }