# tests/test_connection_discovery.py
"""
Comprehensive tests for connection discovery engine.
Run these tests to validate the implementation before integration.
"""

import pytest
import time
import numpy as np
from typing import List, Dict, Any

import sys
import os

# Add the parent directory (where src is located) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the tools to test
from src.innovation_catalyst.tools.connection_discovery import (
    discover_semantic_connections,
    connection_discovery_engine,
    get_connection_discovery_info
)
from src.innovation_catalyst.tools.similarity_calculator import similarity_calculator
from src.innovation_catalyst.tools.novelty_scorer import novelty_scorer
from src.innovation_catalyst.models.document import TextChunk

# Import the models and enums
from src.innovation_catalyst.models.connections import (
    ConnectionType, 
    InnovationPotential,
    ConnectionStrength
)

class TestConnectionDiscovery:
    """Test connection discovery functionality."""
    
    def create_sample_chunks_data(self) -> List[Dict[str, Any]]:
        """Create sample chunks data for testing."""
        return [
            {
                "chunk_id": "chunk_1",
                "content": "Artificial intelligence is transforming healthcare through machine learning algorithms.",
                "chunk_index": 0,
                "word_count": 10,
                "sentence_count": 1,
                "start_position": 0,
                "end_position": 100,
                "source_document_id": "doc_1",
                "entities": ["artificial intelligence", "healthcare", "machine learning"],
                "topics": ["AI", "healthcare", "technology"],
                "keywords": ["artificial", "intelligence", "healthcare", "machine", "learning"],
                "concepts": ["AI transformation", "healthcare innovation"],
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 76  # 384 dimensions
            },
            {
                "chunk_id": "chunk_2", 
                "content": "Machine learning models are revolutionizing medical diagnosis and treatment planning.",
                "chunk_index": 1,
                "word_count": 11,
                "sentence_count": 1,
                "start_position": 100,
                "end_position": 200,
                "source_document_id": "doc_1",
                "entities": ["machine learning", "medical diagnosis", "treatment"],
                "topics": ["machine learning", "medical", "diagnosis"],
                "keywords": ["machine", "learning", "medical", "diagnosis", "treatment"],
                "concepts": ["ML in medicine", "diagnostic innovation"],
                "embedding": [0.15, 0.25, 0.35, 0.45, 0.55] * 76  # Similar but different
            },
            {
                "chunk_id": "chunk_3",
                "content": "Financial markets are experiencing volatility due to economic uncertainty.",
                "chunk_index": 2,
                "word_count": 9,
                "sentence_count": 1,
                "start_position": 200,
                "end_position": 300,
                "source_document_id": "doc_2",
                "entities": ["financial markets", "economic uncertainty"],
                "topics": ["finance", "markets", "economy"],
                "keywords": ["financial", "markets", "volatility", "economic", "uncertainty"],
                "concepts": ["market volatility", "economic instability"],
                "embedding": [0.8, 0.7, 0.6, 0.5, 0.4] * 76  # Very different
            }
        ]
    
    def test_discover_connections_basic(self):
        """Test basic connection discovery."""
        chunks_data = self.create_sample_chunks_data()
        
        connections = discover_semantic_connections(chunks_data, max_connections=10)
        
        # Check structure
        assert isinstance(connections, list)
        
        if connections:  # Only check if connections were found
            connection = connections[0]
            assert isinstance(connection, dict)
            
            # Check required fields
            required_fields = [
                "chunk_a_index", "chunk_b_index", "similarity_score", 
                "novelty_score", "innovation_potential", "connection_type",
                "explanation", "shared_entities", "shared_topics"
            ]
            
            for field in required_fields:
                assert field in connection
            
            # Check value ranges
            assert 0 <= connection["similarity_score"] <= 1
            assert 0 <= connection["novelty_score"] <= 1
            assert 0 <= connection["innovation_potential"] <= 1
            assert connection["chunk_a_index"] != connection["chunk_b_index"]
    
    def test_discover_connections_empty(self):
        """Test connection discovery with empty input."""
        connections = discover_semantic_connections([])
        assert connections == []
    
    def test_discover_connections_single_chunk(self):
        """Test connection discovery with single chunk."""
        chunks_data = self.create_sample_chunks_data()[:1]
        connections = discover_semantic_connections(chunks_data)
        assert connections == []  # Can't create connections with single chunk
    
    def test_connection_types(self):
        """Test different connection types are detected."""
        chunks_data = self.create_sample_chunks_data()
        connections = discover_semantic_connections(chunks_data)
        
        if connections:
            # Check that connection types are valid
            valid_types = [ct.value for ct in ConnectionType]
            for connection in connections:
                assert connection["connection_type"] in valid_types
    
    def test_similarity_thresholds(self):
        """Test similarity threshold filtering."""
        chunks_data = self.create_sample_chunks_data()
        connections = discover_semantic_connections(chunks_data)
        
        # All connections should be within similarity thresholds
        for connection in connections:
            similarity = connection["similarity_score"]
            # Should be within configured thresholds (0.3 to 0.9 by default)
            assert 0.0 <= similarity <= 1.0
    
    def test_max_connections_limit(self):
        """Test max connections limit is respected."""
        chunks_data = self.create_sample_chunks_data()
        max_connections = 2
        
        connections = discover_semantic_connections(chunks_data, max_connections=max_connections)
        
        assert len(connections) <= max_connections
    
    def test_connection_explanation(self):
        """Test that connections have meaningful explanations."""
        chunks_data = self.create_sample_chunks_data()
        connections = discover_semantic_connections(chunks_data)
        
        for connection in connections:
            assert isinstance(connection["explanation"], str)
            assert len(connection["explanation"]) > 0
    
    def test_shared_elements(self):
        """Test shared elements detection."""
        chunks_data = self.create_sample_chunks_data()
        connections = discover_semantic_connections(chunks_data)
        
        for connection in connections:
            # Check shared elements structure
            assert isinstance(connection["shared_entities"], list)
            assert isinstance(connection["shared_topics"], list)
            assert isinstance(connection["shared_keywords"], list)
            assert isinstance(connection["shared_concepts"], list)

"""
Comprehensive tests for connection discovery engine.
Run these tests to validate the implementation before integration.
"""
class TestConnectionDiscovery:
    """Test connection discovery functionality."""
    
    def create_sample_chunks_data(self) -> List[Dict[str, Any]]:
        """Create sample chunks data for testing."""
        return [
            {
                "chunk_id": "chunk_1",
                "content": "Artificial intelligence is transforming healthcare through machine learning algorithms.",
                "chunk_index": 0,
                "word_count": 10,
                "sentence_count": 1,
                "start_position": 0,
                "end_position": 100,
                "source_document_id": "doc_1",
                "entities": ["artificial intelligence", "healthcare", "machine learning"],
                "topics": ["AI", "healthcare", "technology"],
                "keywords": ["artificial", "intelligence", "healthcare", "machine", "learning"],
                "concepts": ["AI transformation", "healthcare innovation"],
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 77  # 385 dimensions (adjusted)
            },
            {
                "chunk_id": "chunk_2", 
                "content": "Machine learning models are revolutionizing medical diagnosis and treatment planning.",
                "chunk_index": 1,
                "word_count": 11,
                "sentence_count": 1,
                "start_position": 100,
                "end_position": 200,
                "source_document_id": "doc_1",
                "entities": ["machine learning", "medical diagnosis", "treatment"],
                "topics": ["machine learning", "medical", "diagnosis"],
                "keywords": ["machine", "learning", "medical", "diagnosis", "treatment"],
                "concepts": ["ML in medicine", "diagnostic innovation"],
                "embedding": [0.15, 0.25, 0.35, 0.45, 0.55] * 77  # Similar but different
            },
            {
                "chunk_id": "chunk_3",
                "content": "Financial markets are experiencing volatility due to economic uncertainty.",
                "chunk_index": 2,
                "word_count": 9,
                "sentence_count": 1,
                "start_position": 200,
                "end_position": 300,
                "source_document_id": "doc_2",
                "entities": ["financial markets", "economic uncertainty"],
                "topics": ["finance", "markets", "economy"],
                "keywords": ["financial", "markets", "volatility", "economic", "uncertainty"],
                "concepts": ["market volatility", "economic instability"],
                "embedding": [0.8, 0.7, 0.6, 0.5, 0.4] * 77  # Very different
            }
        ]
    
    def test_discover_connections_basic(self):
        """Test basic connection discovery."""
        chunks_data = self.create_sample_chunks_data()
        
        connections = discover_semantic_connections(chunks_data, max_connections=10)
        
        # Check structure
        assert isinstance(connections, list)
        
        if connections:  # Only check if connections were found
            connection = connections[0]
            assert isinstance(connection, dict)
            
            # Check required fields
            required_fields = [
                "chunk_a_index", "chunk_b_index", "similarity_score", 
                "novelty_score", "innovation_potential", "connection_type",
                "explanation", "shared_entities", "shared_topics"
            ]
            
            for field in required_fields:
                assert field in connection
            
            # Check value ranges
            assert 0 <= connection["similarity_score"] <= 1
            assert 0 <= connection["novelty_score"] <= 1
            assert 0 <= connection["innovation_potential"] <= 1
            assert connection["chunk_a_index"] != connection["chunk_b_index"]
    
    def test_discover_connections_empty(self):
        """Test connection discovery with empty input."""
        connections = discover_semantic_connections([])
        assert connections == []
    
    def test_discover_connections_single_chunk(self):
        """Test connection discovery with single chunk."""
        chunks_data = self.create_sample_chunks_data()[:1]
        connections = discover_semantic_connections(chunks_data)
        assert connections == []  # Can't create connections with single chunk
    
    def test_connection_types(self):
        """Test different connection types are detected."""
        chunks_data = self.create_sample_chunks_data()
        connections = discover_semantic_connections(chunks_data)
        
        if connections:
            # Check that connection types are valid
            valid_types = [ct.value for ct in ConnectionType]
            for connection in connections:
                assert connection["connection_type"] in valid_types
    
    def test_similarity_thresholds(self):
        """Test similarity threshold filtering."""
        chunks_data = self.create_sample_chunks_data()
        connections = discover_semantic_connections(chunks_data)
        
        # All connections should be within similarity thresholds
        for connection in connections:
            similarity = connection["similarity_score"]
            # Should be within configured thresholds (0.3 to 0.9 by default)
            assert 0.0 <= similarity <= 1.0
    
    def test_max_connections_limit(self):
        """Test max connections limit is respected."""
        chunks_data = self.create_sample_chunks_data()
        max_connections = 2
        
        connections = discover_semantic_connections(chunks_data, max_connections=max_connections)
        
        assert len(connections) <= max_connections
    
    def test_connection_explanation(self):
        """Test that connections have meaningful explanations."""
        chunks_data = self.create_sample_chunks_data()
        connections = discover_semantic_connections(chunks_data)
        
        for connection in connections:
            assert isinstance(connection["explanation"], str)
            assert len(connection["explanation"]) > 0
    
    def test_shared_elements(self):
        """Test shared elements detection."""
        chunks_data = self.create_sample_chunks_data()
        connections = discover_semantic_connections(chunks_data)
        
        for connection in connections:
            # Check shared elements structure
            assert isinstance(connection["shared_entities"], list)
            assert isinstance(connection["shared_topics"], list)
            assert isinstance(connection["shared_keywords"], list)
            assert isinstance(connection["shared_concepts"], list)

class TestSimilarityCalculator:
    """Test similarity calculation functionality."""
    
    def create_sample_embeddings(self) -> List[List[float]]:
        """Create sample embeddings for testing."""
        return [
            [0.1, 0.2, 0.3, 0.4, 0.5] * 77,  # 385 dimensions
            [0.15, 0.25, 0.35, 0.45, 0.55] * 77,  # Similar to first
            [0.8, 0.7, 0.6, 0.5, 0.4] * 77,  # Different from others
            [0.2, 0.3, 0.4, 0.5, 0.6] * 77   # Another variation
        ]
    
    def create_sample_chunks(self) -> List[TextChunk]:
        """Create sample TextChunk objects for testing."""
        return [
            TextChunk(
                content="AI is transforming healthcare",
                chunk_index=0,
                word_count=4,
                sentence_count=1,
                start_position=0,
                end_position=30,
                source_document_id="doc_1",
                entities=["AI", "healthcare"],
                topics=["technology", "healthcare"],
                keywords=["AI", "transforming", "healthcare"],
                concepts=["AI transformation"]
            ),
            TextChunk(
                content="Machine learning improves medical diagnosis",
                chunk_index=1,
                word_count=5,
                sentence_count=1,
                start_position=30,
                end_position=70,
                source_document_id="doc_1",
                entities=["machine learning", "medical diagnosis"],
                topics=["machine learning", "medical"],
                keywords=["machine", "learning", "medical", "diagnosis"],
                concepts=["ML in medicine"]
            ),
            TextChunk(
                content="Financial markets show volatility",
                chunk_index=2,
                word_count=4,
                sentence_count=1,
                start_position=70,
                end_position=100,
                source_document_id="doc_2",
                entities=["financial markets"],
                topics=["finance", "markets"],
                keywords=["financial", "markets", "volatility"],
                concepts=["market volatility"]
            )
        ]
    
    def test_similarity_matrix_calculation(self):
        """Test similarity matrix calculation."""
        embeddings = self.create_sample_embeddings()
        
        similarity_matrix = similarity_calculator.calculate_similarity_matrix(embeddings)
        
        # Check matrix properties
        assert isinstance(similarity_matrix, np.ndarray)
        assert similarity_matrix.shape == (len(embeddings), len(embeddings))
        
        # Check diagonal is 1.0 (self-similarity)
        for i in range(len(embeddings)):
            assert abs(similarity_matrix[i, i] - 1.0) < 1e-6
        
        # Check symmetry
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                assert abs(similarity_matrix[i, j] - similarity_matrix[j, i]) < 1e-6
    
    def test_pairwise_similarities(self):
        """Test pairwise similarity calculation."""
        embeddings = self.create_sample_embeddings()
        chunks = self.create_sample_chunks()
        
        similarities = similarity_calculator.calculate_pairwise_similarities(
            embeddings, chunks, use_parallel=False
        )
        
        # Check structure
        assert isinstance(similarities, list)
        
        for i, j, metrics in similarities:
            assert isinstance(i, int)
            assert isinstance(j, int)
            assert i != j
            assert 0 <= i < len(embeddings)
            assert 0 <= j < len(embeddings)
            
            # Check metrics
            assert hasattr(metrics, 'cosine_similarity')
            assert hasattr(metrics, 'euclidean_distance')
            assert hasattr(metrics, 'jaccard_similarity')
            
            # Check value ranges
            assert 0 <= metrics.cosine_similarity <= 1
            assert metrics.euclidean_distance >= 0
            assert 0 <= metrics.jaccard_similarity <= 1
    
    def test_empty_embeddings(self):
        """Test handling of empty embeddings."""
        similarities = similarity_calculator.calculate_pairwise_similarities([], [])
        assert similarities == []
        
        similarity_matrix = similarity_calculator.calculate_similarity_matrix([])
        assert similarity_matrix.size == 0
    
    def test_single_embedding(self):
        """Test handling of single embedding."""
        embeddings = self.create_sample_embeddings()[:1]
        chunks = self.create_sample_chunks()[:1]
        
        similarities = similarity_calculator.calculate_pairwise_similarities(embeddings, chunks)
        assert similarities == []  # No pairs possible with single embedding

class TestNoveltyScorer:
    """Test novelty scoring functionality."""
    
    def create_sample_chunks_for_novelty(self) -> List[TextChunk]:
        """Create sample chunks with different novelty characteristics."""
        return [
            TextChunk(
                content="AI revolutionizes healthcare diagnostics",
                chunk_index=0,
                word_count=4,
                sentence_count=1,
                start_position=0,
                end_position=35,
                source_document_id="doc_1",
                entities=["AI", "healthcare"],
                topics=["technology", "healthcare"],
                keywords=["AI", "revolutionizes", "healthcare", "diagnostics"],
                concepts=["AI revolution", "healthcare diagnostics"]
            ),
            TextChunk(
                content="Blockchain technology transforms financial services",
                chunk_index=1,
                word_count=5,
                sentence_count=1,
                start_position=35,
                end_position=80,
                source_document_id="doc_2",
                entities=["blockchain", "financial services"],
                topics=["blockchain", "finance"],
                keywords=["blockchain", "technology", "transforms", "financial", "services"],
                concepts=["blockchain transformation", "financial innovation"]
            )
        ]
    
    def test_novelty_calculation(self):
        """Test novelty score calculation."""
        chunks = self.create_sample_chunks_for_novelty()
        chunk1, chunk2 = chunks[0], chunks[1]
        similarity_score = 0.4  # Moderate similarity
        
        novelty_metrics = novelty_scorer.calculate_novelty_score(
            chunk1, chunk2, similarity_score, chunks
        )
        
        # Check novelty metrics structure
        assert hasattr(novelty_metrics, 'entity_diversity')
        assert hasattr(novelty_metrics, 'topic_diversity')
        assert hasattr(novelty_metrics, 'concept_diversity')
        assert hasattr(novelty_metrics, 'semantic_distance')
        assert hasattr(novelty_metrics, 'contextual_novelty')
        assert hasattr(novelty_metrics, 'cross_domain_score')
        
        # Check value ranges
        assert 0 <= novelty_metrics.entity_diversity <= 1
        assert 0 <= novelty_metrics.topic_diversity <= 1
        assert 0 <= novelty_metrics.concept_diversity <= 1
        assert 0 <= novelty_metrics.semantic_distance <= 1
        assert 0 <= novelty_metrics.contextual_novelty <= 1
        assert 0 <= novelty_metrics.cross_domain_score <= 1
        
        # Check combined novelty
        combined_novelty = novelty_metrics.get_combined_novelty()
        assert 0 <= combined_novelty <= 1
    
    def test_innovation_potential_calculation(self):
        """Test innovation potential calculation."""
        chunks = self.create_sample_chunks_for_novelty()
        chunk1, chunk2 = chunks[0], chunks[1]
        similarity_score = 0.5
        
        novelty_metrics = novelty_scorer.calculate_novelty_score(
            chunk1, chunk2, similarity_score, chunks
        )
        
        innovation_score, innovation_level = novelty_scorer.calculate_innovation_potential(
            similarity_score, novelty_metrics, chunk1, chunk2
        )
        
        # Check innovation score
        assert 0 <= innovation_score <= 1
        
        # Check innovation level
        assert innovation_level in [
            InnovationPotential.LOW,
            InnovationPotential.MEDIUM,
            InnovationPotential.HIGH,
            InnovationPotential.BREAKTHROUGH
        ]
    
    def test_cross_domain_detection(self):
        """Test cross-domain connection detection."""
        chunks = self.create_sample_chunks_for_novelty()
        chunk1, chunk2 = chunks[0], chunks[1]  # AI+healthcare vs blockchain+finance
        
        novelty_metrics = novelty_scorer.calculate_novelty_score(
            chunk1, chunk2, 0.3, chunks
        )
        
        # Should detect cross-domain connection (tech+healthcare vs blockchain+finance)
        assert novelty_metrics.cross_domain_score > 0

class TestIntegration:
    """Integration tests for connection discovery engine."""
    
    def test_full_connection_discovery_workflow(self):
        """Test complete connection discovery workflow."""
        # Create comprehensive test data
        chunks_data = [
            {
                "chunk_id": f"chunk_{i}",
                "content": f"Test content {i} with various topics and entities.",
                "chunk_index": i,
                "word_count": 8,
                "sentence_count": 1,
                "start_position": i * 50,
                "end_position": (i + 1) * 50,
                "source_document_id": f"doc_{i // 2}",
                "entities": [f"entity_{i}", f"entity_{i+1}"],
                "topics": [f"topic_{i}", f"topic_{i+1}"],
                "keywords": [f"keyword_{i}", f"keyword_{i+1}", "test", "content"],
                "concepts": [f"concept_{i}"],
                "embedding": [0.1 + i * 0.1, 0.2 + i * 0.1, 0.3 + i * 0.1] * 128  # 384 dimensions
            }
            for i in range(5)
        ]
        
        # Test connection discovery
        connections = discover_semantic_connections(chunks_data, max_connections=10)
        
        # Validate results
        assert isinstance(connections, list)
        assert len(connections) <= 10
        
        # Test each connection
        for connection in connections:
            # Structure validation
            assert isinstance(connection, dict)
            assert "connection_id" in connection
            assert "explanation" in connection
            assert "innovation_potential" in connection
            
            # Value validation
            assert 0 <= connection["similarity_score"] <= 1
            assert 0 <= connection["novelty_score"] <= 1
            assert 0 <= connection["innovation_potential"] <= 1
            
            # Explanation validation
            assert isinstance(connection["explanation"], str)
            assert len(connection["explanation"]) > 0
    
    def test_performance_benchmarks(self):
        """Test performance of connection discovery."""
        # Create larger dataset for performance testing
        chunks_data = []
        for i in range(20):  # 20 chunks = 190 possible connections
            chunks_data.append({
                "chunk_id": f"chunk_{i}",
                "content": f"Performance test content {i} with various elements for testing.",
                "chunk_index": i,
                "word_count": 10,
                "sentence_count": 1,
                "start_position": i * 60,
                "end_position": (i + 1) * 60,
                "source_document_id": f"doc_{i // 5}",
                "entities": [f"entity_{i}", f"entity_{(i+1) % 20}"],
                "topics": [f"topic_{i % 5}", f"topic_{(i+2) % 5}"],
                "keywords": [f"keyword_{i}", "performance", "test", "content"],
                "concepts": [f"concept_{i % 3}"],
                "embedding": [np.random.random() for _ in range(384)]
            })
        
        # Time the connection discovery
        start_time = time.time()
        connections = discover_semantic_connections(chunks_data, max_connections=15)
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert processing_time < 30.0  # Should complete within 30 seconds
        assert isinstance(connections, list)
        assert len(connections) <= 15
        
        print(f"Connection discovery performance: {processing_time:.2f}s for {len(chunks_data)} chunks")
    
    def test_connection_discovery_info(self):
        """Test connection discovery information retrieval."""
        info = get_connection_discovery_info()
        
        assert isinstance(info, dict)
        assert "max_connections" in info
        assert "similarity_threshold_min" in info
        assert "similarity_threshold_max" in info
        assert "connection_types" in info
        assert "innovation_levels" in info
        
        # Validate connection types
        assert isinstance(info["connection_types"], list)
        assert len(info["connection_types"]) > 0
        
        # Validate innovation levels
        assert isinstance(info["innovation_levels"], list)
        assert len(info["innovation_levels"]) > 0

def run_all_connection_tests():
    """Run all connection discovery tests and report results."""
    print("🔗 Running Connection Discovery Tests...")
    
    # Test classes
    test_classes = [
        TestConnectionDiscovery,
        TestSimilarityCalculator, 
        TestNoveltyScorer,
        TestIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}...")
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"  ✅ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ❌ {method_name}: {str(e)}")
    
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("🎉 All connection discovery tests passed!")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    run_all_connection_tests()
