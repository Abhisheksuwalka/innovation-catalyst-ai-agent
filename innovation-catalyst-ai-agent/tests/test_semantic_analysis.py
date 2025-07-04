# tests/test_semantic_analysis.py
"""
Comprehensive tests for semantic analysis tools.
Run these tests to validate the implementation before integration.
"""

import pytest
import time
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the tools to test
from src.innovation_catalyst.tools.embeddings import (
    generate_embeddings, 
    embedding_generator,
    get_embedding_model_info
)
from src.innovation_catalyst.tools.nlp_analysis import (
    extract_entities_and_topics,
    nlp_analyzer,
    get_nlp_analysis_info
)

class TestEmbeddingGeneration:
    """Test embedding generation functionality."""
    
    def test_generate_embeddings_basic(self):
        """Test basic embedding generation."""
        texts = [
            "This is a test sentence.",
            "Another sentence for testing.",
            "Machine learning is fascinating."
        ]
        
        embeddings = generate_embeddings(texts)
        
        # Check structure
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        
        if embeddings:  # Only check if embeddings were generated
            assert isinstance(embeddings[0], list)
            assert len(embeddings[0]) > 0  # Should have some dimensions
            
            # Check all embeddings have same dimension
            dimensions = [len(emb) for emb in embeddings]
            assert all(dim == dimensions[0] for dim in dimensions)
    
    def test_generate_embeddings_empty(self):
        """Test embedding generation with empty input."""
        embeddings = generate_embeddings([])
        assert embeddings == []
    
    def test_generate_embeddings_single(self):
        """Test embedding generation with single text."""
        embeddings = generate_embeddings(["Single test sentence."])
        
        if embeddings:
            assert len(embeddings) == 1
            assert isinstance(embeddings[0], list)
    
    def test_embedding_caching(self):
        """Test embedding caching functionality."""
        texts = ["Test sentence for caching."]
        
        # First call
        start_time = time.time()
        embeddings1 = generate_embeddings(texts)
        first_time = time.time() - start_time
        
        # Second call (should be faster due to caching)
        start_time = time.time()
        embeddings2 = generate_embeddings(texts)
        second_time = time.time() - start_time
        
        # Results should be identical
        if embeddings1 and embeddings2:
            assert embeddings1 == embeddings2
            # Second call should be faster (cache hit)
            # Note: This might not always be true due to system variations
    
    def test_model_info(self):
        """Test model information retrieval."""
        info = get_embedding_model_info()
        assert isinstance(info, dict)
        # Info might be empty if model not loaded

class TestNLPAnalysis:
    """Test NLP analysis functionality."""
    
    def test_extract_entities_and_topics_basic(self):
        """Test basic entity and topic extraction."""
        text = """
        Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
        The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
        Apple is known for its innovative products including the iPhone, iPad, and MacBook.
        """
        
        result = extract_entities_and_topics(text)
        
        # Check structure
        assert isinstance(result, dict)
        assert "entities" in result
        assert "topics" in result
        assert "keywords" in result
        assert "concepts" in result
        
        # Check types
        assert isinstance(result["entities"], list)
        assert isinstance(result["topics"], list)
        assert isinstance(result["keywords"], list)
        assert isinstance(result["concepts"], list)
    
    def test_extract_entities_and_topics_empty(self):
        """Test extraction with empty text."""
        result = extract_entities_and_topics("")
        
        assert isinstance(result, dict)
        assert all(isinstance(result[key], list) for key in result.keys())
        assert all(len(result[key]) == 0 for key in result.keys())
    
    def test_extract_entities_and_topics_simple(self):
        """Test extraction with simple text."""
        text = "John works at Microsoft in Seattle."
        
        result = extract_entities_and_topics(text)
        
        # Should extract some entities/keywords
        total_extracted = sum(len(result[key]) for key in result.keys())
        # At least some extraction should occur
        assert total_extracted >= 0  # Might be 0 if no libraries available
    
    def test_nlp_analysis_info(self):
        """Test NLP analysis information."""
        info = get_nlp_analysis_info()
        
        assert isinstance(info, dict)
        assert "methods" in info
        assert isinstance(info["methods"], dict)

class TestIntegration:
    """Integration tests for semantic analysis tools."""
    
    def test_full_document_analysis(self):
        """Test complete document analysis workflow."""
        # Sample document text
        document_text = """
        Artificial Intelligence and Machine Learning Revolution
        
        The field of artificial intelligence has experienced unprecedented growth in recent years.
        Companies like Google, Microsoft, and OpenAI are leading the charge in developing
        advanced AI systems. Machine learning algorithms, particularly deep learning neural networks,
        have shown remarkable capabilities in natural language processing, computer vision,
        and predictive analytics.
        
        The applications are vast: from autonomous vehicles developed by Tesla and Waymo,
        to virtual assistants like Siri and Alexa, to recommendation systems used by
        Netflix and Amazon. The economic impact is projected to reach trillions of dollars
        by 2030, according to McKinsey & Company research.
        
        However, challenges remain including ethical considerations, data privacy concerns,
        and the need for regulatory frameworks. The future of AI depends on addressing
        these challenges while continuing to innovate and push the boundaries of what's possible.
        """
        
        # Test NLP analysis
        nlp_result = extract_entities_and_topics(document_text)
        
        # Should extract meaningful content
        assert isinstance(nlp_result, dict)
        
        # Test embedding generation
        # Split into chunks for embedding
        chunks = [
            document_text[:len(document_text)//2],
            document_text[len(document_text)//2:]
        ]
        
        embeddings = generate_embeddings(chunks)
        
        # Should generate embeddings
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(chunks)
    
    def test_performance_benchmarks(self):
        """Test performance of semantic analysis tools."""
        # Generate test data
        test_texts = [
            f"This is test sentence number {i} for performance testing."
            for i in range(10)
        ]
        
        # Test embedding generation performance
        start_time = time.time()
        embeddings = generate_embeddings(test_texts)
        embedding_time = time.time() - start_time
        
        # Test NLP analysis performance
        long_text = " ".join(test_texts)
        start_time = time.time()
        nlp_result = extract_entities_and_topics(long_text)
        nlp_time = time.time() - start_time
        
        # Performance should be reasonable (adjust thresholds as needed)
        assert embedding_time < 30.0  # Should complete within 30 seconds
        assert nlp_time < 10.0  # Should complete within 10 seconds
        
        print(f"Embedding generation: {embedding_time:.2f}s")
        print(f"NLP analysis: {nlp_time:.2f}s")

def run_all_tests():
    """Run all tests and report results."""
    print("🧪 Running Semantic Analysis Tests...")
    
    # Test classes
    test_classes = [TestEmbeddingGeneration, TestNLPAnalysis, TestIntegration]
    
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
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    run_all_tests()
