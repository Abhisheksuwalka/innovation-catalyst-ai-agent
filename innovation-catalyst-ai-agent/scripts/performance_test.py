# scripts/performance_test.py
"""
Performance testing for connection discovery engine.
"""

import time
import numpy as np
from src.innovation_catalyst.tools.connection_discovery import discover_semantic_connections

def performance_test():
    """Test performance with various data sizes."""
    print("⚡ Performance Testing Connection Discovery...")
    
    sizes = [5, 10, 20, 50]
    
    for size in sizes:
        print(f"\nTesting with {size} chunks...")
        
        # Generate test data
        chunks_data = []
        for i in range(size):
            chunks_data.append({
                "chunk_id": f"chunk_{i}",
                "content": f"Test content {i} with various topics and entities for performance testing.",
                "chunk_index": i,
                "word_count": 12,
                "sentence_count": 1,
                "start_position": i * 80,
                "end_position": (i + 1) * 80,
                "source_document_id": f"doc_{i // 10}",
                "entities": [f"entity_{i}", f"entity_{(i+1) % size}"],
                "topics": [f"topic_{i % 5}", f"topic_{(i+2) % 5}"],
                "keywords": [f"keyword_{i}", "test", "performance", "content"],
                "concepts": [f"concept_{i % 3}"],
                "embedding": np.random.random(384).tolist()
            })
        
        # Time the discovery
        start_time = time.time()
        connections = discover_semantic_connections(chunks_data, max_connections=10)
        processing_time = time.time() - start_time
        
        possible_connections = size * (size - 1) // 2
        
        print(f"   Processed {size} chunks ({possible_connections} possible connections)")
        print(f"   Found {len(connections)} connections")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Rate: {size / processing_time:.1f} chunks/second")

if __name__ == "__main__":
    performance_test()
