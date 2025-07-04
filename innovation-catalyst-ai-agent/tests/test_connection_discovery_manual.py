# scripts/test_connection_discovery.py
"""
Manual testing script for connection discovery engine.
Run this to validate the complete workflow.
"""
import sys
import os

# Add the parent directory (where src is located) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.innovation_catalyst.tools.connection_discovery import discover_semantic_connections
from src.innovation_catalyst.tools.innovation_synthesis import generate_innovation_synthesis

def test_complete_workflow():
    """Test the complete connection discovery and synthesis workflow."""
    print("🔗 Testing Connection Discovery Engine...")
    
    # Sample data
    chunks_data = [
        {
            "chunk_id": "chunk_1",
            "content": "Artificial intelligence is revolutionizing healthcare through predictive analytics and personalized treatment plans.",
            "chunk_index": 0,
            "word_count": 14,
            "sentence_count": 1,
            "start_position": 0,
            "end_position": 100,
            "source_document_id": "doc_1",
            "entities": ["artificial intelligence", "healthcare", "predictive analytics"],
            "topics": ["AI", "healthcare", "analytics"],
            "keywords": ["artificial", "intelligence", "healthcare", "predictive", "analytics"],
            "concepts": ["AI revolution", "healthcare transformation"],
            "embedding": [0.1, 0.2, 0.3] * 128  # 384 dimensions
        },
        {
            "chunk_id": "chunk_2",
            "content": "Machine learning algorithms are enabling breakthrough discoveries in drug development and clinical trials.",
            "chunk_index": 1,
            "word_count": 14,
            "sentence_count": 1,
            "start_position": 100,
            "end_position": 200,
            "source_document_id": "doc_1",
            "entities": ["machine learning", "drug development", "clinical trials"],
            "topics": ["machine learning", "pharmaceuticals", "research"],
            "keywords": ["machine", "learning", "algorithms", "drug", "development"],
            "concepts": ["ML in pharma", "clinical innovation"],
            "embedding": [0.15, 0.25, 0.35] * 128  # Similar to first
        },
        {
            "chunk_id": "chunk_3",
            "content": "Blockchain technology is transforming supply chain management through transparent and immutable record keeping.",
            "chunk_index": 2,
            "word_count": 14,
            "sentence_count": 1,
            "start_position": 200,
            "end_position": 300,
            "source_document_id": "doc_2",
            "entities": ["blockchain", "supply chain", "record keeping"],
            "topics": ["blockchain", "supply chain", "transparency"],
            "keywords": ["blockchain", "technology", "supply", "chain", "transparent"],
            "concepts": ["blockchain transformation", "supply chain innovation"],
            "embedding": [0.8, 0.7, 0.6] * 128  # Different domain
        }
    ]
    
    # Test connection discovery
    print("1. Testing connection discovery...")
    connections = discover_semantic_connections(chunks_data, max_connections=5)
    
    print(f"   ✅ Found {len(connections)} connections")
    
    if connections:
        for i, conn in enumerate(connections):
            print(f"   Connection {i+1}:")
            print(f"     - Similarity: {conn['similarity_score']:.3f}")
            print(f"     - Novelty: {conn['novelty_score']:.3f}")
            print(f"     - Innovation: {conn['innovation_potential']:.3f}")
            print(f"     - Type: {conn['connection_type']}")
            print(f"     - Explanation: {conn['explanation'][:100]}...")
    
    # Test synthesis generation
    if connections:
        print("\n2. Testing innovation synthesis...")
        synthesis = generate_innovation_synthesis(connections)
        
        print(f"   ✅ Generated synthesis with innovation score: {synthesis['innovation_score']:.3f}")
        print(f"   ✅ {len(synthesis['actionable_steps'])} actionable steps identified")
        print(f"   ✅ {len(synthesis['potential_applications'])} potential applications found")
        print(f"   ✅ Technical feasibility: {synthesis['technical_feasibility']:.3f}")
        
        print("\n   Sample synthesis text:")
        print(f"   {synthesis['synthesis_text'][:200]}...")
        
        print("\n   Sample actionable steps:")
        for i, step in enumerate(synthesis['actionable_steps'][:3]):
            print(f"   {i+1}. {step}")
    
    print("\n🎉 Connection discovery engine test completed successfully!")

if __name__ == "__main__":
    test_complete_workflow()