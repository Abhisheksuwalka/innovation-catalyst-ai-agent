"""
Comprehensive tests for the Innovation Catalyst Agent.
Tests the complete agent workflow and integration with API-based models.
"""

import pytest
import json
import os
import sys
import time
from unittest.mock import patch, MagicMock

# Add project root to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from src.innovation_catalyst.agents.innovation_agent import (
    InnovationCatalystAgent,
    get_innovation_agent,
    process_documents_for_innovation,
    reset_innovation_agent
)

class TestInnovationAgent:
    """Test the main Innovation Catalyst Agent with API integration."""
    
    def setup_method(self):
        """Setup test environment."""
        # Reset global agent for clean tests
        reset_innovation_agent()
        
        # Mock environment variable if not set
        if not os.getenv("HF_TOKEN_TO_CALL_LLM"):
            os.environ["HF_TOKEN_TO_CALL_LLM"] = "test_token_for_testing"
        
        try:
            self.agent = InnovationCatalystAgent()
        except Exception as e:
            pytest.skip(f"Agent initialization failed: {e}")
        
        # Create sample documents with proper string content
        self.sample_docs = [
            {
                "name": "ai_healthcare.txt",
                "content": "Artificial intelligence is revolutionizing healthcare through predictive analytics, personalized treatment plans, and automated diagnosis systems. Machine learning algorithms can analyze medical images with unprecedented accuracy, enabling early detection of diseases and improving patient outcomes.",
            },
            {
                "name": "blockchain_finance.txt", 
                "content": "Blockchain technology is transforming financial services by enabling secure, transparent, and decentralized transactions. Smart contracts automate complex financial processes and reduce intermediary costs, while providing immutable audit trails for regulatory compliance.",
            }
        ]
    
    def test_agent_initialization(self):
        """Test agent initialization with API configuration."""
        assert self.agent is not None
        assert self.agent.model is not None
        assert len(self.agent.tools) >= 6  # Should have core tools
        assert self.agent.model_name is not None
        
        # Test agent info
        info = self.agent.get_agent_info()
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "tools_count" in info
        assert "api_configuration" in info
        assert info["tools_count"] >= 6
        assert info["model_type"] == "api_based"
    
    def test_process_documents_empty(self):
        """Test processing with no documents."""
        result = self.agent.process_documents([])
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error_message" in result
        assert "No documents provided" in result["error_message"]
    
    def test_process_documents_invalid_structure(self):
        """Test processing with invalid document structure."""
        invalid_docs = [
            {"name": "test.txt"},  # Missing content
            "not_a_dict"  # Not a dictionary
        ]
        
        result = self.agent.process_documents(invalid_docs)
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error_message" in result
    
    def test_tool_integration(self):
        """Test that all required tools are properly integrated."""
        tool_names = []
        for tool in self.agent.tools:
            if hasattr(tool, 'name'):
                tool_names.append(tool.name)
            elif hasattr(tool, '__name__'):
                tool_names.append(tool.__name__)
            else:
                tool_names.append(str(tool))
        
        expected_tools = [
            "extract_text_from_document",
            "chunk_text_intelligently", 
            "generate_embeddings",
            "extract_entities_and_topics",
            "discover_semantic_connections",
            "generate_innovation_synthesis"
        ]
        
        print(f"Available tools: {tool_names}")  # Debug output
        
        for expected_tool in expected_tools:
            assert any(expected_tool in tool_name for tool_name in tool_names), \
                f"Tool '{expected_tool}' not found in agent tools: {tool_names}"
    
    def test_global_agent_instance(self):
        """Test global agent instance management."""
        reset_innovation_agent()
        
        agent1 = get_innovation_agent()
        agent2 = get_innovation_agent()
        
        # Should return the same instance
        assert agent1 is agent2
        assert isinstance(agent1, InnovationCatalystAgent)
    
    def test_process_documents_tool_function(self):
        """Test the @tool decorated function."""
        files_data = json.dumps(self.sample_docs)
        
        result_str = process_documents_for_innovation(files_data, "innovation")
        
        assert isinstance(result_str, str)
        
        try:
            result = json.loads(result_str)
            assert isinstance(result, dict)
            assert "success" in result
            assert "processing_time" in result
            
            # If processing succeeded, check for expected fields
            if result.get("success"):
                assert "documents_processed" in result
                assert result["documents_processed"] == len(self.sample_docs)
                
        except json.JSONDecodeError:
            pytest.fail(f"Tool function did not return valid JSON. Got: {result_str}")
    
    def test_process_documents_tool_function_invalid_json(self):
        """Test tool function with invalid JSON input."""
        invalid_json = "not valid json"
        
        result_str = process_documents_for_innovation(invalid_json, "innovation")
        result = json.loads(result_str)
        
        assert result["success"] is False
        assert "error_type" in result
        assert result["error_type"] == "json_decode_error"
    
    def test_process_documents_tool_function_invalid_structure(self):
        """Test tool function with invalid document structure."""
        invalid_docs = json.dumps([{"name": "test.txt"}])  # Missing content
        
        result_str = process_documents_for_innovation(invalid_docs, "innovation")
        result = json.loads(result_str)
        
        assert result["success"] is False
        assert "error_type" in result
        assert result["error_type"] == "validation_error"
    
    def test_different_focus_themes(self):
        """Test processing with different focus themes."""
        themes = ["technology", "healthcare", "sustainability", "business"]
        
        for theme in themes:
            result = self.agent.process_documents(
                self.sample_docs[:1],  # Use one document for speed
                focus_theme=theme
            )
            
            assert isinstance(result, dict)
            assert "processing_time" in result
            assert "focus_theme" in result
            assert result["focus_theme"] == theme
    
    def test_agent_configuration_info(self):
        """Test agent configuration information."""
        info = self.agent.get_agent_info()
        
        # Check required fields
        required_fields = [
            "agent_version", "model_name", "model_type", "tools_count",
            "tools", "configuration", "api_configuration"
        ]
        
        for field in required_fields:
            assert field in info, f"Missing required field: {field}"
        
        # Check API configuration
        api_config = info["api_configuration"]
        assert "base_url" in api_config
        assert "timeout" in api_config
        assert "token_configured" in api_config
        
        # Check tools information
        assert len(info["tools"]) == info["tools_count"]
        for tool_info in info["tools"]:
            assert "name" in tool_info
            assert "type" in tool_info
            assert "callable" in tool_info

class TestAgentPerformance:
    """Test agent performance and error handling."""
    
    def setup_method(self):
        """Setup performance test environment."""
        reset_innovation_agent()
        
        if not os.getenv("HF_TOKEN_TO_CALL_LLM"):
            os.environ["HF_TOKEN_TO_CALL_LLM"] = "test_token_for_testing"
        
        try:
            self.agent = InnovationCatalystAgent()
        except Exception as e:
            pytest.skip(f"Agent initialization failed: {e}")
    
    def test_processing_timeout_handling(self):
        """Test handling of processing timeouts."""
        # Create a large document that might timeout
        large_doc = [{
            "name": "large_document.txt",
            "content": "This is a test document. " * 1000,  # Large content
        }]
        
        start_time = time.time()
        result = self.agent.process_documents(large_doc)
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time or handle timeout gracefully
        assert processing_time < 120  # 2 minutes max
        assert isinstance(result, dict)
        assert "processing_time" in result
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        # Test with problematic content
        problematic_docs = [
            {
                "name": "empty.txt",
                "content": "",  # Empty content
            },
            {
                "name": "special_chars.txt",
                "content": "Special characters: 🚀 💡 🔬 ⚡ 🌟",
            }
        ]
        
        result = self.agent.process_documents(problematic_docs)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert "processing_time" in result
        # Don't require success, but should not crash
    
    @pytest.mark.skipif(
        not os.getenv("HF_TOKEN_TO_CALL_LLM") or 
        os.getenv("HF_TOKEN_TO_CALL_LLM") == "test_token_for_testing",
        reason="Real API token required for integration test"
    )
    def test_real_api_integration(self):
        """Test real API integration (requires valid token)."""
        small_docs = [{
            "name": "integration_test.txt",
            "content": "Artificial intelligence and blockchain technology are converging to create new opportunities in decentralized AI systems.",
        }]
        
        result = self.agent.process_documents(small_docs, focus_theme="technology")
        
        assert isinstance(result, dict)
        assert "processing_time" in result
        
        # If successful, check for meaningful output
        if result.get("success"):
            assert "raw_output" in result
            assert len(result["raw_output"]) > 100  # Should have substantial output

def run_agent_tests():
    """Run all agent tests with proper error handling."""
    print("🤖 Running Innovation Agent Tests...")
    
    # Check for API token
    if not os.getenv("HF_TOKEN_TO_CALL_LLM"):
        print("⚠️  Warning: HF_TOKEN_TO_CALL_LLM not set. Using mock token for basic tests.")
        os.environ["HF_TOKEN_TO_CALL_LLM"] = "test_token_for_testing"
    
    test_classes = [TestInnovationAgent, TestAgentPerformance]
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}...")
        
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                test_instance = test_class()
                test_instance.setup_method()
                
                method = getattr(test_instance, method_name)
                method()
                print(f"  ✅ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ❌ {method_name}: {str(e)}")
    
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests < total_tests:
        print("\n💡 Tips for failing tests:")
        print("   - Ensure HF_TOKEN_TO_CALL_LLM is set with a valid token")
        print("   - Check internet connectivity for API calls")
        print("   - Verify all required dependencies are installed")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    run_agent_tests()
