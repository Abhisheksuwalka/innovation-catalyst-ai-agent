# /src/innovation_catalyst/agents/innovation_agent.py
"""
Main Innovation Catalyst Agent implementation using SmolAgent framework.
Orchestrates all tools to provide comprehensive innovation discovery capabilities.

This module provides:
- InnovationCatalystAgent: Main agent class with API-based model access
- Tool integration and management
- Document processing workflow
- Error handling and logging
- Performance monitoring
"""

import time
import json
import logging
import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

import nltk
nltk.download('punkt_tab')
nltk.download('punkt')

# Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# SmolAgent framework imports
from smolagents import CodeAgent, tool, PythonInterpreterTool, InferenceClientModel, LiteLLMModel

# Internal imports
from ..utils.config import get_config, validate_setup
from ..utils.logging import get_logger, log_function_performance

# Initialize logging and configuration
logger = get_logger(__name__)
config = get_config()

class InnovationCatalystAgent:
    """
    Main Innovation Catalyst Agent that orchestrates the complete workflow.
    
    This agent uses the HuggingFace Inference API to access language models
    without downloading them locally. It integrates multiple specialized tools
    for document processing, semantic analysis, and innovation discovery.
    
    Features:
        - API-based model access (no local downloads)
        - Comprehensive document processing pipeline
        - Semantic connection discovery
        - Innovation synthesis and recommendations
        - Error handling and recovery
        - Performance monitoring
        - Configurable tool integration
    
    Attributes:
        model_name (str): Name of the language model being used
        model (InferenceClientModel): API client for model inference
        tools (List): List of available tools for the agent
        agent (CodeAgent): The underlying SmolAgent instance
    """
    
    def __init__(self, model_name: Optional[str] = None, custom_tools: Optional[List] = None):
        """
        Initialize the Innovation Catalyst Agent with API-based model access.
        
        Args:
            model_name (Optional[str]): HuggingFace model ID to use via API.
                                      Defaults to config.model.llm_model
            custom_tools (Optional[List]): Additional tools to include in the agent
        
        Raises:
            ValueError: If HF_TOKEN_TO_CALL_LLM is not set
            Exception: If model initialization fails
        """
        # Validate setup first
        if not validate_setup():
            raise ValueError(
                "Setup validation failed. Please ensure HF_TOKEN_TO_CALL_LLM is set correctly."
            )
        
        # Use instruction-following model from config
        self.model_name = model_name or config.model.llm_model
        logger.info(f"Initializing agent with API model: {self.model_name}")
        
        # Initialize model client for API access
        self._initialize_model_client()
        
        # Prepare and validate tools
        self.tools = self._initialize_tools(custom_tools)
        
        # Initialize the CodeAgent with validated tools
        self._initialize_agent()
        
        logger.info(
            f"Innovation Catalyst Agent successfully initialized with "
            f"{len(self.tools)} tools using model: {self.model_name}"
        )

    def _initialize_model_client(self) -> None:
        """
        Initialize the LLM client with GitHub Models support.
        Falls back automatically and surfaces configuration problems early.
        """
        provider = config.model.llm_provider.lower()
        model_id = self.model_name

        if provider == "github":
            api_key = config.model.github_api_key
            if not api_key:
                raise ValueError("GITHUB_API_KEY is missing or empty.")
            
            # Use LiteLLM for GitHub Models
            self.model = LiteLLMModel(
                model_id=model_id,  # Already includes "github/" prefix
                api_key=api_key,
                api_base=config.model.github_api_base,
                temperature=config.agent.temperature,
                max_tokens=1024,
                timeout=config.model.request_timeout,
            )
            logger.info(f"LiteLLMModel ready (provider = GitHub Models): {model_id}")
            
        elif provider in {"huggingface", "hf"}:
            api_key = config.model.hf_token
            if not api_key:
                raise ValueError("HF_TOKEN_TO_CALL_LLM is missing or empty.")
            
            self.model = InferenceClientModel(
                model_id=model_id,
                token=api_key,
                timeout=config.model.request_timeout,
                max_tokens=1024,
                temperature=config.agent.temperature,
                top_p=0.9,
            )
            logger.info("InferenceClientModel ready (provider = HuggingFace)")
            
        elif provider in {"together_ai", "together"}:
            api_key = config.model.together_ai_key
            if not api_key:
                raise ValueError("TOGETHER_AI_KEY is missing or empty.")
            
            self.model = LiteLLMModel(
                model_id=f"together_ai/{model_id}",
                api_key=api_key,
                request_timeout=config.model.request_timeout,
            )
            logger.info("LiteLLMModel ready (provider = Together AI)")
            
        else:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                "Valid options: 'github', 'huggingface', 'together_ai'."
            )
    
    def _initialize_tools(self, custom_tools: Optional[List] = None) -> List:
        """
        Initialize and validate all tools for the agent.
        
        Args:
            custom_tools (Optional[List]): Additional custom tools to include
        
        Returns:
            List: List of validated tool instances
        
        Raises:
            ImportError: If required tools cannot be imported
            Exception: If tool validation fails
        """
        try:
            # Import custom tools (these should be @tool decorated functions)
            from ..tools.text_extraction import extract_text_from_document
            from ..tools.text_chunking import chunk_text_intelligently
            from ..tools.embeddings import generate_embeddings
            from ..tools.nlp_analysis import extract_entities_and_topics
            from ..tools.connection_discovery import discover_semantic_connections
            from ..tools.innovation_synthesis import generate_innovation_synthesis
            
            # Core tools list - these are SimpleTool instances after @tool decoration
            core_tools = [
                extract_text_from_document,
                chunk_text_intelligently,
                generate_embeddings,
                extract_entities_and_topics,
                discover_semantic_connections,
                generate_innovation_synthesis,
                PythonInterpreterTool()  # Built-in tool for code execution
            ]
            
            # Add custom tools if provided
            if custom_tools:
                core_tools.extend(custom_tools)
                logger.info(f"Added {len(custom_tools)} custom tools")
            
            # Validate tools
            self._validate_tools(core_tools)
            
            logger.info(f"Successfully initialized {len(core_tools)} tools")
            return core_tools
            
        except ImportError as e:
            logger.error(f"Failed to import required tools: {e}")
            raise
        except Exception as e:
            logger.error(f"Tool initialization failed: {e}")
            raise
    
    def _validate_tools(self, tools: List) -> None:
        """
        Validate that all tools are properly configured.
        
        Args:
            tools (List): List of tools to validate
        
        Raises:
            ValueError: If tool validation fails
        """
        for i, tool in enumerate(tools):
            if not hasattr(tool, 'name') and not hasattr(tool, '__name__'):
                raise ValueError(f"Tool at index {i} is missing name attribute: {tool}")
            
            # Check if it's a proper tool instance
            if not (hasattr(tool, '__call__') or hasattr(tool, 'run')):
                raise ValueError(f"Tool {getattr(tool, 'name', i)} is not callable")
        
        logger.debug("All tools validated successfully")
    
    def _initialize_agent(self) -> None:
        """Initialize the CodeAgent with validated tools."""
        try:
            self.agent = CodeAgent(
                model=self.model,
                tools=self.tools
            )
            # # Initialize the agent with context management
            # self.agent = CodeAgent(
            #     model=self.model,
            #     tools=self.tools,
            #     system_prompt=self.system_prompt,
            #     max_iterations=config.agent.max_iterations,
            #     max_prompt_tokens=6000,  # Leave buffer for response
            #     verbose=True
            # )
            logger.info("CodeAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CodeAgent: {e}")
            # Provide detailed error information
            tool_info = []
            for tool in self.tools:
                tool_name = getattr(tool, 'name', getattr(tool, '__name__', str(tool)))
                tool_type = type(tool).__name__
                tool_info.append(f"{tool_name} ({tool_type})")
            
            logger.error(f"Tools being used: {', '.join(tool_info)}")
            raise
    
    @log_function_performance("agent_process_documents")
    def process_documents(
        self, 
        files: List[Dict[str, Any]], 
        focus_theme: str = "innovation",
        max_connections: int = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process documents to generate innovation insights using the agent workflow.
        
        This method orchestrates the complete innovation discovery pipeline:
        1. Document text extraction and validation
        2. Intelligent text chunking
        3. Embedding generation and semantic analysis
        4. Entity and topic extraction
        5. Cross-document connection discovery
        6. Innovation synthesis and recommendations
        
        Args:
            files (List[Dict[str, Any]]): List of document objects, each containing:
                - name (str): Document name/identifier
                - content (str): Document text content
                - Additional metadata (optional)
            focus_theme (str): Primary theme for innovation analysis
            max_connections (int): Maximum number of connections to discover
            **kwargs: Additional parameters for processing
        
        Returns:
            Dict[str, Any]: Structured results containing:
                - success (bool): Whether processing succeeded
                - processing_time (float): Time taken in seconds
                - raw_output (str): Raw agent output
                - structured_results (Dict): Parsed and structured insights
                - error_message (str): Error details if processing failed
        
        Raises:
            ValueError: If input validation fails
        """
        start_time = time.time()
        
        # Input validation
        if not files:
            return {
                "success": False,
                "error_message": "No documents provided for processing",
                "processing_time": 0.0,
                "documents_processed": 0
            }
        
        # Validate document structure
        for i, file_obj in enumerate(files):
            if not isinstance(file_obj, dict):
                return {
                    "success": False,
                    "error_message": f"Document {i} is not a dictionary",
                    "processing_time": time.time() - start_time
                }
            
            if 'content' not in file_obj:
                return {
                    "success": False,
                    "error_message": f"Document {i} missing 'content' field",
                    "processing_time": time.time() - start_time
                }
        
        try:
            logger.info(
                f"Processing {len(files)} documents with focus theme: '{focus_theme}', "
                f"max connections: {max_connections}"
            )
            
            # Create comprehensive processing prompt
            processing_prompt = self._create_processing_prompt(
                files, focus_theme, max_connections, **kwargs
            )
            
            # Execute agent workflow
            logger.debug("Executing agent workflow...")
            result = self.agent.run(processing_prompt)
            
            processing_time = time.time() - start_time
            
            # Structure and validate results
            structured_result = self._structure_agent_result(result, processing_time)
            structured_result.update({
                "documents_processed": len(files),
                "focus_theme": focus_theme,
                "max_connections": max_connections
            })
            
            logger.info(f"Document processing completed successfully in {processing_time:.2f}s")
            return structured_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Document processing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "success": False,
                "error_message": error_msg,
                "processing_time": processing_time,
                "documents_processed": len(files),
                "focus_theme": focus_theme
            }
    
    def _create_concise_processing_prompt(
        self, 
        files: List[Dict[str, Any]], 
        focus_theme: str, 
        max_connections: int
    ) -> str:
        """Create a concise prompt that stays within token limits."""
        
        # Summarize files instead of including full content
        file_summaries = []
        for i, file in enumerate(files):
            content_preview = file.get('content', '')[:200] + "..." if len(file.get('content', '')) > 200 else file.get('content', '')
            file_summaries.append(f"File {i+1}: {file.get('name', f'doc_{i+1}')} ({len(file.get('content', ''))} chars)")
        
        files_summary = "\n".join(file_summaries)
        
        # Much more concise prompt
        prompt = f"""# Innovation Discovery Mission

    Process {len(files)} documents for innovation opportunities:

    {files_summary}

    **Workflow:**
    1. Extract text from documents
    2. Create chunks (500 words, 50 overlap)  
    3. Generate embeddings
    4. Extract entities/topics
    5. Find {max_connections} semantic connections
    6. Generate innovation synthesis

    **Focus:** {focus_theme}

    **Output:** Structured analysis with innovation opportunities, actionable steps, and feasibility assessment.

    Execute systematically using available tools."""
        
        return prompt


    def _create_processing_prompt(
        self, 
        files: List[Dict[str, Any]], 
        focus_theme: str,
        max_connections: int,
        **kwargs
    ) -> str:
        """
        Create a comprehensive processing prompt for the agent.
        
        This method generates a detailed prompt that guides the agent through
        the complete innovation discovery workflow, including specific instructions
        for each phase and expected outputs.
        
        Args:
            files (List[Dict[str, Any]]): Documents to process
            focus_theme (str): Innovation focus theme
            max_connections (int): Maximum connections to discover
            **kwargs: Additional parameters
        
        Returns:
            str: Comprehensive processing prompt
        """
#         # Prepare document summaries
#         file_summaries = []
#         total_content_length = 0
        
#         for i, file_obj in enumerate(files):
#             name = file_obj.get('name', f'document_{i+1}')
#             content = file_obj.get('content', '')
#             content_length = len(content)
#             total_content_length += content_length
            
#             # Create content preview
#             preview = content[:200] + "..." if len(content) > 200 else content
#             file_summaries.append(
#                 f"Document {i+1}: {name} ({content_length} chars)\n"
#                 f"Preview: {preview}"
#             )
        
#         files_overview = "\n\n".join(file_summaries)
        
#         # Create comprehensive prompt
#         prompt = f"""
# # Innovation Catalyst Agent - Document Processing Mission

# ## Mission Overview
# You are an expert AI system specialized in discovering breakthrough innovation opportunities by analyzing documents and finding novel connections between ideas.

# ## Documents to Process ({len(files)} total, {total_content_length} characters)
# {files_overview}

# ## Processing Workflow

# ### Phase 1: Document Analysis
# For each document, execute these steps:
# 1. **Extract and validate text content** using `extract_text_from_document()`
# 2. **Create intelligent chunks** using `chunk_text_intelligently()` 
#    - Target: {config.processing.chunk_size} words per chunk
#    - Overlap: {config.processing.chunk_overlap} words
# 3. **Generate semantic embeddings** using `generate_embeddings()`
# 4. **Extract entities and topics** using `extract_entities_and_topics()`

# ### Phase 2: Connection Discovery
# 1. **Find semantic connections** using `discover_semantic_connections()`
#    - Maximum connections: {max_connections}
#    - Focus theme: "{focus_theme}"
#    - Prioritize cross-domain and novel connections
#    - Filter for innovation potential

# ### Phase 3: Innovation Synthesis
# 1. **Generate comprehensive insights** using `generate_innovation_synthesis()`
#    - Focus on actionable opportunities
#    - Include feasibility assessment
#    - Provide implementation roadmap
#    - Calculate innovation scores

# ## Key Innovation Principles
# - **Cross-pollination**: Look for connections between different domains
# - **Novelty**: Prioritize unexpected combinations of familiar concepts
# - **Feasibility**: Focus on implementable ideas with clear value
# - **Impact**: Identify opportunities with significant potential
# - **Actionability**: Provide concrete next steps

# ## Expected Output Structure
# Provide a comprehensive analysis including:
# 1. **Processing Summary**: Documents analyzed, chunks created, connections found
# 2. **Top Innovation Opportunities**: Ranked list with explanations
# 3. **Cross-Domain Connections**: Novel relationships discovered
# 4. **Implementation Roadmap**: Actionable steps and timelines
# 5. **Risk Assessment**: Potential challenges and mitigation strategies
# 6. **Innovation Metrics**: Scores and confidence levels

# ## Document Contents
# {self._format_document_contents(files)}

# Begin the innovation discovery process now. Use the available tools systematically and provide comprehensive insights.
# """
        
#         return prompt
     return self._create_concise_processing_prompt(files, focus_theme, max_connections)
    
    def _format_document_contents(self, files: List[Dict[str, Any]]) -> str:
        """Format document contents for the prompt."""
        formatted_contents = []
        
        for i, file_obj in enumerate(files):
            name = file_obj.get('name', f'document_{i+1}')
            content = file_obj.get('content', '')
            
            formatted_contents.append(
                f"=== {name} ===\n{content}\n"
            )
        
        return "\n".join(formatted_contents)
    
    def _structure_agent_result(self, raw_result: str, processing_time: float) -> Dict[str, Any]:
        """
        Structure the agent's raw result into a standardized format.
        
        Args:
            raw_result (str): Raw output from the agent
            processing_time (float): Time taken for processing
        
        Returns:
            Dict[str, Any]: Structured result dictionary
        """
        try:
            # Try to parse as JSON first
            if raw_result.strip().startswith('{'):
                try:
                    parsed_result = json.loads(raw_result)
                    parsed_result.update({
                        "processing_time": processing_time,
                        "success": True,
                        "result_type": "structured_json"
                    })
                    return parsed_result
                except json.JSONDecodeError:
                    pass
            
            # Extract structured information from text
            structured_result = {
                "success": True,
                "processing_time": processing_time,
                "result_type": "parsed_text",
                "raw_output": raw_result,
                "summary": self._extract_summary(raw_result),
                "innovation_opportunities": self._extract_opportunities(raw_result),
                "connections": self._extract_connections(raw_result),
                "actionable_steps": self._extract_action_items(raw_result),
                "innovation_score": self._calculate_innovation_score(raw_result),
                "confidence_level": self._assess_confidence(raw_result)
            }
            
            return structured_result
            
        except Exception as e:
            logger.warning(f"Failed to structure agent result: {e}")
            return {
                "success": True,
                "processing_time": processing_time,
                "result_type": "raw_output",
                "raw_output": raw_result,
                "note": "Raw agent output - manual parsing recommended"
            }
    
    def _extract_summary(self, text: str) -> str:
        """Extract summary from agent output."""
        patterns = [
            r"(?i)(?:summary|overview|key findings)[:\s]+(.*?)(?=\n\n|\n[A-Z]|\n\d+\.|\n-|\n\*|$)",
            r"(?i)processing summary[:\s]+(.*?)(?=\n\n|\n[A-Z]|\n\d+\.|\n-|\n\*|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback: first meaningful paragraph
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        return paragraphs[0] if paragraphs else "Summary extraction failed"
    
    def _extract_opportunities(self, text: str) -> List[Dict[str, Any]]:
        """Extract innovation opportunities from text."""
        opportunities = []
        
        # Look for numbered lists or bullet points about opportunities
        patterns = [
            r"(?i)(?:opportunity|innovation)[:\s]*\d+[:\.]?\s*([^\n]+)",
            r"(?i)\d+\.\s*([^:\n]+):\s*([^\n]+)",
            r"[•\-\*]\s*([^:\n]+):\s*([^\n]+)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    opportunities.append({
                        "title": match[0].strip(),
                        "description": match[1].strip()
                    })
                else:
                    opportunities.append({
                        "title": "Innovation Opportunity",
                        "description": match.strip() if isinstance(match, str) else str(match)
                    })
        
        return opportunities[:10]  # Limit to top 10
    
    def _extract_connections(self, text: str) -> List[Dict[str, Any]]:
        """Extract semantic connections from text."""
        connections = []
        
        # Look for connection patterns
        connection_patterns = [
            r"(?i)connection[:\s]+([^:\n]+)\s*(?:and|with|to)\s*([^\n]+)",
            r"(?i)([^:\n]+)\s*(?:connects to|links with|relates to)\s*([^\n]+)",
            r"(?i)relationship between\s*([^:\n]+)\s*and\s*([^\n]+)"
        ]
        
        for pattern in connection_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 2:
                    connections.append({
                        "concept_a": match[0].strip(),
                        "concept_b": match[1].strip(),
                        "type": "semantic_connection"
                    })
        
        return connections[:15]  # Limit to top 15
    
    def _extract_action_items(self, text: str) -> List[str]:
        """Extract actionable steps from text."""
        actions = []
        
        action_patterns = [
            r"(?i)(?:step|action|recommendation|next step)[:\s]*\d*[:\.]?\s*([^\n]+)",
            r"(?i)implement[:\s]*([^\n]+)",
            r"(?i)develop[:\s]*([^\n]+)",
            r"(?i)create[:\s]*([^\n]+)"
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text)
            actions.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        return list(set(actions))[:8]  # Remove duplicates, limit to 8
    
    def _calculate_innovation_score(self, text: str) -> float:
        """Calculate innovation score based on text content."""
        # Look for explicit scores first
        score_patterns = [
            r"(?i)(?:innovation|novelty|potential)\s*score[:\s]*(\d+\.?\d*)",
            r"(?i)score[:\s]*(\d+\.?\d*)",
            r"(?i)rating[:\s]*(\d+\.?\d*)"
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    score = float(match.group(1))
                    return min(1.0, score / 100.0 if score > 1.0 else score)
                except ValueError:
                    continue
        
        # Calculate based on innovation keywords
        innovation_keywords = [
            "breakthrough", "novel", "innovative", "revolutionary", "disruptive",
            "unprecedented", "cutting-edge", "transformative", "paradigm",
            "cross-domain", "interdisciplinary", "synergy", "convergence"
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in innovation_keywords if keyword in text_lower)
        
        # Base score + keyword bonus
        base_score = 0.3
        keyword_bonus = min(0.6, keyword_count * 0.05)
        
        return base_score + keyword_bonus
    
    def _assess_confidence(self, text: str) -> str:
        """Assess confidence level of the analysis."""
        text_length = len(text)
        
        if text_length < 500:
            return "low"
        elif text_length < 2000:
            return "medium"
        else:
            return "high"
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the agent configuration.
        
        Returns:
            Dict[str, Any]: Agent configuration and status information
        """
        tool_info = []
        for tool in self.tools:
            tool_name = getattr(tool, 'name', getattr(tool, '__name__', 'unknown'))
            tool_type = type(tool).__name__
            tool_info.append({
                "name": tool_name,
                "type": tool_type,
                "callable": hasattr(tool, '__call__') or hasattr(tool, 'run')
            })
        
        return {
            "agent_version": "1.0.0",
            "model_name": self.model_name,
            "model_type": "api_based",
            "tools_count": len(self.tools),
            "tools": tool_info,
            "configuration": {
                "chunk_size": config.processing.chunk_size,
                "chunk_overlap": config.processing.chunk_overlap,
                "max_connections": config.connection.max_connections,
                "similarity_threshold_min": config.connection.similarity_threshold_min,
                "similarity_threshold_max": config.connection.similarity_threshold_max,
                "max_iterations": config.agent.max_iterations,
                "timeout_seconds": config.agent.timeout_seconds
            },
            "api_configuration": {
                "base_url": config.model.api_base_url,
                "timeout": config.model.request_timeout,
                "token_configured": bool(config.model.hf_token)
            }
        }

# Global agent instance management
innovation_agent = None

def get_innovation_agent() -> InnovationCatalystAgent:
    """
    Get or create the global innovation agent instance.
    
    Returns:
        InnovationCatalystAgent: The global agent instance
    """
    global innovation_agent
    if innovation_agent is None:
        innovation_agent = InnovationCatalystAgent()
    return innovation_agent

def reset_innovation_agent() -> None:
    """Reset the global agent instance (useful for testing)."""
    global innovation_agent
    innovation_agent = None

@tool
def process_documents_for_innovation(files_data: str, focus_theme: str = "innovation") -> str:
    """
    Process multiple documents to discover innovation opportunities and generate insights.
    
    This is the main entry point for the Innovation Catalyst Agent workflow.
    It orchestrates document analysis, semantic connection discovery, and innovation synthesis.
    
    Args:
        files_data (str): JSON string containing a list of document objects. Each document 
                         must have 'name' and 'content' fields. Example:
                         '[{"name": "doc1.txt", "content": "document text here"}, ...]'
        focus_theme (str): The primary theme to focus the innovation analysis on. 
                          Examples: "technology", "healthcare", "sustainability", "business"
    
    Returns:
        str: A JSON string containing the structured results of the innovation analysis,
             including discovered opportunities, connections, and actionable recommendations.
    
    Example:
        >>> files = '[{"name": "ai_research.txt", "content": "AI is transforming..."}]'
        >>> result = process_documents_for_innovation(files, "artificial intelligence")
        >>> parsed_result = json.loads(result)
        >>> print(parsed_result["innovation_opportunities"])
    """
    try:
        # Parse input data
        files = json.loads(files_data)
        
        # Validate input structure
        if not isinstance(files, list):
            raise ValueError("files_data must be a JSON array of document objects")
        
        for i, file_obj in enumerate(files):
            if not isinstance(file_obj, dict):
                raise ValueError(f"Document {i} must be a dictionary")
            if 'content' not in file_obj:
                raise ValueError(f"Document {i} missing required 'content' field")
        
        # Get agent instance and process documents
        agent = get_innovation_agent()
        result = agent.process_documents(files, focus_theme=focus_theme)
        
        # Return structured JSON result
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in files_data: {e}")
        return json.dumps({
            "success": False,
            "error_message": f"Invalid JSON format: {str(e)}",
            "error_type": "json_decode_error"
        })
    except ValueError as e:
        logger.error(f"Input validation error: {e}")
        return json.dumps({
            "success": False,
            "error_message": str(e),
            "error_type": "validation_error"
        })
    except Exception as e:
        logger.error(f"Tool 'process_documents_for_innovation' failed: {e}", exc_info=True)
        return json.dumps({
            "success": False,
            "error_message": f"Processing failed: {str(e)}",
            "error_type": "processing_error"
        })
