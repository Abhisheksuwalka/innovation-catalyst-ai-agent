# src/innovation_catalyst/utils/config.py
"""
Configuration management for Innovation Catalyst Agent.
Handles all configuration settings with proper validation and type safety.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, validator, Field
import logging
# from dotenv import load_dotenv
# load_dotenv()

class ModelConfig(BaseModel):
    """Configuration for AI models with API support."""
    
    hf_token: Optional[str] = Field(
        default_factory=lambda: os.getenv("HF_TOKEN_TO_CALL_LLM"),
        description="Hugging Face API token for model access"
    )
    together_ai_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("TOGETHER_AI_KEY"),
        description="Together AI API key for inference"
    )
    
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )

    github_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GITHUB_API_KEY"),
        description="GitHub API key for GitHub Models marketplace"
    )

    llm_provider: str = Field(
        default="github",  # Changed from "together_ai" to "github"
        description="LLM API provider (e.g., 'github', 'together_ai', 'huggingface')"
    )

    llm_model: str = Field(
        default="github/microsoft/phi-4-multimodal-instruct",  # Changed to GitHub model
        description="Language model for agent reasoning (instruction-following model)"
    )

    github_api_base: str = Field(
        default="https://models.github.ai/inference",
        description="GitHub Models API base URL"
    )
    
    spacy_model: str = Field(
        default="en_core_web_sm",
        description="spaCy model for NLP tasks"
    )
    
    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for model inference"
    )
    
    max_sequence_length: int = Field(
        default=512,
        ge=128,
        le=4096,
        description="Maximum sequence length for models"
    )
    
    device: str = Field(
        default="auto",
        description="Device for model inference (auto, cpu, cuda)"
    )
    
    api_base_url: str = Field(
        default="https://api-inference.huggingface.co",
        description="HuggingFace Inference API base URL"
    )
    
    request_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="API request timeout in seconds"
    )
    
    @validator('device')
    def validate_device(cls, v):
        """Validate device configuration."""
        valid_devices = ['auto', 'cpu', 'cuda', 'mps']
        if v not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}")
        return v
    
    @validator('llm_provider')
    def validate_llm_provider(cls, v):
        """Validate LLM provider."""
        valid_providers = ['github', 'huggingface', 'together_ai']
        if v not in valid_providers:
            raise ValueError(f"Provider must be one of {valid_providers}")
        return v

    @validator('llm_model')
    def validate_llm_model(cls, v, values):
        """Ensure we're using an appropriate model for the provider."""
        provider = values.get('llm_provider', 'github')
        
        if provider == 'github':
            # GitHub Models format: github/{publisher}/{model_name}
            github_models = [
                'github/openai/gpt-4o',
                'github/openai/gpt-4o-mini',
                'github/deepseek/deepseek-r1-0528',
                'github/meta-llama/llama-3.3-70b-instruct',
                'github/microsoft/phi-4-multimodal-instruct',
                'github/mistralai/mistral-large-2407'
            ]
            if not v.startswith('github/'):
                logging.warning(f"GitHub model should start with 'github/': {v}")
        
        elif provider == 'together_ai':
            # Together AI models
            chat_models = [
                'meta-llama/Llama-2-7b-chat-hf',
                'meta-llama/Llama-2-13b-chat-hf',
                'mistralai/Mixtral-8x7B-Instruct-v0.1',
                'mistralai/Mistral-7B-Instruct-v0.1'
            ]
        
        elif provider == 'huggingface':
            # HuggingFace models
            if v == 'microsoft/DialoGPT-medium':
                logging.warning(
                    "DialoGPT-medium is a conversational model, not instruction-following. "
                    "Consider using meta-llama/Llama-2-7b-chat-hf or similar."
                )
        
        return v

class ProcessingConfig(BaseModel):
    """Configuration for document processing."""
    
    chunk_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Target words per chunk"
    )
    
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=200,
        description="Overlapping words between chunks"
    )
    
    max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,
        description="Maximum file size in bytes"
    )
    
    supported_formats: list = Field(
        default=['.pdf', '.txt', '.docx', '.md', '.json'],
        description="Supported file formats"
    )
    
    min_chunk_words: int = Field(
        default=50,
        ge=10,
        description="Minimum words per chunk"
    )
    
    @validator('chunk_overlap')
    def validate_overlap(cls, v, values):
        """Ensure overlap is less than chunk size."""
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v

class ConnectionConfig(BaseModel):
    """Configuration for connection discovery."""
    
    similarity_threshold_min: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )
    
    similarity_threshold_max: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Maximum similarity threshold"
    )
    
    max_connections: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum connections to discover"
    )
    
    novelty_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for novelty in scoring"
    )
    
    similarity_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for similarity in scoring"
    )
    
    @validator('similarity_threshold_max')
    def validate_thresholds(cls, v, values):
        """Ensure max threshold is greater than min."""
        if 'similarity_threshold_min' in values and v <= values['similarity_threshold_min']:
            raise ValueError("Max threshold must be greater than min threshold")
        return v

class AgentConfig(BaseModel):
    """Configuration for the SmolAgent."""
    
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum agent iterations"
    )
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation"
    )
    
    timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=1800,
        description="Timeout for agent execution"
    )

class InnovationCatalystConfig(BaseModel):
    """Main configuration class for Innovation Catalyst Agent."""
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    connection: ConnectionConfig = Field(default_factory=ConnectionConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    
    # Paths
    cache_dir: Path = Field(
        default=Path("cache"),
        description="Directory for caching"
    )
    logs_dir: Path = Field(
        default=Path("logs"),
        description="Directory for logs"
    )
    
    # Logging configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)"
    )
    # CORRECTED: Added the missing log_format field
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Format string for log messages"
    )


    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"

class ConfigManager:
    """
    Manages configuration loading, validation, and access.
    
    Features:
        - Environment variable support
        - Configuration validation
        - Default value management
        - API token management
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else Path("config.yaml")
        self._config: Optional[InnovationCatalystConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from environment variables and defaults."""
        config_data = {}
        
        # Load from YAML file if exists
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
            except Exception as e:
                logging.warning(f"Failed to load config file {self.config_path}: {e}")
        
        # Apply environment variable overrides
        config_data = self._apply_env_overrides(config_data)
        
        # Create and validate configuration
        try:
            self._config = InnovationCatalystConfig(**config_data)
            logging.info("Configuration loaded successfully")
        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            # Fall back to default configuration
            self._config = InnovationCatalystConfig()
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            'HF_TOKEN_TO_CALL_LLM': ['model', 'hf_token'],
            'IC_EMBEDDING_MODEL': ['model', 'embedding_model'],
            'IC_LLM_MODEL': ['model', 'llm_model'],
            'IC_BATCH_SIZE': ['model', 'batch_size'],
            'IC_DEVICE': ['model', 'device'],
            'IC_CHUNK_SIZE': ['processing', 'chunk_size'],
            'IC_CHUNK_OVERLAP': ['processing', 'chunk_overlap'],
            'IC_MAX_CONNECTIONS': ['connection', 'max_connections'],
            'IC_LOG_LEVEL': ['log_level'],
            'IC_LLM_PROVIDER': ['model', 'llm_provider'],
            'TOGETHER_AI_KEY': ['model', 'together_ai_key'],
            'GITHUB_API_KEY': ['model', 'github_api_key'],
        }
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Type conversion
                if any(x in env_var.lower() for x in ['size', 'connections', 'batch']):
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                
                # Set nested configuration
                current = config_data
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = value
        
        return config_data
    
    @property
    def config(self) -> InnovationCatalystConfig:
        """Get the current configuration."""
        if self._config is None:
            self._load_config()
        return self._config
    
    def validate_api_setup(self) -> bool:
        """Validate API configuration for the selected provider."""
        provider = self.config.model.llm_provider.lower()
        
        if provider == "github":
            if not self.config.model.github_api_key:
                logging.error("GITHUB_API_KEY environment variable not set")
                return False
            if len(self.config.model.github_api_key) < 10:
                logging.error("GITHUB_API_KEY appears to be invalid (too short)")
                return False
            logging.info("GitHub Models API configuration validated successfully")
            return True
            
        elif provider in {"huggingface", "hf"}:
            if not self.config.model.hf_token:
                logging.error("HF_TOKEN_TO_CALL_LLM environment variable not set")
                return False
            if len(self.config.model.hf_token) < 10:
                logging.error("HF_TOKEN_TO_CALL_LLM appears to be invalid (too short)")
                return False
            logging.info("HuggingFace API configuration validated successfully")
            return True
            
        elif provider in {"together_ai", "together"}:
            if not self.config.model.together_ai_key:
                logging.error("TOGETHER_AI_KEY environment variable not set")
                return False
            logging.info("Together AI configuration validated successfully")
            return True
            
        else:
            logging.error(f"Unknown provider: {provider}")
            return False


# Global configuration instance
config_manager = ConfigManager()

def get_config() -> InnovationCatalystConfig:
    """Get the global configuration instance."""
    return config_manager.config

def validate_setup() -> bool:
    """Validate the complete setup."""
    return config_manager.validate_api_setup()
