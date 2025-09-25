"""
Configuration settings for HANRAG system.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class HANRAGConfig(BaseSettings):
    """Configuration class for HANRAG system."""

    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    langsmith_api_key: Optional[str] = Field(None, env="LANGSMITH_API_KEY")

    # Model Configuration
    default_model: str = Field("gpt-4o", env="DEFAULT_MODEL")
    embedding_model: str = Field("text-embedding-ada-002", env="EMBEDDING_MODEL")

    # Retrieval Configuration
    top_k_documents: int = Field(5, env="TOP_K_DOCUMENTS")
    similarity_threshold: float = Field(0.7, env="SIMILARITY_THRESHOLD")
    max_hops: int = Field(3, env="MAX_HOPS")

    # Noise Resistance Configuration
    noise_threshold: float = Field(0.4, env="NOISE_THRESHOLD")
    query_noise_threshold: float = Field(0.12, env="QUERY_NOISE_THRESHOLD")
    confidence_threshold: float = Field(0.8, env="CONFIDENCE_THRESHOLD")

    # LangSmith Configuration
    langsmith_tracing: bool = Field(False, env="LANGSMITH_TRACING")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global configuration instance
config = HANRAGConfig()
