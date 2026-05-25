"""
AgentFlow - Core Configuration
Centralized settings via Pydantic BaseSettings for all components.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Application
    app_name: str = "AgentFlow AI Platform"
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    embedding_model: str = Field(
        default="text-embedding-3-small", env="EMBEDDING_MODEL"
    )
    embedding_dim: int = 1536

    # Weaviate
    weaviate_url: str = Field(default="http://localhost:8080", env="WEAVIATE_URL")
    weaviate_api_key: str | None = Field(default=None, env="WEAVIATE_API_KEY")
    weaviate_index_name: str = "AgentFlowDocs"

    # RAG Pipeline
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 8
    hybrid_alpha: float = 0.75  # 0=BM25 only, 1=vector only

    # HNSW tuning (key to 38% latency reduction)
    hnsw_ef: int = 128          # query-time ef parameter
    hnsw_ef_construction: int = 256
    hnsw_max_connections: int = 64

    # Agent settings
    agent_max_iterations: int = 10
    agent_verbose: bool = True
    validator_retry_limit: int = 3

    # Evaluation
    selfcheck_num_samples: int = 5
    eval_batch_size: int = 20

    # Database (prompt registry)
    database_url: str = Field(
        default="sqlite+aiosqlite:///./agentflow.db", env="DATABASE_URL"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
