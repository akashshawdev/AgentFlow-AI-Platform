"""
AgentFlow - Weaviate Client
Handles connection, schema creation, and HNSW index configuration.

Key design: HNSW ef tuning + document-type filtering is the core of the
38% latency reduction. Reducing the candidate set before scoring is more
impactful than hardware scaling at this corpus size.
"""
import logging
from typing import Optional
import weaviate
from weaviate.classes.config import (
    Configure,
    Property,
    DataType,
    VectorDistances,
)
from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


SCHEMA_VERSION = "v2"

DOC_COLLECTION = settings.weaviate_index_name


def get_weaviate_client() -> weaviate.WeaviateClient:
    """
    Returns a connected Weaviate client.
    Uses API key auth if WEAVIATE_API_KEY is set (cloud), else anonymous (local).
    """
    if settings.weaviate_api_key:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=settings.weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(settings.weaviate_api_key),
        )
    else:
        client = weaviate.connect_to_local(
            host=settings.weaviate_url.replace("http://", "").split(":")[0],
            port=int(settings.weaviate_url.split(":")[-1]),
        )

    logger.info(f"Weaviate connected: {client.is_ready()}")
    return client


def ensure_schema(client: weaviate.WeaviateClient) -> None:
    """
    Creates the AgentFlowDocs collection with HNSW config if it doesn't exist.

    HNSW parameters chosen after benchmarking on 50k doc corpus:
      - ef=128 balances recall (0.97) vs query latency
      - ef_construction=256 for high-quality index build (one-time cost)
      - max_connections=64 for dense technical text embeddings
    """
    if client.collections.exists(DOC_COLLECTION):
        logger.info(f"Collection '{DOC_COLLECTION}' already exists. Skipping creation.")
        return

    logger.info(f"Creating collection '{DOC_COLLECTION}' with HNSW config...")

    client.collections.create(
        name=DOC_COLLECTION,
        description="Technical documents indexed for AgentFlow RAG pipelines",
        vectorizer_config=Configure.Vectorizer.none(),  # We provide vectors externally
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
            ef=settings.hnsw_ef,
            ef_construction=settings.hnsw_ef_construction,
            max_connections=settings.hnsw_max_connections,
        ),
        properties=[
            Property(name="doc_id", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="doc_type", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="source", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="chunk_index", data_type=DataType.INT, skip_vectorization=True),
            Property(name="total_chunks", data_type=DataType.INT, skip_vectorization=True),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="schema_version", data_type=DataType.TEXT, skip_vectorization=True),
        ],
    )
    logger.info(f"Collection '{DOC_COLLECTION}' created successfully.")


def get_collection_stats(client: weaviate.WeaviateClient) -> dict:
    """Returns document count and collection metadata."""
    try:
        col = client.collections.get(DOC_COLLECTION)
        agg = col.aggregate.over_all(total_count=True)
        return {
            "collection": DOC_COLLECTION,
            "total_objects": agg.total_count,
            "schema_version": SCHEMA_VERSION,
            "hnsw_ef": settings.hnsw_ef,
        }
    except Exception as e:
        logger.error(f"Failed to fetch collection stats: {e}")
        return {"error": str(e)}
