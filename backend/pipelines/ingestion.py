"""
AgentFlow - RAG Ingestion Pipeline
Processes raw documents into chunked, embedded nodes stored in Weaviate.

Designed for the 50,000-document corpus. Key features:
  - Async batch ingestion with configurable batch size
  - Document type tagging (api_reference, tutorial, runbook, spec)
  - Deduplication via doc_id hash
  - Progress tracking with tqdm
  - Retry logic on Weaviate write failures
"""
import hashlib
import logging
import time
from pathlib import Path
from typing import Iterator
import uuid

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from backend.core.config import get_settings
from backend.core.weaviate_client import get_weaviate_client, ensure_schema, DOC_COLLECTION

logger = logging.getLogger(__name__)
settings = get_settings()


DOC_TYPE_MAP = {
    ".md": "tutorial",
    ".rst": "tutorial",
    ".txt": "runbook",
    ".pdf": "specification",
    ".json": "api_reference",
    ".yaml": "configuration",
    ".yml": "configuration",
}


def detect_doc_type(file_path: Path) -> str:
    return DOC_TYPE_MAP.get(file_path.suffix.lower(), "general")


def doc_hash(text: str, source: str) -> str:
    return hashlib.sha256(f"{source}::{text[:200]}".encode()).hexdigest()[:16]


class IngestionPipeline:
    """
    End-to-end document ingestion: load → chunk → embed → upsert to Weaviate.
    """

    def __init__(self, batch_size: int = 500):
        self.batch_size = batch_size
        self.embed_model = OpenAIEmbedding(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            embed_batch_size=100,
        )
        self.splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.client = get_weaviate_client()
        ensure_schema(self.client)
        self.collection = self.client.collections.get(DOC_COLLECTION)
        self.stats = {
            "documents_loaded": 0,
            "chunks_created": 0,
            "chunks_upserted": 0,
            "duplicates_skipped": 0,
            "errors": 0,
        }

    def load_documents(self, source_dir: str) -> list[Document]:
        """Load all supported documents from a directory."""
        reader = SimpleDirectoryReader(
            input_dir=source_dir,
            recursive=True,
            required_exts=[".md", ".txt", ".pdf", ".rst", ".json"],
        )
        docs = reader.load_data()
        self.stats["documents_loaded"] = len(docs)
        logger.info(f"Loaded {len(docs)} documents from {source_dir}")
        return docs

    def chunk_documents(self, documents: list[Document]) -> list[dict]:
        """Split documents into chunks with metadata."""
        all_chunks = []
        for doc in documents:
            nodes = self.splitter.get_nodes_from_documents([doc])
            source = doc.metadata.get("file_path", "unknown")
            doc_type = detect_doc_type(Path(source))
            title = doc.metadata.get("file_name", Path(source).name)

            for i, node in enumerate(nodes):
                all_chunks.append({
                    "doc_id": doc_hash(node.text, source),
                    "text": node.text,
                    "doc_type": doc_type,
                    "source": source,
                    "title": title,
                    "chunk_index": i,
                    "total_chunks": len(nodes),
                    "schema_version": "v2",
                })

        self.stats["chunks_created"] = len(all_chunks)
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def _batched(self, items: list, n: int) -> Iterator[list]:
        for i in range(0, len(items), n):
            yield items[i: i + n]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _upsert_batch(self, batch: list[dict], embeddings: list[list[float]]) -> int:
        """Upsert a batch of chunks with their embeddings into Weaviate."""
        upserted = 0
        with self.collection.batch.dynamic() as b:
            for chunk, vector in zip(batch, embeddings):
                b.add_object(
                    properties={k: v for k, v in chunk.items() if k != "doc_id"},
                    uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["doc_id"])),
                    vector=vector,
                )
                upserted += 1
        return upserted

    def run(self, source_dir: str) -> dict:
        """Full ingestion run. Returns stats dict."""
        start = time.time()
        documents = self.load_documents(source_dir)
        chunks = self.chunk_documents(documents)

        logger.info(f"Starting batch ingestion ({self.batch_size}/batch)...")

        for batch in tqdm(
            list(self._batched(chunks, self.batch_size)),
            desc="Ingesting batches",
        ):
            texts = [c["text"] for c in batch]
            try:
                embeddings = self.embed_model.get_text_embedding_batch(texts)
                upserted = self._upsert_batch(batch, embeddings)
                self.stats["chunks_upserted"] += upserted
            except Exception as e:
                logger.error(f"Batch failed: {e}")
                self.stats["errors"] += len(batch)

        elapsed = time.time() - start
        self.stats["elapsed_seconds"] = round(elapsed, 2)
        self.stats["throughput_chunks_per_sec"] = round(
            self.stats["chunks_upserted"] / max(elapsed, 1), 1
        )
        logger.info(f"Ingestion complete: {self.stats}")
        return self.stats
