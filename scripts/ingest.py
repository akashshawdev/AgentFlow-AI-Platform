#!/usr/bin/env python3
"""
AgentFlow - Document Ingestion Script
Usage: python scripts/ingest.py --source data/sample_docs --batch-size 500
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.config import get_settings
from backend.core.llama_setup import configure_llama_index
from backend.pipelines.ingestion import IngestionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ingest")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into AgentFlow")
    parser.add_argument("--source", required=True, help="Directory containing documents")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for upsert")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        logger.error(f"Source directory not found: {source}")
        sys.exit(1)

    logger.info(f"Starting ingestion from: {source}")
    logger.info(f"Batch size: {args.batch_size}")

    configure_llama_index()
    pipeline = IngestionPipeline(batch_size=args.batch_size)
    stats = pipeline.run(str(source))

    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k:35s}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
