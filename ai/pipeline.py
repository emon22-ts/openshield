"""
Main RAG ingestion pipeline for OpenShield.

Orchestrates document loading, chunking, embedding,
and vector store ingestion in a single pipeline call.
Supports full build and incremental refresh.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ai.chunker import chunk_documents
from ai.embedder import embed_chunks, get_embedder
from ai.loader import load_all_documents
from ai.vector_store.chroma_store import OpenShieldVectorStore

logger = logging.getLogger(__name__)


def build_pipeline(
    persist_dir: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> OpenShieldVectorStore:
    """
    Run the full RAG ingestion pipeline.

    Steps:
        1. Load all OpenShield rules and compliance documents
        2. Chunk documents into overlapping segments
        3. Embed chunks using sentence-transformers
        4. Ingest embeddings into ChromaDB vector store

    Args:
        persist_dir: Optional custom path for ChromaDB storage.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        Populated OpenShieldVectorStore instance.
    """
    start = time.time()
    logger.info("=== OpenShield RAG Pipeline — Full Build ===")

    # Step 1 — Load documents
    logger.info("Step 1: Loading documents...")
    documents = load_all_documents()
    logger.info("Loaded %d documents", len(documents))

    # Step 2 — Chunk documents
    logger.info("Step 2: Chunking documents...")
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    logger.info("Created %d chunks", len(chunks))

    # Step 3 — Embed chunks
    logger.info("Step 3: Embedding chunks...")
    model = get_embedder()
    chunks = embed_chunks(chunks, model=model)
    logger.info("Generated %d embeddings", len(chunks))

    # Step 4 — Ingest into vector store
    logger.info("Step 4: Ingesting into vector store...")
    store = OpenShieldVectorStore(persist_dir=persist_dir)
    store.ingest(chunks)

    elapsed = round(time.time() - start, 2)
    logger.info(
        "=== Pipeline complete in %ss — %d chunks stored ===",
        elapsed,
        store.count(),
    )
    return store


def refresh_pipeline(
    persist_dir: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> OpenShieldVectorStore:
    """
    Refresh the vector store without full rebuild.

    Loads and re-embeds all documents then upserts into
    the existing vector store. Only changed chunks are updated.

    Args:
        persist_dir: Optional custom path for ChromaDB storage.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        Updated OpenShieldVectorStore instance.
    """
    start = time.time()
    logger.info("=== OpenShield RAG Pipeline — Refresh ===")

    documents = load_all_documents()
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    model = get_embedder()
    chunks = embed_chunks(chunks, model=model)

    store = OpenShieldVectorStore(persist_dir=persist_dir)
    store.refresh(chunks)

    elapsed = round(time.time() - start, 2)
    logger.info(
        "=== Refresh complete in %ss — %d chunks in store ===",
        elapsed,
        store.count(),
    )
    return store


def query_pipeline(
    query: str,
    persist_dir: Optional[str] = None,
    n_results: int = 5,
    filter_metadata: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Query the vector store for relevant documents.

    Args:
        query: Natural language query string.
        persist_dir: Optional custom path for ChromaDB storage.
        n_results: Number of results to return.
        filter_metadata: Optional metadata filter.

    Returns:
        List of relevant document chunks with scores.
    """
    store = OpenShieldVectorStore(persist_dir=persist_dir)
    model = get_embedder()
    return store.query(query, model=model, n_results=n_results,
                       filter_metadata=filter_metadata)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import sys
    command = sys.argv[1] if len(sys.argv) > 1 else "build"

    if command == "build":
        build_pipeline()
    elif command == "refresh":
        refresh_pipeline()
    elif command == "query":
        query = sys.argv[2] if len(sys.argv) > 2 else "What is Network Watcher?"
        results = query_pipeline(query)
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (score: {r['score']}) ---")
            print(r["content"][:300])
    else:
        print("Usage: python -m ai.pipeline [build|refresh|query <text>]")
