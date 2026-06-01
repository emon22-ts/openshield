"""
Embedding pipeline for OpenShield document chunks.
Converts text chunks into vector embeddings using
sentence-transformers for local inference with no API cost.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Model used for embeddings — lightweight and fast
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedder():
    """
    Load and return the sentence-transformers embedding model.

    Returns:
        SentenceTransformer model instance.
    """
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
        return model
    except ImportError:
        logger.error(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        )
        raise


def embed_chunks(
    chunks: List[Dict[str, Any]],
    model=None,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for all chunks.

    Args:
        chunks: List of chunk dicts with content and metadata.
        model: Optional pre-loaded SentenceTransformer model.
        batch_size: Number of chunks to embed at once.

    Returns:
        List of chunk dicts with embeddings added.
    """
    if model is None:
        model = get_embedder()

    texts = [chunk["content"] for chunk in chunks]

    logger.info("Embedding %d chunks in batches of %d...", len(texts), batch_size)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_list=True,
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    logger.info("Embedding complete — %d vectors generated", len(embeddings))
    return chunks
