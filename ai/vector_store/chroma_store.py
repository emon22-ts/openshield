"""ChromaDB vector store for OpenShield."""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "openshield_docs"


class OpenShieldVectorStore:
    def __init__(self, persist_dir=None):
        self.persist_dir = str(persist_dir or DEFAULT_PERSIST_DIR)
        self._client = None
        self._collection = None
        self._connect()

    def _connect(self):
        import chromadb
        from chromadb.config import Settings
        os.makedirs(self.persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Connected to ChromaDB — %d documents", self._collection.count())

    def ingest(self, chunks):
        if not chunks:
            return
        ids = [c["id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas = [{k: str(v) for k, v in c.get("metadata", {}).items()} for c in chunks]
        for i in range(0, len(ids), 100):
            self._collection.upsert(
                ids=ids[i:i+100],
                documents=documents[i:i+100],
                embeddings=embeddings[i:i+100],
                metadatas=metadatas[i:i+100],
            )
        logger.info("Ingested %d chunks", len(chunks))

    def query(self, query_text, model=None, n_results=5, filter_metadata=None):
        if model is None:
            from ai.embedder import get_embedder
            model = get_embedder()
        query_embedding = model.encode([query_text], convert_to_list=True)[0]
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata
        results = self._collection.query(**kwargs)
        output = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            output.append({"content": doc, "metadata": meta, "score": round(1 - dist, 4)})
        return output

    def refresh(self, chunks):
        logger.info("Refreshing vector store...")
        self.ingest(chunks)
        logger.info("Refresh complete — %d documents", self._collection.count())

    def count(self):
        return self._collection.count()

    def clear(self):
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store cleared")
