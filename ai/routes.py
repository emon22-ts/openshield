"""
Flask routes for the OpenShield RAG pipeline API.
Exposes endpoints for querying the vector store
and triggering pipeline refresh.
"""

import logging

from flask import Blueprint, jsonify, request

from ai.pipeline import query_pipeline, refresh_pipeline

logger = logging.getLogger(__name__)

ai_bp = Blueprint("ai", __name__, url_prefix="/api/ai")


@ai_bp.get("/query")
def query():
    """
    Query the vector store for relevant documents.

    Query params:
        q: The search query string (required)
        n: Number of results to return (default 5)
        source: Filter by source type (optional)

    Returns:
        JSON list of relevant document chunks with scores.
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    n = int(request.args.get("n", 5))
    source = request.args.get("source")

    filter_metadata = {"source": source} if source else None

    try:
        results = query_pipeline(
            query=q,
            n_results=n,
            filter_metadata=filter_metadata,
        )
        return jsonify({
            "query": q,
            "count": len(results),
            "results": results,
        })
    except Exception as exc:
        logger.error("RAG query failed: %s", exc)
        return jsonify({"error": "Query failed", "detail": str(exc)}), 500


@ai_bp.post("/refresh")
def refresh():
    """
    Trigger a vector store refresh.

    Reloads all documents, re-embeds them, and upserts
    into the existing vector store without full rebuild.

    Returns:
        JSON with refresh status and chunk count.
    """
    try:
        store = refresh_pipeline()
        return jsonify({
            "status": "ok",
            "message": "Vector store refreshed successfully",
            "chunks_in_store": store.count(),
        })
    except Exception as exc:
        logger.error("RAG refresh failed: %s", exc)
        return jsonify({"error": "Refresh failed", "detail": str(exc)}), 500


@ai_bp.get("/status")
def status():
    """
    Return vector store status.

    Returns:
        JSON with document count in the vector store.
    """
    try:
        from ai.vector_store.chroma_store import OpenShieldVectorStore
        store = OpenShieldVectorStore()
        return jsonify({
            "status": "ok",
            "chunks_in_store": store.count(),
        })
    except Exception as exc:
        logger.error("RAG status check failed: %s", exc)
        return jsonify({"error": "Status check failed", "detail": str(exc)}), 500
