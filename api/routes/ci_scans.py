"""CI/CD scan routes (#259 PR 3/3): trigger AZ-CI-* scans and read persisted results."""

import logging
import os
import re
import threading
import uuid

from flask import Blueprint, g, jsonify, request

from api.models.ci_scan import CIDatabase
from api.validation import (
    VALIDATION_ERROR_MESSAGE,
    ValidationError,
    bounded_string,
    reject_unknown_fields,
    require_json_object,
    uuid_string,
)
from scanner.ci_engine import CIScanEngine

ci_scans_bp = Blueprint("ci_scans", __name__)
logger = logging.getLogger(__name__)

_ALLOWLIST_ENV = "OPENSHIELD_AUTHORIZED_REPOSITORIES"
# GitHub login: alphanumeric or single hyphens, max 39 chars. Repo: alnum, '.', '_', '-'.
_OWNER_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})")
_REPO_RE = re.compile(r"[A-Za-z0-9._-]{1,100}")


def _repo_is_authorized(owner: str, repo: str) -> bool:
    """True unless an allowlist is configured and this owner/repo is not on it.

    OPENSHIELD_AUTHORIZED_REPOSITORIES is a comma-separated list of ``owner/repo``
    or ``owner/*`` entries. Left unset, every repository is accepted, matching how
    OPENSHIELD_AUTHORIZED_SUBSCRIPTIONS behaves for Azure scans.
    """
    raw = os.environ.get(_ALLOWLIST_ENV, "")
    allowlist = {value.strip().lower() for value in raw.split(",") if value.strip()}
    if not allowlist:
        return True
    return f"{owner}/{repo}".lower() in allowlist or f"{owner}/*".lower() in allowlist


def _get_ci_db() -> CIDatabase:
    # Stored under g.db_conn so api.app's teardown handler returns the pooled
    # connection after the request (it closes g.db and g.db_conn).
    if "db_conn" not in g:
        dsn = os.environ.get("DATABASE_URL")
        if not dsn:
            raise RuntimeError("DATABASE_URL environment variable is not set")
        db = CIDatabase(dsn)
        db.connect()
        g.db_conn = db
    return g.db_conn


def _run_scan_in_background(scan_id: str, owner: str, repo: str, dsn: str) -> None:
    """Run the scan off the request thread. run_and_persist records failures itself."""
    try:
        CIScanEngine(owner=owner, repo=repo, dsn=dsn).run_and_persist(scan_id=scan_id)
    except Exception:
        # Already logged with a traceback and marked 'failed' by run_and_persist.
        logger.error("Background CI scan %s for %s/%s did not complete", scan_id, owner, repo)


@ci_scans_bp.post("/api/ci-scans/trigger")
def trigger_ci_scan():
    """Queue an asynchronous CI/CD scan of one GitHub repository. Returns 202."""
    try:
        body = require_json_object(request.get_json(silent=True))
        reject_unknown_fields(body, {"owner", "repo"})
        owner = bounded_string(body.get("owner"), "owner", maximum=39, pattern=_OWNER_RE)
        repo = bounded_string(body.get("repo"), "repo", maximum=100, pattern=_REPO_RE)
        if repo in {".", ".."}:
            raise ValidationError("repo has an invalid format")

        if not _repo_is_authorized(owner, repo):
            logger.warning("CI scan rejected: %s/%s is not on the authorized allowlist", owner, repo)
            return jsonify({"error": "Repository is not authorized for this deployment"}), 403

        scan_id = str(uuid.uuid4())
        try:
            db = _get_ci_db()
            db.create_pending_ci_scan(scan_id, owner, repo)
        except Exception as exc:
            logger.error("Failed to create pending CI scan: %s", exc, exc_info=True)
            return jsonify({"error": "Database error"}), 500

        threading.Thread(
            target=_run_scan_in_background,
            args=(scan_id, owner, repo, db.dsn),
            daemon=True,
        ).start()
        logger.info("CI scan %s queued for %s/%s", scan_id, owner, repo)

        return jsonify(
            {"scan_id": scan_id, "status": "pending", "message": "CI scan has been queued and will start shortly."}
        ), 202

    except ValidationError:
        return jsonify({"error": VALIDATION_ERROR_MESSAGE}), 400
    except Exception as exc:
        logger.error("Critical error in trigger_ci_scan route: %s", exc, exc_info=True)
        return jsonify({"error": "Critical route failure"}), 500


@ci_scans_bp.get("/api/ci-scans")
def list_ci_scans():
    """Return recent CI scans, optionally filtered by ?owner= and ?repo=."""
    try:
        owner = request.args.get("owner")
        repo = request.args.get("repo")
        if owner is not None:
            owner = bounded_string(owner, "owner", maximum=39, pattern=_OWNER_RE)
        if repo is not None:
            repo = bounded_string(repo, "repo", maximum=100, pattern=_REPO_RE)
        scans = _get_ci_db().get_ci_scans(owner=owner, repo=repo)
        return jsonify([s for s in scans if _repo_is_authorized(s["owner"], s["repo"])])
    except ValidationError:
        return jsonify({"error": VALIDATION_ERROR_MESSAGE}), 400
    except Exception as exc:
        logger.error("Failed to list CI scans: %s", exc, exc_info=True)
        return jsonify({"error": "Failed to retrieve CI scans"}), 500


@ci_scans_bp.get("/api/ci-scans/<scan_id>")
def get_ci_scan(scan_id):
    """Return one CI scan with all of its rule evaluations."""
    try:
        scan_id = uuid_string(scan_id, "scan_id")
        db = _get_ci_db()
        scan = db.get_ci_scan(scan_id)
        # A scan outside the allowlist is reported as missing, not forbidden.
        if not scan or not _repo_is_authorized(scan["owner"], scan["repo"]):
            return jsonify({"error": "Scan not found"}), 404
        return jsonify({**scan, "evaluations": db.get_ci_scan_evaluations(scan_id)})
    except ValidationError:
        return jsonify({"error": VALIDATION_ERROR_MESSAGE}), 400
    except Exception as exc:
        logger.error("Failed to get CI scan: %s", exc, exc_info=True)
        return jsonify({"error": "Database error"}), 500
