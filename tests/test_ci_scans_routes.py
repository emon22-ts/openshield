"""Route tests for CI/CD scans (#259 PR 3/3). No database or network access."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

import api.routes.ci_scans as ci_routes
from api.validation import VALIDATION_ERROR_MESSAGE

_SCAN_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture(autouse=True)
def _no_allowlist(monkeypatch):
    monkeypatch.delenv(ci_routes._ALLOWLIST_ENV, raising=False)


def _db(**attrs):
    db = MagicMock()
    db.dsn = "postgresql://unused"
    for name, value in attrs.items():
        getattr(db, name).return_value = value
    return db


def _trigger(client, headers, body):
    return client.post("/api/ci-scans/trigger", json=body, headers=headers)


def test_trigger_queues_scan_and_returns_202(client, auth_headers):
    db = _db()
    with (
        patch.object(ci_routes, "_get_ci_db", return_value=db),
        patch("api.routes.ci_scans.threading.Thread") as thread,
    ):
        response = _trigger(client, auth_headers, {"owner": "my-org", "repo": "my-repo"})
    assert response.status_code == 202
    body = response.get_json()
    assert body["status"] == "pending"
    uuid.UUID(body["scan_id"])
    db.create_pending_ci_scan.assert_called_once_with(body["scan_id"], "my-org", "my-repo")
    thread.return_value.start.assert_called_once()


@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        {"owner": "my-org"},
        {"repo": "my-repo"},
        {"owner": "-bad", "repo": "r"},
        {"owner": "o", "repo": ".."},
        {"owner": "o", "repo": "a/b"},
        {"owner": "o", "repo": "r", "token": "ghp_x"},
        {"owner": 1, "repo": "r"},
    ],
)
def test_trigger_rejects_invalid_bodies_before_database(client, auth_headers, body):
    with patch.object(ci_routes, "_get_ci_db") as get_db:
        response = _trigger(client, auth_headers, body)
    assert response.status_code == 400
    assert response.get_json() == {"error": VALIDATION_ERROR_MESSAGE}
    get_db.assert_not_called()


def test_trigger_requires_authentication(client):
    assert client.post("/api/ci-scans/trigger", json={"owner": "o", "repo": "r"}).status_code == 401


def test_trigger_enforces_repository_allowlist(client, auth_headers, monkeypatch):
    monkeypatch.setenv(ci_routes._ALLOWLIST_ENV, "my-org/allowed, other-org/*")
    with patch.object(ci_routes, "_get_ci_db") as get_db:
        denied = _trigger(client, auth_headers, {"owner": "my-org", "repo": "blocked"})
    assert denied.status_code == 403
    get_db.assert_not_called()

    with patch.object(ci_routes, "_get_ci_db", return_value=_db()), patch("api.routes.ci_scans.threading.Thread"):
        assert _trigger(client, auth_headers, {"owner": "my-org", "repo": "allowed"}).status_code == 202
        assert _trigger(client, auth_headers, {"owner": "Other-Org", "repo": "anything"}).status_code == 202


def test_trigger_database_failure_returns_500_without_starting_scan(client, auth_headers):
    db = _db()
    db.create_pending_ci_scan.side_effect = RuntimeError("secret connection string")
    with (
        patch.object(ci_routes, "_get_ci_db", return_value=db),
        patch("api.routes.ci_scans.threading.Thread") as thread,
    ):
        response = _trigger(client, auth_headers, {"owner": "o", "repo": "r"})
    assert response.status_code == 500
    assert "secret" not in response.get_data(as_text=True)
    thread.assert_not_called()


def test_get_scan_rejects_non_uuid_before_database(client, auth_headers):
    with patch.object(ci_routes, "_get_ci_db") as get_db:
        response = client.get("/api/ci-scans/not-a-uuid", headers=auth_headers)
    assert response.status_code == 400
    get_db.assert_not_called()


def test_get_scan_not_found(client, auth_headers):
    with patch.object(ci_routes, "_get_ci_db", return_value=_db(get_ci_scan=None)):
        assert client.get(f"/api/ci-scans/{_SCAN_ID}", headers=auth_headers).status_code == 404


def test_get_scan_returns_scan_with_evaluations(client, auth_headers):
    scan = {"scan_id": _SCAN_ID, "owner": "my-org", "repo": "my-repo", "status": "completed"}
    evaluations = [{"rule_id": "AZ-CI-001", "status": "PASS"}]
    db = _db(get_ci_scan=scan, get_ci_scan_evaluations=evaluations)
    with patch.object(ci_routes, "_get_ci_db", return_value=db):
        response = client.get(f"/api/ci-scans/{_SCAN_ID}", headers=auth_headers)
    assert response.status_code == 200
    assert response.get_json()["evaluations"] == evaluations


def test_get_scan_outside_allowlist_is_reported_missing(client, auth_headers, monkeypatch):
    monkeypatch.setenv(ci_routes._ALLOWLIST_ENV, "my-org/allowed")
    scan = {"scan_id": _SCAN_ID, "owner": "my-org", "repo": "secret-repo"}
    with patch.object(ci_routes, "_get_ci_db", return_value=_db(get_ci_scan=scan)):
        assert client.get(f"/api/ci-scans/{_SCAN_ID}", headers=auth_headers).status_code == 404


def test_list_filters_to_allowlisted_repositories(client, auth_headers, monkeypatch):
    monkeypatch.setenv(ci_routes._ALLOWLIST_ENV, "my-org/a")
    rows = [{"owner": "my-org", "repo": "a"}, {"owner": "my-org", "repo": "b"}]
    with patch.object(ci_routes, "_get_ci_db", return_value=_db(get_ci_scans=rows)):
        response = client.get("/api/ci-scans", headers=auth_headers)
    assert response.get_json() == [{"owner": "my-org", "repo": "a"}]


def test_list_rejects_invalid_filter(client, auth_headers):
    with patch.object(ci_routes, "_get_ci_db") as get_db:
        response = client.get("/api/ci-scans?owner=../x", headers=auth_headers)
    assert response.status_code == 400
    get_db.assert_not_called()
