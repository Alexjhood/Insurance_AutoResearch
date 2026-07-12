from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from flightdeck.server import main


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    snapshots = tmp_path / "snapshots"
    files = snapshots / "orch-1" / "files"
    files.mkdir(parents=True)
    (snapshots / "index.json").write_text(json.dumps({"built_at": "now", "orchestrations": [{"orch_id": "orch-1"}]}))
    (snapshots / "orch-1" / "snapshot.json").write_text(json.dumps({"campaign": {"orch_id": "orch-1"}}))
    (snapshots / "orch-1" / "telemetry_d01.json").write_text(json.dumps({"delegation_id": "d01"}))
    (files / "report.md").write_text("# Report")
    monkeypatch.setattr(main, "SNAPSHOTS_DIR", snapshots)
    return TestClient(main.app)


def test_index_health_and_orchestration(client: TestClient) -> None:
    assert client.get("/api/index").json()["orchestrations"][0]["orch_id"] == "orch-1"
    assert client.get("/api/orchestrations/orch-1").json()["campaign"]["orch_id"] == "orch-1"
    assert client.get("/healthz").json() == {"status": "ok", "snapshot_count": 1, "built_at": "now"}


def test_telemetry_and_file(client: TestClient) -> None:
    assert client.get("/api/orchestrations/orch-1/telemetry/d01").json()["delegation_id"] == "d01"
    response = client.get("/api/orchestrations/orch-1/files/report.md")
    assert response.text == "# Report"
    assert response.headers["content-type"].startswith("text/markdown")


@pytest.mark.parametrize("path", [
    "/api/orchestrations/missing",
    "/api/orchestrations/orch-1/telemetry/d99",
    "/api/orchestrations/orch-1/files/missing.txt",
])
def test_not_found(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404


def test_file_path_traversal_rejected(client: TestClient) -> None:
    response = client.get("/api/orchestrations/orch-1/files/%2e%2e/snapshot.json")
    assert response.status_code in {400, 404}


def test_rebuild_sse(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_stream():
        yield main._sse("progress", {"message": "working"})
        yield main._sse("complete", {"refresh": True})

    monkeypatch.setattr(main, "_rebuild_stream", fake_stream)
    response = client.post("/api/etl/rebuild")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: complete" in response.text
