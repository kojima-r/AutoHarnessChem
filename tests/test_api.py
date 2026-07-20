import json

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.api import create_app  # noqa: E402
from harness.config import load_config  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    config = load_config()
    config.paths.workspaces = tmp_path / "workspaces"
    config.paths.traces = tmp_path / "traces"
    config.paths.workspaces.mkdir()
    config.paths.traces.mkdir()
    return TestClient(create_app(config)), config


def test_index_serves_web_ui(client):
    c, _ = client
    response = c.get("/")
    assert response.status_code == 200
    assert "AutoHarnessChem" in response.text
    assert "/api/tasks" in response.text


def test_skills_and_providers(client):
    c, _ = client
    skills = c.get("/api/skills").json()["skills"]
    assert any(s["name"] == "pyscf-orbitals" for s in skills)
    providers = c.get("/api/providers").json()
    assert set(providers["available"]) == {"deepagents", "claude", "openai"}


def test_submit_task_runs_in_background(client):
    c, _ = client
    response = c.post("/api/tasks", json={"request": "水のHOMO/LUMOを計算して"})
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    # SDK未導入環境では即エラー終了する。TestClient はループ終了まで待つので状態は確定済み
    detail = c.get(f"/api/runs/{run_id}").json()
    assert detail["status"] in ("running", "error", "failed", "succeeded")
    runs = c.get("/api/runs").json()["runs"]
    assert any(r["run_id"] == run_id for r in runs)


def test_task_validation(client):
    c, _ = client
    assert c.post("/api/tasks", json={"request": "  "}).status_code == 400
    assert c.post("/api/tasks", json={
        "request": "x", "input_paths": ["/no/such/file.csv"],
    }).status_code == 400


def test_artifact_serving_and_traversal_guard(client):
    c, config = client
    workspace = config.paths.workspaces / "run-test1"
    workspace.mkdir()
    (workspace / "orbital_features.csv").write_text("smiles,homo_ev,lumo_ev\nO,-13.2,4.1\n")
    (workspace / "report.json").write_text(json.dumps({
        "run_id": "run-test1", "passed": True, "provider": "claude", "attempts": 1,
        "task": {"task_type": "orbital_calculation", "description": "test"},
    }))
    ok = c.get("/api/runs/run-test1/artifacts/orbital_features.csv")
    assert ok.status_code == 200 and "homo_ev" in ok.text
    evil = c.get("/api/runs/run-test1/artifacts/..%2F..%2Fskills.lock")
    assert evil.status_code in (403, 404)
    detail = c.get("/api/runs/run-test1").json()
    assert detail["status"] == "succeeded"
    assert detail["provider"] == "claude"


def test_trace_incremental(client):
    c, config = client
    trace_file = config.paths.traces / "run-test2.jsonl"
    events = [{"run_id": "run-test2", "event_type": "tool_call", "actor": "x",
               "payload": {"tool": "inspect_dataset"},
               "timestamp": "2026-07-20T00:00:00+00:00"}] * 3
    trace_file.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    first = c.get("/api/runs/run-test2/trace?after=0").json()
    assert len(first["events"]) == 3 and first["next"] == 3
    second = c.get(f"/api/runs/run-test2/trace?after={first['next']}").json()
    assert second["events"] == []
