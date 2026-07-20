import json
from pathlib import Path

import pytest

from evolver.analyzer import analyze_traces
from evolver.guard import assert_allowed, classify
from evolver.proposer import propose

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_guard_allows_skill_md():
    assert classify("skills/pyscf-orbitals/SKILL.md", REPO_ROOT) == "allowed"


def test_guard_forbids_verifier_benchmarks_and_itself():
    assert classify("harness/verifier.py", REPO_ROOT) == "forbidden"
    assert classify("benchmarks/tasks.yaml", REPO_ROOT) == "forbidden"
    assert classify("evolver/guard.py", REPO_ROOT) == "forbidden"
    with pytest.raises(PermissionError):
        assert_allowed(["harness/verifier.py"], REPO_ROOT)


def test_guard_treats_unlisted_as_not_allowed():
    assert classify("harness/controller.py", REPO_ROOT) == "unlisted"
    with pytest.raises(PermissionError):
        assert_allowed(["harness/controller.py"], REPO_ROOT)


def _write_trace(path: Path, run_id: str, events: list[dict]):
    with (path / f"{run_id}.jsonl").open("w") as fp:
        for event in events:
            fp.write(json.dumps({"run_id": run_id, "actor": "test",
                                 "timestamp": "2026-07-20T00:00:00+00:00", **event}) + "\n")


def test_analyze_and_propose_from_timeouts(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    timeout_event = {
        "event_type": "tool_result",
        "payload": {"tool": "run_python_sandbox", "status": "failed",
                    "error_type": "timeout", "summary": "timed out"},
    }
    _write_trace(traces, "run-a", [timeout_event])
    _write_trace(traces, "run-b", [timeout_event])

    analysis = analyze_traces(traces)
    assert analysis.counts.get("timeout") == 2

    proposals = propose(analysis, REPO_ROOT, REPO_ROOT / "skills")
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.target_file == "skills/pyscf-orbitals/SKILL.md"
    assert proposal.diff.startswith("--- a/skills/pyscf-orbitals/SKILL.md")
    # 提案は diff のみ。正本の SKILL.md が変更されていないこと
    assert "[evolver候補" not in (REPO_ROOT / "skills/pyscf-orbitals/SKILL.md").read_text()
