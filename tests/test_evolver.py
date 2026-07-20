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


def _call(tool, args):
    return {"event_type": "tool_call", "payload": {"tool": tool, "arguments": args}}


def _result(tool, status, **kw):
    return {"event_type": "tool_result",
            "payload": {"tool": tool, "status": status, **kw}}


def test_recovery_produces_concrete_proposal_not_generic(tmp_path):
    """失敗→成功が観測されたら、汎用文ではなく実際の引数変更を恒久化する。"""
    traces = tmp_path / "traces"
    traces.mkdir()
    # calculate_orbitals が basis=6-31g* で SCF 失敗 → sto-3g に変えて成功
    events = [
        _call("calculate_orbitals", {"smiles": "['C1=CC=CC=C1']", "basis": "6-31g*"}),
        _result("calculate_orbitals", "failed", error_type="scf_failed",
                summary="SCF not converged"),
        _call("calculate_orbitals", {"smiles": "['C1=CC=CC=C1']", "basis": "sto-3g"}),
        _result("calculate_orbitals", "success", summary="1/1 molecules"),
    ]
    _write_trace(traces, "run-r", events)

    analysis = analyze_traces(traces)
    assert len(analysis.recoveries) == 1
    recovery = analysis.recoveries[0]
    assert recovery.tool == "calculate_orbitals"
    assert recovery.changed_keys == ["basis"]

    proposals = propose(analysis, REPO_ROOT, REPO_ROOT / "skills")
    assert len(proposals) == 1
    proposal = proposals[0]
    # required_tools 経由で calculate_orbitals を持つドメイン Skill が対象
    assert proposal.target_file == "skills/pyscf-orbitals/SKILL.md"
    assert proposal.category == "recovery:scf_failed"
    # 具体的な引数変更が diff 本文に含まれる（汎用テンプレートではない）
    assert "6-31g*" in proposal.diff and "sto-3g" in proposal.diff
    assert "basis" in proposal.diff
    # 汎用 scf_failed テンプレートの文言は使われていない
    assert "mf.level_shift=0.2" not in proposal.diff


def test_generic_suppressed_when_recovery_present(tmp_path):
    """同一 (skill, category) に具体的リカバリがあれば汎用提案は出さない。"""
    traces = tmp_path / "traces"
    traces.mkdir()
    recovered = [
        _call("calculate_orbitals", {"basis": "6-31g*"}),
        _result("calculate_orbitals", "failed", error_type="scf_failed", summary="no conv"),
        _call("calculate_orbitals", {"basis": "sto-3g"}),
        _result("calculate_orbitals", "success", summary="ok"),
    ]
    _write_trace(traces, "run-1", recovered)
    # scf_failed の別の失敗（リカバリなし）も存在させる
    _write_trace(traces, "run-2", [
        _result("calculate_orbitals", "failed", error_type="scf_failed", summary="no conv")])

    analysis = analyze_traces(traces)
    proposals = propose(analysis, REPO_ROOT, REPO_ROOT / "skills")
    categories = [p.category for p in proposals]
    assert "recovery:scf_failed" in categories
    # 汎用 scf_failed 提案は抑制される
    assert "scf_failed" not in categories


def test_transient_retry_recovery_guidance(tmp_path):
    """引数を変えず再試行で成功した場合は『一時的エラー、再試行可』と記録する。"""
    traces = tmp_path / "traces"
    traces.mkdir()
    same_args = {"code": "print(1)"}
    _write_trace(traces, "run-t", [
        _call("run_python_sandbox", same_args),
        _result("run_python_sandbox", "failed", error_type="runtime_error", summary="flaky"),
        _call("run_python_sandbox", same_args),
        _result("run_python_sandbox", "success", summary="ok"),
    ])
    analysis = analyze_traces(traces)
    assert analysis.recoveries[0].changed_keys == []
    proposals = propose(analysis, REPO_ROOT, REPO_ROOT / "skills")
    assert proposals and "再試行" in proposals[0].diff
