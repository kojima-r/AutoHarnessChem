"""失敗分類 + リカバリ検出。

SDK横断の trace (traces/*.jsonl) を解析する。単なる失敗の集計に加えて、
「失敗したツール呼び出し → 後続で同じツールが成功」というリカバリを検出し、
その際に引数がどう変わったか（＝具体的な回復方法）を抽出する。
proposer はこれを使い、汎用テンプレートではなく再現可能な回復手順を Skill に恒久化する。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

FAILURE_CATEGORIES = (
    "missing_dependency",
    "timeout",
    "policy_violation",
    "scf_failed",
    "tool_error",
    "verification_missing_output",
    "scientific_warning",
    "adapter_unavailable",
    "unknown",
)


class FailureRecord(BaseModel):
    run_id: str
    provider: str = "unknown"
    category: str
    evidence: str = ""
    skill_hint: str | None = None  # 改善対象になりそうな Skill 名


class RecoveryRecord(BaseModel):
    """失敗 → 成功 のリカバリ1件。引数の差分が「具体的な回復方法」を表す。"""
    run_id: str
    provider: str = "unknown"
    tool: str
    error_type: str = "tool_error"
    failure_summary: str = ""
    failed_arguments: dict[str, Any] = Field(default_factory=dict)
    recovered_arguments: dict[str, Any] = Field(default_factory=dict)
    changed_keys: list[str] = Field(default_factory=list)


class AnalysisReport(BaseModel):
    n_runs: int
    failures: list[FailureRecord] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    recoveries: list[RecoveryRecord] = Field(default_factory=list)


_SKILL_BY_CATEGORY = {
    "scf_failed": "pyscf-orbitals",
    "timeout": "pyscf-orbitals",
    "verification_missing_output": "result-reporting",
    "tool_error": "execution-recovery",
    "missing_dependency": "execution-recovery",
}


def analyze_traces(traces_dir: Path) -> AnalysisReport:
    failures: list[FailureRecord] = []
    recoveries: list[RecoveryRecord] = []
    run_ids = set()

    for trace_file in sorted(Path(traces_dir).glob("*.jsonl")):
        run_id = trace_file.stem
        run_ids.add(run_id)
        events = [json.loads(line) for line in
                  trace_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        provider = _detect_provider(events)
        failures += _collect_failures(run_id, provider, events)
        recoveries += _detect_recoveries(run_id, provider, events)

    counts = Counter(f.category for f in failures)
    return AnalysisReport(n_runs=len(run_ids), failures=failures,
                          counts=dict(counts), recoveries=recoveries)


def _detect_provider(events: list[dict]) -> str:
    for event in events:
        if event.get("event_type") == "reasoning_summary" and "provider" in event.get("payload", {}):
            return event["payload"]["provider"]
    return "unknown"


def _collect_failures(run_id: str, provider: str, events: list[dict]) -> list[FailureRecord]:
    failures: list[FailureRecord] = []
    for event in events:
        payload = event.get("payload", {})
        event_type = event.get("event_type")
        if event_type == "error":
            category = "adapter_unavailable" if "unavailable" in str(payload) else "unknown"
            failures.append(FailureRecord(run_id=run_id, provider=provider,
                                          category=category, evidence=str(payload)[:500]))
        elif event_type == "tool_result" and payload.get("status") in ("failed", "blocked"):
            error_type = payload.get("error_type") or "tool_error"
            category = error_type if error_type in FAILURE_CATEGORIES else "tool_error"
            failures.append(FailureRecord(
                run_id=run_id, provider=provider, category=category,
                evidence=f"{payload.get('tool')}: {payload.get('summary', '')}"[:500],
                skill_hint=_SKILL_BY_CATEGORY.get(category),
            ))
        elif event_type == "reasoning_summary" and event.get("actor") == "verifier" \
                and not payload.get("passed", True):
            for item in payload.get("requirements_missing", []):
                failures.append(FailureRecord(
                    run_id=run_id, provider=provider,
                    category="verification_missing_output", evidence=item[:500],
                    skill_hint=_SKILL_BY_CATEGORY["verification_missing_output"]))
            for warning in payload.get("scientific_warnings", []):
                failures.append(FailureRecord(
                    run_id=run_id, provider=provider, category="scientific_warning",
                    evidence=warning[:500], skill_hint="scientific-verification"))
    return failures


def _detect_recoveries(run_id: str, provider: str, events: list[dict]) -> list[RecoveryRecord]:
    """ツールごとに (失敗 → その後の成功) を対にして回復方法を抽出する。"""
    # tool -> 直近の tool_call 引数（次の tool_result と対にする）
    pending_args: dict[str, dict] = {}
    attempts: dict[str, list[dict]] = {}

    for event in events:
        payload = event.get("payload", {})
        tool = payload.get("tool")
        if not tool:
            continue
        if event.get("event_type") == "tool_call":
            args = payload.get("arguments")
            pending_args[tool] = args if isinstance(args, dict) else {}
        elif event.get("event_type") == "tool_result":
            attempts.setdefault(tool, []).append({
                "args": pending_args.get(tool, {}),
                "status": payload.get("status"),
                "summary": payload.get("summary", ""),
                "error_type": payload.get("error_type") or "tool_error",
            })

    recoveries: list[RecoveryRecord] = []
    for tool, sequence in attempts.items():
        last_failure: dict | None = None
        for attempt in sequence:
            if attempt["status"] in ("failed", "blocked"):
                last_failure = attempt
            elif attempt["status"] == "success" and last_failure is not None:
                recoveries.append(_make_recovery(run_id, provider, tool, last_failure, attempt))
                last_failure = None  # 同じ失敗を次の成功で二重計上しない
    return recoveries


def _make_recovery(run_id: str, provider: str, tool: str,
                   failure: dict, success: dict) -> RecoveryRecord:
    failed_args = failure.get("args") or {}
    recovered_args = success.get("args") or {}
    changed = [k for k in set(failed_args) | set(recovered_args)
               if str(failed_args.get(k)) != str(recovered_args.get(k))]
    return RecoveryRecord(
        run_id=run_id, provider=provider, tool=tool,
        error_type=failure.get("error_type", "tool_error"),
        failure_summary=str(failure.get("summary", ""))[:300],
        failed_arguments=failed_args, recovered_arguments=recovered_args,
        changed_keys=sorted(changed),
    )
