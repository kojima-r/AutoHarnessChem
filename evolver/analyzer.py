"""失敗分類。SDK横断の trace (traces/*.jsonl) と benchmark 結果を解析する。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

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


class AnalysisReport(BaseModel):
    n_runs: int
    failures: list[FailureRecord] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


_SKILL_BY_CATEGORY = {
    "scf_failed": "pyscf-orbitals",
    "timeout": "pyscf-orbitals",
    "verification_missing_output": "result-reporting",
    "tool_error": "execution-recovery",
    "missing_dependency": "execution-recovery",
}


def analyze_traces(traces_dir: Path) -> AnalysisReport:
    failures: list[FailureRecord] = []
    run_ids = set()

    for trace_file in sorted(Path(traces_dir).glob("*.jsonl")):
        run_id = trace_file.stem
        run_ids.add(run_id)
        provider = "unknown"
        for line in trace_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            payload = event.get("payload", {})
            if event["event_type"] == "reasoning_summary" and "provider" in payload:
                provider = payload["provider"]
            if event["event_type"] == "error":
                category = "adapter_unavailable" if "unavailable" in str(payload) else "unknown"
                failures.append(FailureRecord(
                    run_id=run_id, provider=provider, category=category,
                    evidence=str(payload)[:500],
                ))
            if event["event_type"] == "tool_result" and payload.get("status") in ("failed", "blocked"):
                error_type = payload.get("error_type") or "tool_error"
                category = error_type if error_type in FAILURE_CATEGORIES else "tool_error"
                failures.append(FailureRecord(
                    run_id=run_id, provider=provider, category=category,
                    evidence=f"{payload.get('tool')}: {payload.get('summary', '')}"[:500],
                    skill_hint=_SKILL_BY_CATEGORY.get(category),
                ))
            if event["event_type"] == "reasoning_summary" and event.get("actor") == "verifier":
                if not payload.get("passed", True):
                    for item in payload.get("requirements_missing", []):
                        failures.append(FailureRecord(
                            run_id=run_id, provider=provider,
                            category="verification_missing_output",
                            evidence=item[:500],
                            skill_hint=_SKILL_BY_CATEGORY["verification_missing_output"],
                        ))
                    for warning in payload.get("scientific_warnings", []):
                        failures.append(FailureRecord(
                            run_id=run_id, provider=provider,
                            category="scientific_warning", evidence=warning[:500],
                            skill_hint="scientific-verification",
                        ))

    counts = Counter(f.category for f in failures)
    return AnalysisReport(n_runs=len(run_ids), failures=failures, counts=dict(counts))
