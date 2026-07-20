"""改善候補の評価。

Proposal の patch を一時コピーした skills ツリーへ適用し、Benchmark を再実行して
Baseline と比較する。リポジトリ本体には一切書き込まない。
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from benchmarks.runner import run_benchmark
from evolver.proposer import Proposal
from harness.config import HarnessConfig


class EvaluationReport(BaseModel):
    proposal_id: str
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    success_delta: float | None = None
    latency_ratio: float | None = None


async def evaluate(
    proposal: Proposal,
    config: HarnessConfig,
    providers: list[str],
    baseline_summary: dict[str, Any],
    tags: list[str] | None = None,
) -> EvaluationReport:
    repo_root = config.paths.root
    with tempfile.TemporaryDirectory(prefix="ahc-evolve-") as tmp:
        candidate_skills = Path(tmp) / "skills"
        shutil.copytree(config.paths.skills, candidate_skills)
        _apply_patch(proposal, repo_root, candidate_skills)

        candidate_config = config.model_copy(deep=True)
        candidate_config.paths.skills = candidate_skills
        result = await run_benchmark(
            candidate_config, providers, tags=tags,
            label=f"candidate-{proposal.proposal_id}",
        )

    return _compare(proposal.proposal_id, baseline_summary, result["summary"])


def _apply_patch(proposal: Proposal, repo_root: Path, candidate_skills: Path) -> None:
    # diff のパスは skills/... 起点なので、候補ツリーの親を作って適用する
    staging = candidate_skills.parent
    patch_file = staging / "patch.diff"
    patch_file.write_text(proposal.diff, encoding="utf-8")
    subprocess.run(
        ["git", "apply", "--unsafe-paths", "--directory", str(staging), str(patch_file)],
        check=True, capture_output=True, text=True,
    )


def _compare(proposal_id: str, baseline: dict[str, Any], candidate: dict[str, Any]) -> EvaluationReport:
    def mean_rate(summary: dict[str, Any]) -> float | None:
        rates = [s["success_rate"] for s in summary.get("by_provider", {}).values()
                 if s.get("success_rate") is not None]
        return sum(rates) / len(rates) if rates else None

    def mean_latency(summary: dict[str, Any]) -> float | None:
        vals = [s["mean_elapsed_sec"] for s in summary.get("by_provider", {}).values()
                if s.get("mean_elapsed_sec") is not None]
        return sum(vals) / len(vals) if vals else None

    base_rate, cand_rate = mean_rate(baseline), mean_rate(candidate)
    base_lat, cand_lat = mean_latency(baseline), mean_latency(candidate)
    return EvaluationReport(
        proposal_id=proposal_id,
        baseline=baseline,
        candidate=candidate,
        success_delta=(cand_rate - base_rate) if None not in (base_rate, cand_rate) else None,
        latency_ratio=(cand_lat / base_lat) if base_lat and cand_lat else None,
    )
