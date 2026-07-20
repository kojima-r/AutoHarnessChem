"""採用判定（promotion gate）。

Gate を通過した Proposal のみ「人手承認待ち」として提示する。
Baseline（skills/ 本体・skills.lock）の更新は必ず人間が `git apply` で行う。
"""
from __future__ import annotations

from pydantic import BaseModel

from evolver.evaluator import EvaluationReport
from harness.config import PromotionGate


class PromotionDecision(BaseModel):
    proposal_id: str
    promoted: bool
    reasons: list[str]
    next_step: str


def decide(evaluation: EvaluationReport, gate: PromotionGate) -> PromotionDecision:
    reasons: list[str] = []

    if evaluation.success_delta is None:
        reasons.append("成功率が計測できていません（実行可能な provider がない可能性）。")
    elif evaluation.success_delta < gate.minimum_task_success_improvement:
        reasons.append(
            f"成功率改善 {evaluation.success_delta:+.3f} が閾値 "
            f"+{gate.minimum_task_success_improvement} に達していません。"
        )

    if evaluation.latency_ratio is not None and \
            evaluation.latency_ratio > 1.0 + gate.maximum_latency_increase:
        reasons.append(
            f"実行時間が {evaluation.latency_ratio:.2f}x に増加"
            f"（許容 {1.0 + gate.maximum_latency_increase:.2f}x）。"
        )

    promoted = not reasons
    if promoted and gate.human_approval_required:
        next_step = ("Gate 通過。人手レビューの上 `git apply evolver/proposals/"
                     f"{evaluation.proposal_id}/patch.diff` を実行し、"
                     "`ahc skills lock` で skills.lock を更新してください。")
    elif promoted:
        next_step = "Gate 通過。"
    else:
        next_step = "Rollback（提案は不採用。proposals/ に記録は残ります）。"

    return PromotionDecision(
        proposal_id=evaluation.proposal_id,
        promoted=promoted,
        reasons=reasons or ["all gate conditions satisfied"],
        next_step=next_step,
    )
