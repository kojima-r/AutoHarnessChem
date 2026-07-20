"""Common Harness Core — 自律実行ループ。

1. ユーザ要求を TaskSpec へ変換
2. 達成条件を設定
3. 必要な Skill を選択
4. Agent SDK を起動（フォールバック付き）
5. 計画・実行・検査
6. Verifier が結果を評価
7. 不合格なら修復指示付きで再実行（max_replans まで）
8. 合格なら Artifact と最終報告を保存
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from adapters import PROVIDERS, AdapterUnavailable, create_adapter
from harness.artifacts import ArtifactManager
from harness.config import HarnessConfig
from harness.policy import PolicyGate
from harness.skill_registry import SkillRegistry
from harness.task_ledger import TaskLedger
from harness.traces import TraceWriter
from harness.verifier import ScientificVerifier
from schemas import RunReport, RunState, TaskSpec, new_id, utcnow
from tools import build_default_registry

# タスク種別の判定ルール（Task Interpreter）
_TASK_TYPE_RULES: list[tuple[str, str]] = [
    # 明示的なMLワークフロー（回帰・交差検証）は reaction 系キーワードより優先する
    (r"回帰|regression|cross.?valid|交差検証|機械学習", "molecular_regression"),
    (r"収率|逆合成|レトロ合成|retrosynthesis|生成物.{0,4}予測|反応予測|reactiont5", "reaction_prediction"),
    (r"予測モデル|target.{0,8}予測", "molecular_regression"),
    (r"homo|lumo|軌道|orbital|励起|吸収スペクトル|エネルギー計算|scf|dft", "orbital_calculation"),
    (r"データセット|dataset|データ.{0,4}(確認|調査|inspect)|欠損|統計量", "dataset_analysis"),
    (r"リファクタ|refactor|コード修正|bug|バグ", "code_editing"),
    (r"文献|調査レポート|research|survey", "long_running_research"),
]

_DEFAULT_EXPECTED_OUTPUTS = {
    "orbital_calculation": ["orbital_features.csv"],
    "molecular_regression": ["cv_metrics.json", "true_vs_pred.png"],
    "reaction_prediction": ["reactiont5_predictions.csv"],
    "dataset_analysis": [],
    "generic": [],
}


def interpret_task(request: str, inputs: dict | None = None,
                   task_type: str | None = None,
                   expected_outputs: list[str] | None = None) -> TaskSpec:
    """ユーザ要求 → TaskSpec。達成条件・期待出力をルールベースで補完する。"""
    lowered = request.lower()
    if task_type is None:
        task_type = "generic"
        for pattern, candidate in _TASK_TYPE_RULES:
            if re.search(pattern, lowered):
                task_type = candidate
                break
    if task_type == "molecular_regression" and re.search(r"homo|lumo|軌道|orbital", lowered):
        # 回帰タスクでも軌道特徴量が必要なら orbital 出力も要求する
        expected = expected_outputs or (
            _DEFAULT_EXPECTED_OUTPUTS["molecular_regression"] + ["orbital_features.csv"]
        )
    else:
        expected = expected_outputs or list(_DEFAULT_EXPECTED_OUTPUTS.get(task_type, []))

    criteria = [f"出力ファイル `{o}` が workspace に存在すること" for o in expected]
    criteria.append("Scientific Verifier の構造化判定に合格すること")

    return TaskSpec(
        description=request,
        task_type=task_type,  # type: ignore[arg-type]
        inputs=inputs or {},
        expected_outputs=expected,
        success_criteria=criteria,
    )


class HarnessController:
    def __init__(self, config: HarnessConfig):
        self.config = config
        self.skill_registry = SkillRegistry(config.paths.skills)
        self.verifier = ScientificVerifier()

    def route(self, task: TaskSpec) -> str:
        provider = self.config.routing.get(task.task_type)
        if provider is None:
            provider = self.config.routing.get("fallback", "deepagents")
        return provider

    def _fallback_chain(self, first: str) -> list[str]:
        if not self.config.runtime_fallback:
            return [first]
        rest = [p for p in PROVIDERS if p != first]
        return [first, *rest]

    async def run(self, request: str, provider: str | None = None,
                  inputs: dict | None = None, task_type: str | None = None,
                  expected_outputs: list[str] | None = None,
                  copy_inputs: list[str] | None = None,
                  run_id: str | None = None,
                  on_event=None) -> RunReport:
        """on_event: AgentEvent を受け取る callable。CLI/監視系の逐次表示に使う。"""
        # 本番モードでは skills.lock との整合を確認する
        lock_path = self.config.paths.root / self.config.skill_lockfile
        if self.config.mode == "production" and lock_path.exists():
            drift = self.skill_registry.check_lockfile(lock_path)
            if drift:
                raise RuntimeError(f"skills.lock と不一致の Skill があります: {drift}")

        task = interpret_task(request, inputs, task_type, expected_outputs)
        chosen = provider or self.route(task)
        started_at = utcnow()

        run_id = run_id or new_id("run")
        workspace = self.config.paths.workspaces / run_id
        workspace.mkdir(parents=True, exist_ok=True)
        for src in copy_inputs or []:
            shutil.copy(src, workspace / Path(src).name)
            task.inputs.setdefault("files", []).append(Path(src).name)

        tracer = TraceWriter(self.config.paths.traces, run_id)
        if on_event is not None:
            tracer.subscribe(on_event)
        policy = PolicyGate(approval=self.config.approval, allowed_write_roots=[workspace])
        tools = build_default_registry(workspace, self.config.runtime.sandbox, policy)
        artifacts_mgr = ArtifactManager(workspace)
        ledger = TaskLedger(workspace / "ledger.json")
        skills = self.skill_registry.select(task)

        adapter = None
        for candidate in self._fallback_chain(chosen):
            try:
                adapter = create_adapter(candidate, tracer, policy)
                await adapter.create(self.config.runtime, skills, tools)
                chosen = candidate
                break
            except AdapterUnavailable as e:
                tracer.emit("error", actor="controller",
                            payload={"provider": candidate, "error": str(e)})
                adapter = None
        if adapter is None:
            raise AdapterUnavailable(chosen, "no runtime provider available (install one of the SDK extras)")

        ledger.open(task, chosen)
        state = RunState(run_id=run_id, provider=chosen, workspace=str(workspace))  # type: ignore[arg-type]
        tracer.emit("reasoning_summary", actor="controller", payload={
            "phase": "start", "run_id": run_id,
            "task_type": task.task_type, "provider": chosen,
            "skills": [s.name for s in skills],
        })

        final_text = ""
        verification = None
        try:
            for attempt in range(1, self.config.runtime.max_replans + 2):
                state.attempts = attempt
                state.status = "running"
                tracer.emit("reasoning_summary", actor="controller", payload={
                    "phase": "attempt", "attempt": attempt,
                    "max_attempts": self.config.runtime.max_replans + 1,
                    "repairs": task.repairs,
                })
                final_text = await adapter.run(task, state)
                state.status = "verifying"
                tracer.emit("reasoning_summary", actor="controller",
                            payload={"phase": "verifying", "attempt": attempt})
                artifacts = artifacts_mgr.scan()
                verification = self.verifier.verify(task, workspace)
                ledger.record_attempt(attempt, verification)
                tracer.emit("reasoning_summary", actor="verifier",
                            payload=verification.model_dump())
                if verification.passed:
                    state.status = "succeeded"
                    break
                if attempt > self.config.runtime.max_replans:
                    state.status = "failed"
                    break
                state.status = "repairing"
                task.repairs = verification.required_repairs
        finally:
            await adapter.shutdown()

        tracer.emit("reasoning_summary", actor="controller",
                    payload={"phase": "finished", "status": state.status,
                             "attempts": state.attempts})
        artifacts = artifacts_mgr.scan()
        ledger.close(state.status)
        report = RunReport(
            run_id=run_id, task=task, provider=chosen,  # type: ignore[arg-type]
            passed=bool(verification and verification.passed),
            attempts=state.attempts,
            verification=verification,
            artifacts=artifacts,
            final_message=final_text,
            started_at=started_at,
            finished_at=utcnow(),
        )
        (workspace / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        (workspace / "report.md").write_text(_render_report_md(report), encoding="utf-8")
        return report


def _render_report_md(report: RunReport) -> str:
    v = report.verification
    duration = ""
    if report.finished_at:
        duration = f"{(report.finished_at - report.started_at).total_seconds():.1f}s"
    lines = [
        f"# Run report: {report.run_id}",
        "",
        f"- provider: **{report.provider}**",
        f"- passed: **{report.passed}** (attempts: {report.attempts}, duration: {duration})",
        f"- task_type: {report.task.task_type}",
        "",
        "## Task",
        report.task.description,
        "",
        "## Verification",
        *(f"- ✅ {s}" for s in v.requirements_satisfied),
        *(f"- ❌ {m}" for m in v.requirements_missing),
        *(f"- ⚠️ {w}" for w in v.scientific_warnings),
        "",
        "## Artifacts",
        *(f"- `{a.path}` ({a.kind}, {a.bytes} bytes)" for a in report.artifacts),
        "",
        "## Final message",
        report.final_message,
    ]
    return "\n".join(lines) + "\n"
