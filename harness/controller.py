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

import asyncio
import inspect
import re
import shutil
from pathlib import Path

from pydantic import BaseModel

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

# タスク種別の判定ルール（Task Interpreter）。上から順に最初に一致したものを採用する
_TASK_TYPE_RULES: list[tuple[str, str]] = [
    # 明示的なMLワークフロー（回帰・交差検証）は reaction 系キーワードより優先する
    (r"回帰|regression|cross.?valid|交差検証|機械学習", "molecular_regression"),
    # 多段の経路探索（AiZynthFinder）は 1 段階の逆合成予測（ReactionT5）より具体的。
    # 「合成可能な化合物を（経路を出力して）」のような自然な言い回しも拾う
    (r"合成経路|逆合成経路|合成ルート|合成法|合成可能|合成でき|合成の?(可否|手順|方法)"
     r"|経路.{0,4}(出力|提案|提示|示|探索)|retrosynthe\w*\s*(route|plan)"
     r"|synthe\w*\s*(route|plan|path)|route\s*(search|planning)|aizynth|多段",
     "retrosynthesis_planning"),
    (r"収率|逆合成|レトロ合成|retrosynthesis|生成物.{0,4}予測|反応予測|reactiont5", "reaction_prediction"),
    (r"予測モデル|target.{0,8}予測", "molecular_regression"),
    (r"esipt|pes.{0,2}スキャン|pes.?scan|ポテンシャル.{0,4}曲面|プロトン移動|反応経路.{0,4}スキャン",
     "pes_scan"),
    # 分子・材料の探索（設計）。「骨格をベースに…化合物を探す」も対象にする
    (r"分子設計|材料設計|材料探索|optuna|スクリーニング|screening"
     r"|(分子|化合物|候補|誘導体|材料).{0,6}(探索|を探|設計|絞り込|列挙|提案)"
     r"|目標.{0,6}波長|target.{0,10}wavelength|(骨格|scaffold).{0,12}(ベース|基に|もとに|から)"
     r"|波長.{0,6}(探索|最適化)", "molecular_design"),
    (r"homo|lumo|軌道|orbital|励起|吸収スペクトル|uv.?vis|tddft|エネルギー計算|scf|dft",
     "orbital_calculation"),
    (r"データセット|dataset|データ.{0,4}(確認|調査|inspect)|欠損|統計量", "dataset_analysis"),
    (r"リファクタ|refactor|コード修正|bug|バグ", "code_editing"),
    (r"文献|調査レポート|research|survey", "long_running_research"),
]

# 試行が時間切れになったときに次の試行へ渡す修復指示
_TIMEOUT_REPAIR = (
    "前回の試行は実時間の上限で打ち切られました。次は (1) 重い計算を分割する"
    "（分子は1件ずつ、基底関数・励起状態数・trial 数を下げる）、"
    "(2) 途中結果を毎回ファイルへ保存してから次に進む、"
    "(3) sleep による待機・ポーリングをしない、"
    "(4) 残り時間が足りない場合は部分的な結果でも report_user.md / report_user.html を"
    "先に作る、の順で進めてください。"
)

_DEFAULT_EXPECTED_OUTPUTS = {
    "orbital_calculation": ["orbital_features.csv"],
    "molecular_design": ["optimization_summary.json", "optuna_trials.csv"],
    "pes_scan": ["esipt_scan_results.csv", "esipt_pes_profile.png"],
    "molecular_regression": ["cv_metrics.json", "true_vs_pred.png"],
    "reaction_prediction": ["reactiont5_predictions.csv"],
    "retrosynthesis_planning": ["retrosynthesis_routes.json", "retrosynthesis_routes.csv"],
    "dataset_analysis": [],
    "generic": [],
}


def detect_task_types(request: str) -> list[str]:
    """要求に含まれるタスクの側面を、ルール順に重複なく列挙する。

    複合タスク（例「骨格から吸収波長で分子を探索し、合成経路も出す」）では
    先頭を primary、残りを secondary として扱い、Skill と検査を両方に効かせる。
    """
    lowered = request.lower()
    detected: list[str] = []
    for pattern, candidate in _TASK_TYPE_RULES:
        if candidate not in detected and re.search(pattern, lowered):
            detected.append(candidate)
    return detected


def interpret_task(request: str, inputs: dict | None = None,
                   task_type: str | None = None,
                   expected_outputs: list[str] | None = None) -> TaskSpec:
    """ユーザ要求 → TaskSpec。達成条件・期待出力をルールベースで補完する。"""
    lowered = request.lower()
    detected = detect_task_types(request)
    if task_type is None:
        task_type = detected[0] if detected else "generic"
    # 明示指定された task_type は primary。検出された他の側面は secondary に回す
    secondary = [t for t in detected if t != task_type]
    if task_type == "molecular_regression" and re.search(r"homo|lumo|軌道|orbital", lowered):
        # 回帰タスクでも軌道特徴量が必要なら orbital 出力も要求する
        expected = expected_outputs or (
            _DEFAULT_EXPECTED_OUTPUTS["molecular_regression"] + ["orbital_features.csv"]
        )
    else:
        expected = expected_outputs or list(_DEFAULT_EXPECTED_OUTPUTS.get(task_type, []))

    if secondary and not expected_outputs:
        # 複合タスクでは各側面の成果物を必須にはしない（誤検出で詰むのを避ける）代わりに、
        # すべてに触れた報告を必須にする
        expected = expected + [o for o in ("report_user.md", "report_user.html")
                               if o not in expected]

    criteria = [f"出力ファイル `{o}` が workspace に存在すること" for o in expected]
    criteria.append("Scientific Verifier の構造化判定に合格すること")
    if secondary:
        aspects = ", ".join(secondary)
        criteria.append(
            f"要求に含まれる他の側面（{aspects}）にも回答すること"
            "（対応する成果物を作り、報告で言及する）"
        )

    return TaskSpec(
        description=request,
        task_type=task_type,  # type: ignore[arg-type]
        secondary_task_types=secondary,  # type: ignore[arg-type]
        inputs=inputs or {},
        expected_outputs=expected,
        success_criteria=criteria,
    )


class TimeoutDecision(BaseModel):
    """実時間上限に達したときの判断（延長する秒数、または打ち切り）。"""
    extend_sec: int = 0

    @property
    def keep_waiting(self) -> bool:
        return self.extend_sec > 0


class HarnessController:
    def __init__(self, config: HarnessConfig):
        self.config = config
        self.skill_registry = SkillRegistry(config.paths.skills)
        self.verifier = ScientificVerifier()

    async def _decide_on_timeout(self, info: dict, on_timeout, tracer) -> TimeoutDecision:
        """上限到達時に「さらに待つか」を決める。

        優先順:
          1. 呼び出し側が渡した on_timeout（CLI の対話プロンプト / Web UI の待機）
          2. config の on_attempt_timeout（extend / stop）
          3. ask なのに確認手段が無い場合は stop（run が終わらなくなるのを避ける）
        数日かかる計算を無人で待つ場合は on_attempt_timeout: extend を使う。
        """
        runtime = self.config.runtime
        policy = runtime.on_attempt_timeout
        limit = runtime.max_timeout_extensions
        if limit is not None and info["extensions"] >= limit:
            tracer.emit("reasoning_summary", actor="controller", payload={
                "phase": "attempt_timeout_limit", **info,
                "max_timeout_extensions": limit,
            })
            return TimeoutDecision()

        if on_timeout is not None:
            tracer.emit("reasoning_summary", actor="controller",
                        payload={"phase": "attempt_timeout_pending", **info})
            answer = on_timeout(info)
            if inspect.isawaitable(answer):
                answer = await answer
            if answer is True:
                return TimeoutDecision(extend_sec=runtime.timeout_extension_sec)
            if isinstance(answer, (int, float)) and answer > 0:
                return TimeoutDecision(extend_sec=int(answer))
            return TimeoutDecision()

        if policy == "extend":
            return TimeoutDecision(extend_sec=runtime.timeout_extension_sec)
        if policy == "ask":
            tracer.emit("error", actor="controller", payload={
                "phase": "attempt_timeout_unanswered", **info,
                "error": "on_attempt_timeout=ask だが確認手段が無いため打ち切ります"
                         "（無人実行では on_attempt_timeout: extend を使ってください）",
            })
        return TimeoutDecision()

    async def _run_attempt(self, adapter, task: TaskSpec, state: RunState, tracer,
                           on_timeout) -> tuple[str, bool]:
        """1 試行を実行する。上限に達したら延長を確認し、打ち切る場合のみ中断する。

        戻り値: (最終メッセージ, 打ち切られたか)。延長中もエージェントは動き続ける
        （待ち直すだけで計算をやり直さない）。
        """
        runtime = self.config.runtime
        agent = asyncio.ensure_future(adapter.run(task, state))
        slice_sec = runtime.attempt_timeout_sec
        waited = 0
        while True:
            done, _ = await asyncio.wait({agent}, timeout=slice_sec)
            if agent in done:
                return agent.result(), False
            waited += slice_sec
            info = {
                "attempt": state.attempts,
                "waited_sec": waited,
                "attempt_timeout_sec": runtime.attempt_timeout_sec,
                "extensions": state.timeout_extensions,
                "extension_sec": runtime.timeout_extension_sec,
                "workspace": state.workspace,
            }
            previous_status = state.status
            state.status = "awaiting_decision"
            decision = await self._decide_on_timeout(info, on_timeout, tracer)
            state.status = previous_status
            if not decision.keep_waiting:
                agent.cancel()
                await asyncio.gather(agent, return_exceptions=True)
                tracer.emit("error", actor="controller", payload={
                    "phase": "attempt_timeout", **info,
                    "error": f"attempt timed out after {waited}s — "
                             "その時点の成果物で検証します",
                })
                return (f"[controller] 試行 {state.attempts} は {waited}s で"
                        "打ち切られました（時間切れ）。"), True
            state.timeout_extensions += 1
            state.extended_sec += decision.extend_sec
            slice_sec = decision.extend_sec
            tracer.emit("reasoning_summary", actor="controller", payload={
                "phase": "attempt_timeout_extended", **info,
                "extend_sec": decision.extend_sec,
                "new_deadline_sec": waited + decision.extend_sec,
            })

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
                  on_event=None, on_timeout=None) -> RunReport:
        """on_event: AgentEvent を受け取る callable。CLI/監視系の逐次表示に使う。

        on_timeout: 実時間上限に達したときに呼ばれる callable（同期/非同期どちらも可）。
        引数は経過時間や延長回数を含む dict で、戻り値は
          True        … 既定の延長幅（timeout_extension_sec）だけ待ち続ける
          秒数 (int)  … その秒数だけ待ち続ける
          False/None  … 打ち切って、その時点の成果物で検証・報告する
        None を渡した場合は config の on_attempt_timeout に従う。
        """
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
        report: RunReport | None = None

        def finalize() -> RunReport:
            """検証結果と成果物から報告を書く。例外・打ち切り時も必ず通す。"""
            nonlocal verification
            if verification is None:
                verification = self.verifier.verify(task, workspace)
            if state.status in ("pending", "running", "verifying", "repairing"):
                # 例外などで正常終了しなかった場合。ledger を running のまま残さない
                state.status = "failed"
            tracer.emit("reasoning_summary", actor="controller",
                        payload={"phase": "finished", "status": state.status,
                                 "attempts": state.attempts, "model": state.model})
            artifacts = artifacts_mgr.scan()
            ledger.close(state.status)
            built = RunReport(
                run_id=run_id, task=task, provider=chosen,  # type: ignore[arg-type]
                model=state.model,
                passed=bool(verification and verification.passed),
                attempts=state.attempts,
                timeout_extensions=state.timeout_extensions,
                extended_sec=state.extended_sec,
                verification=verification,
                artifacts=artifacts,
                final_message=final_text,
                started_at=started_at,
                finished_at=utcnow(),
            )
            (workspace / "report.json").write_text(built.model_dump_json(indent=2),
                                                   encoding="utf-8")
            (workspace / "report.md").write_text(_render_report_md(built),
                                                 encoding="utf-8")
            return built

        try:
            for attempt in range(1, self.config.runtime.max_replans + 2):
                state.attempts = attempt
                state.status = "running"
                tracer.emit("reasoning_summary", actor="controller", payload={
                    "phase": "attempt", "attempt": attempt,
                    "max_attempts": self.config.runtime.max_replans + 1,
                    "attempt_timeout_sec": self.config.runtime.attempt_timeout_sec,
                    "on_attempt_timeout": self.config.runtime.on_attempt_timeout,
                    "repairs": task.repairs,
                })
                # 1 試行に実時間の上限を設ける。上限に達したら「さらに待つか」を
                # 確認して延長でき（数日かかる計算に対応）、打ち切る場合も
                # その時点の成果物を検証・報告する
                final_text, interrupted = await self._run_attempt(
                    adapter, task, state, tracer, on_timeout)
                state.status = "verifying"
                tracer.emit("reasoning_summary", actor="controller",
                            payload={"phase": "verifying", "attempt": attempt,
                                     "interrupted": interrupted})
                artifacts_mgr.scan()
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
                task.repairs = list(verification.required_repairs)
                if interrupted:
                    task.repairs.append(_TIMEOUT_REPAIR)
        finally:
            await adapter.shutdown()
            report = finalize()
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
