"""AgentRuntimeAdapter — 共通Runtimeインターフェース。

SDK固有のイベントは各Adapterが TraceWriter 経由で AgentEvent に正規化する。
SDKが未インストールの場合は AdapterUnavailable を送出し、Controller が
runtime_fallback 設定に従って別プロバイダへ切り替える。
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from harness.policy import PolicyGate
from harness.skill_registry import Skill
from harness.traces import TraceWriter, clip, read_trace
from schemas import AgentEvent, RunState, RuntimeConfig, TaskSpec, ToolResult
from tools.registry import ToolRegistry


class AdapterUnavailable(RuntimeError):
    def __init__(self, provider: str, hint: str):
        super().__init__(f"provider `{provider}` is unavailable: {hint}")
        self.provider = provider


class BaseAdapter(ABC):
    name: str = "base"

    def __init__(self, tracer: TraceWriter, policy: PolicyGate):
        self.tracer = tracer
        self.policy = policy
        self.config: RuntimeConfig | None = None
        self.skills: list[Skill] = []
        self.tools: ToolRegistry | None = None

    # --- AgentRuntimeAdapter interface ---------------------------------

    async def create(self, config: RuntimeConfig, skills: list[Skill], tools: ToolRegistry) -> None:
        self.config = config
        self.skills = skills
        self.tools = tools
        await self._create_impl()

    @abstractmethod
    async def _create_impl(self) -> None: ...

    @abstractmethod
    async def run(self, task: TaskSpec, state: RunState) -> str:
        """タスクを1回実行し、最終メッセージを返す。イベントは tracer へ。"""

    async def resume(self, run_id: str) -> str:
        raise NotImplementedError(f"{self.name} adapter does not support resume yet")

    async def delegate(self, task: TaskSpec, agent_type: str) -> str:
        raise NotImplementedError(f"{self.name} adapter does not support delegation yet")

    async def request_approval(self, action: str) -> bool:
        approved = self.policy.approve(action)
        self.tracer.emit("reasoning_summary", actor=self.name,
                         payload={"approval_request": action, "approved": approved})
        return approved

    async def collect_trace(self, run_id: str) -> list[AgentEvent]:
        return read_trace(self.tracer.path)

    async def shutdown(self) -> None:
        pass

    # --- shared helpers -------------------------------------------------

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """全SDK共通のツール実行経路。trace 正規化もここで行う。"""
        assert self.tools is not None
        self.tracer.emit("tool_call", actor=self.name,
                         payload={"tool": name, "arguments": _clip_args(arguments)})
        spec = self.tools.get(name) if name in self.tools.names() else None
        if spec is not None and spec.risk_level == "high" and not self.policy.approve(f"tool:{name}"):
            result = ToolResult(status="blocked", summary=f"tool `{name}` requires approval",
                                error_type="approval_denied")
        else:
            result = self.tools.call(name, **arguments)
        self.tracer.emit("tool_result", actor=self.name, payload={
            "tool": name, "status": result.status, "summary": clip(result.summary, 500),
            "error_type": result.error_type,
        })
        for artifact in result.artifacts:
            self.tracer.emit("artifact", actor=self.name, payload=artifact.model_dump())
        return result

    def tool_result_json(self, result: ToolResult) -> str:
        return json.dumps(result.model_dump(), ensure_ascii=False, default=str)

    def build_system_prompt(self, task: TaskSpec) -> str:
        skill_sections = []
        for skill in self.skills:
            skill_sections.append(
                f"## Skill: {skill.name} (v{skill.version})\n"
                f"{skill.description}\n\n{skill.body}"
            )
        tool_lines = [f"- {s.name}: {s.description}" for s in (self.tools.specs() if self.tools else [])]
        return (
            "あなたは量子化学・機械学習タスクを自律実行するエージェントです。\n"
            "作業はすべて workspace（カレントディレクトリ）内で行い、"
            "成果物ファイルは workspace 直下に保存してください。\n"
            "sandbox の conda 環境・timeout・作業ディレクトリは固定で変更できません。\n"
            "パッケージ不足 (ModuleNotFoundError) は修復不能として報告してください。\n\n"
            f"# 利用可能ツール\n" + "\n".join(tool_lines) + "\n\n"
            f"# タスク達成条件\n"
            + "\n".join(f"- {c}" for c in task.success_criteria or ["タスク記述の要求をすべて満たすこと"])
            + "\n\n# 期待される出力ファイル\n"
            + "\n".join(f"- {o}" for o in task.expected_outputs or ["(指定なし)"])
            + "\n\n# ロード済み Skill\n\n"
            + "\n\n---\n\n".join(skill_sections)
        )

    def build_task_prompt(self, task: TaskSpec) -> str:
        prompt = f"タスク:\n{task.description}\n"
        if task.inputs:
            prompt += f"\n入力:\n{json.dumps(task.inputs, ensure_ascii=False, indent=2)}\n"
        if task.repairs:
            prompt += (
                "\n前回の実行は Verifier に不合格でした。以下の修復を必ず行ってください:\n"
                + "\n".join(f"- {r}" for r in task.repairs)
            )
        return prompt


def _clip_args(arguments: dict[str, Any], limit: int = 500) -> dict[str, Any]:
    return {k: clip(v, limit) for k, v in arguments.items()}
