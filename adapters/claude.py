"""Claude Agent SDK Adapter。

query / ClaudeAgentOptions / in-process MCP server (create_sdk_mcp_server) を使い、
共通ツールを MCP tool として公開する。Skill は SkillCompiler が .claude/skills に
配置済みであることを前提に setting_sources から拾わせることもできるが、
本Adapterでは system_prompt へ直接埋め込む（SDK差異を吸収するため）。
"""
from __future__ import annotations

from typing import Any

from adapters.base import AdapterUnavailable, BaseAdapter
from harness.traces import clip
from schemas import RunState, TaskSpec

_INSTALL_HINT = "pip install claude-agent-sdk (and `npm install -g @anthropic-ai/claude-code`)"


class ClaudeAgentAdapter(BaseAdapter):
    name = "claude"

    async def _create_impl(self) -> None:
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as e:
            raise AdapterUnavailable(self.name, f"{e}. {_INSTALL_HINT}") from e

    def _build_mcp_server(self):
        from claude_agent_sdk import create_sdk_mcp_server, tool

        sdk_tools = []
        for spec in self.tools.specs():
            def make_handler(tool_name: str):
                async def handler(args: dict[str, Any]) -> dict[str, Any]:
                    result = self.call_tool(tool_name, args or {})
                    return {"content": [{"type": "text", "text": self.tool_result_json(result)}]}
                return handler

            sdk_tools.append(
                tool(spec.name, spec.description, spec.parameters)(make_handler(spec.name))
            )
        return create_sdk_mcp_server(name="harness", version="0.1.0", tools=sdk_tools)

    async def run(self, task: TaskSpec, state: RunState) -> str:
        from claude_agent_sdk import ClaudeAgentOptions, query

        model = None if self.config.model in ("default", "") else self.config.model
        options = ClaudeAgentOptions(
            system_prompt=self.build_system_prompt(task),
            model=model,
            cwd=state.workspace,
            mcp_servers={"harness": self._build_mcp_server()},
            allowed_tools=[f"mcp__harness__{n}" for n in self.tools.names()]
            + ["Read", "Write", "Bash", "Glob", "Grep"],
            permission_mode="acceptEdits",
            max_turns=self.config.max_steps,
        )
        if state.session_ref:
            options.resume = state.session_ref

        final_text = ""
        async for message in query(prompt=self.build_task_prompt(task), options=options):
            self._normalize(message, state)
            kind = type(message).__name__
            if kind == "ResultMessage":
                final_text = getattr(message, "result", "") or final_text
                state.session_ref = getattr(message, "session_id", state.session_ref)
        self.tracer.emit("final", actor=self.name, payload={"text": clip(final_text, 4000)})
        return final_text

    async def resume(self, run_id: str) -> str:
        raise NotImplementedError("resume は state.session_ref 経由で run() に渡してください")

    def _normalize(self, message: Any, state: RunState) -> None:
        kind = type(message).__name__
        if kind == "AssistantMessage":
            for block in getattr(message, "content", []) or []:
                block_kind = type(block).__name__
                if block_kind == "TextBlock":
                    self.tracer.emit("reasoning_summary", actor=self.name,
                                     payload={"text": clip(getattr(block, "text", ""), 1000)})
                elif block_kind == "ToolUseBlock":
                    # MCP経由の共通ツールは call_tool 側で記録済み。SDK組込みツールのみ記録
                    tool_name = getattr(block, "name", "")
                    if not tool_name.startswith("mcp__harness__"):
                        self.tracer.emit("tool_call", actor=self.name,
                                         payload={"tool": tool_name,
                                                  "arguments": clip(getattr(block, "input", {}), 500)})
        elif kind == "ResultMessage":
            usage = getattr(message, "usage", None)
            cost = getattr(message, "total_cost_usd", None)
            state_payload = {"usage": usage if isinstance(usage, dict) else str(usage),
                             "total_cost_usd": cost}
            self.tracer.emit("reasoning_summary", actor=self.name, payload=state_payload)
