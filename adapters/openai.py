"""OpenAI Agents SDK Adapter。

Agent / Runner / FunctionTool / SQLiteSession を使用。共通ツールは
FunctionTool(on_invoke_tool) として登録し、結果は ToolResult の JSON を返す。
"""
from __future__ import annotations

import json
from pathlib import Path

from adapters.base import AdapterUnavailable, BaseAdapter
from harness.traces import clip
from schemas import RunState, TaskSpec

_INSTALL_HINT = "pip install openai-agents"


class OpenAIAgentsAdapter(BaseAdapter):
    name = "openai"

    async def _create_impl(self) -> None:
        try:
            import agents  # noqa: F401
        except ImportError as e:
            raise AdapterUnavailable(self.name, f"{e}. {_INSTALL_HINT}") from e

    def _build_tools(self):
        from agents import FunctionTool

        function_tools = []
        for spec in self.tools.specs():
            def make_invoke(tool_name: str):
                async def on_invoke_tool(ctx, args_json: str) -> str:
                    arguments = json.loads(args_json) if args_json else {}
                    result = self.call_tool(tool_name, arguments)
                    return self.tool_result_json(result)
                return on_invoke_tool

            params = dict(spec.parameters)
            params.setdefault("additionalProperties", False)
            function_tools.append(FunctionTool(
                name=spec.name,
                description=spec.description,
                params_json_schema=params,
                on_invoke_tool=make_invoke(spec.name),
                strict_json_schema=False,
            ))
        return function_tools

    async def run(self, task: TaskSpec, state: RunState) -> str:
        from agents import Agent, Runner, SQLiteSession

        agent = Agent(
            name="AutoHarnessChem",
            instructions=self.build_system_prompt(task),
            tools=self._build_tools(),
            model=None if self.config.model in ("default", "") else self.config.model,
        )
        session_db = Path(state.workspace) / "openai_session.db"
        session = SQLiteSession(state.run_id, str(session_db))
        state.session_ref = str(session_db)

        result = await Runner.run(
            agent,
            input=self.build_task_prompt(task),
            session=session,
            max_turns=self.config.max_steps,
        )

        for item in getattr(result, "new_items", []) or []:
            kind = type(item).__name__
            if kind == "MessageOutputItem":
                self.tracer.emit("reasoning_summary", actor=self.name,
                                 payload={"text": clip(getattr(item, "raw_item", ""), 1000)})
        final_text = str(getattr(result, "final_output", "") or "")
        self.tracer.emit("final", actor=self.name, payload={"text": clip(final_text, 4000)})
        return final_text
