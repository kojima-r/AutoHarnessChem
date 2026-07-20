"""Deep Agents Adapter。

create_deep_agent に共通ツールを callable として渡す。write_todos / task(subagent) /
virtual filesystem は Deep Agents 側の組込み機能をそのまま使う。
LangGraph checkpointer (thread_id=run_id) により resume 可能。
"""
from __future__ import annotations

import functools
import json
from typing import Any, Callable

from adapters.base import AdapterUnavailable, BaseAdapter
from harness.traces import clip
from schemas import RunState, TaskSpec

_INSTALL_HINT = "pip install deepagents langgraph"


class DeepAgentsAdapter(BaseAdapter):
    name = "deepagents"

    async def _create_impl(self) -> None:
        try:
            import deepagents  # noqa: F401
        except ImportError as e:
            raise AdapterUnavailable(self.name, f"{e}. {_INSTALL_HINT}") from e
        self._agent = None
        self._checkpointer = None

    def _build_callables(self) -> list[Callable[..., str]]:
        callables = []
        for spec in self.tools.specs():
            def call(tool_name: str, **kwargs: Any) -> str:
                result = self.call_tool(tool_name, kwargs)
                return self.tool_result_json(result)

            fn = functools.partial(call, spec.name)
            fn = functools.wraps(call)(fn)
            fn.__name__ = spec.name  # type: ignore[attr-defined]
            fn.__doc__ = spec.description + "\nParameters (JSON Schema): " + json.dumps(
                spec.parameters, ensure_ascii=False
            )
            callables.append(fn)
        return callables

    def _ensure_agent(self, task: TaskSpec):
        from deepagents import create_deep_agent

        kwargs: dict[str, Any] = {
            "tools": self._build_callables(),
            "system_prompt": self.build_system_prompt(task),
        }
        if self.config.model not in ("default", ""):
            kwargs["model"] = self.config.model
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            self._checkpointer = InMemorySaver()
            kwargs["checkpointer"] = self._checkpointer
        except ImportError:
            pass
        try:
            self._agent = create_deep_agent(**kwargs)
        except TypeError:
            # 旧バージョンは instructions 引数
            kwargs["instructions"] = kwargs.pop("system_prompt")
            self._agent = create_deep_agent(**kwargs)
        return self._agent

    async def run(self, task: TaskSpec, state: RunState) -> str:
        agent = self._agent or self._ensure_agent(task)
        invoke_config = {
            "configurable": {"thread_id": state.run_id},
            "recursion_limit": max(self.config.max_steps * 2, 25),
        }
        state.session_ref = state.run_id
        payload = {"messages": [{"role": "user", "content": self.build_task_prompt(task)}]}

        result = await _maybe_async(agent.invoke, payload, config=invoke_config)

        messages = result.get("messages", []) if isinstance(result, dict) else []
        final_text = ""
        for message in messages:
            content = getattr(message, "content", None) or (
                message.get("content") if isinstance(message, dict) else ""
            )
            if content:
                final_text = _content_to_text(content)
        self.tracer.emit("final", actor=self.name, payload={"text": clip(final_text, 4000)})
        return final_text


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


async def _maybe_async(fn, *args, **kwargs):
    import asyncio
    result = await asyncio.to_thread(fn, *args, **kwargs)
    return result
