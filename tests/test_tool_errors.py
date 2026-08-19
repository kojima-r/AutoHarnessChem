"""ツール失敗の詳細（stderr 等）が後から確認できるように残ることを検証する。

trace（AgentEvent）には summary と error_type しか載らないため、原因調査には
workspace の tool_errors.jsonl を使う。
"""
import json
from pathlib import Path

from schemas import SandboxConfig, ToolResult
from tools.registry import ERROR_LOG_NAME, ToolRegistry, ToolSpec, build_default_registry


def _spec(name: str, func) -> ToolSpec:
    return ToolSpec(name=name, description=name, parameters={}, func=func)


def _entries(workspace: Path) -> list[dict]:
    path = workspace / ERROR_LOG_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_failure_is_logged_with_stderr_and_arguments(tmp_path):
    registry = ToolRegistry(error_log=tmp_path / ERROR_LOG_NAME)
    registry.register(_spec("boom", lambda **kw: ToolResult(
        status="failed", summary="専用環境でのスクリプト実行に失敗しました。 (returncode=1)",
        data={"stdout": "", "stderr": "torch.AcceleratorError: CUDA error: out of memory"},
        retryable=True, error_type="out_of_memory")))

    registry.call("boom", reactions=["r1"], memory_limit_mb=32768)
    entries = _entries(tmp_path)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["tool"] == "boom" and entry["status"] == "failed"
    assert entry["error_type"] == "out_of_memory" and entry["retryable"] is True
    # 再現に必要な引数と、trace には残らない stderr が入っている
    assert entry["arguments"]["memory_limit_mb"] == 32768
    assert "CUDA error: out of memory" in entry["data"]["stderr"]
    assert entry["at"]


def test_success_is_not_logged_and_failures_accumulate(tmp_path):
    registry = ToolRegistry(error_log=tmp_path / ERROR_LOG_NAME)
    registry.register(_spec("ok", lambda **kw: ToolResult(status="success", summary="ok")))
    registry.register(_spec("bad", lambda **kw: ToolResult(
        status="failed", summary="bad", data={"stderr": "kaboom"}, error_type="runtime_error")))

    registry.call("ok")
    assert _entries(tmp_path) == []
    registry.call("bad")
    registry.call("bad")
    assert len(_entries(tmp_path)) == 2          # 追記されて上書きされない


def test_partial_and_exceptions_are_logged(tmp_path):
    registry = ToolRegistry(error_log=tmp_path / ERROR_LOG_NAME)
    registry.register(_spec("partial", lambda **kw: ToolResult(
        status="partial", summary="途中まで", data={"pending": ["r2"]},
        retryable=True, error_type="timeout")))
    registry.register(_spec("raises", lambda **kw: 1 / 0))

    registry.call("partial")
    registry.call("raises")
    statuses = [(e["tool"], e["status"], e["error_type"]) for e in _entries(tmp_path)]

    assert ("partial", "partial", "timeout") in statuses
    assert ("raises", "failed", "tool_exception") in statuses
    # 例外は traceback が残る（ツール側が data を返せないケース）
    raised = next(e for e in _entries(tmp_path) if e["tool"] == "raises")
    assert "ZeroDivisionError" in raised["data"]["traceback"]


def test_unknown_tool_is_logged(tmp_path):
    registry = ToolRegistry(error_log=tmp_path / ERROR_LOG_NAME)
    registry.call("nope", x=1)
    assert _entries(tmp_path)[0]["error_type"] == "unknown_tool"


def test_default_registry_writes_into_the_workspace(tmp_path):
    from harness.policy import PolicyGate

    registry = build_default_registry(tmp_path, SandboxConfig(type="local"), PolicyGate())
    assert registry.error_log == tmp_path / ERROR_LOG_NAME
    # 存在しないファイルを渡して失敗させ、ログが workspace に作られることを確認
    result = registry.call("inspect_artifact", path="missing.csv")
    assert result.status == "failed"
    assert (tmp_path / ERROR_LOG_NAME).exists()


def test_adapter_puts_stderr_tail_and_log_name_into_the_trace(tmp_path):
    """trace からも「何が起きたか」と「全文の場所」が分かること。"""
    from adapters.base import BaseAdapter
    from harness.policy import PolicyGate
    from harness.traces import TraceWriter

    class Adapter(BaseAdapter):
        name = "dummy"

        async def _create_impl(self): ...

        async def run(self, task, state): ...

    registry = ToolRegistry(error_log=tmp_path / ERROR_LOG_NAME)
    registry.register(_spec("boom", lambda **kw: ToolResult(
        status="failed", summary="失敗", retryable=True, error_type="out_of_memory",
        data={"stderr": "line1\nOpenBLAS error: Memory allocation still failed"})))

    tracer = TraceWriter(tmp_path / "traces", "run-test")
    adapter = Adapter(tracer, PolicyGate())
    adapter.tools = registry
    adapter.call_tool("boom", {"smiles": ["O"]})

    events = [json.loads(line) for line in
              (tmp_path / "traces" / "run-test.jsonl").read_text(encoding="utf-8").splitlines()]
    payload = next(e["payload"] for e in events if e["event_type"] == "tool_result")
    assert "OpenBLAS error" in payload["stderr_tail"]
    assert payload["error_log"] == ERROR_LOG_NAME
    # 全文は workspace のログ側にある
    assert "line1" in _entries(tmp_path)[0]["data"]["stderr"]
