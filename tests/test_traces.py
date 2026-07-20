from app.cli import format_event
from harness.traces import TraceWriter, read_trace


def test_listener_receives_events_and_file_is_written(tmp_path):
    writer = TraceWriter(tmp_path, "run-x")
    seen = []
    writer.subscribe(seen.append)
    writer.emit("tool_call", "claude", {"tool": "calculate_orbitals", "arguments": {}})
    writer.emit("final", "claude", {"text": "done"})
    assert [e.event_type for e in seen] == ["tool_call", "final"]
    assert len(read_trace(tmp_path / "run-x.jsonl")) == 2


def test_listener_exception_does_not_break_emit(tmp_path):
    writer = TraceWriter(tmp_path, "run-y")
    writer.subscribe(lambda e: 1 / 0)
    event = writer.emit("error", "controller", {"error": "boom"})
    assert event.event_type == "error"
    assert len(read_trace(tmp_path / "run-y.jsonl")) == 1


def test_format_event_variants(tmp_path):
    writer = TraceWriter(tmp_path, "run-z")
    lines = []
    writer.subscribe(lambda e: lines.append(format_event(e)))
    writer.emit("tool_call", "openai", {"tool": "inspect_dataset", "arguments": {"path": "a.csv"}})
    writer.emit("tool_result", "openai", {"tool": "inspect_dataset", "status": "success",
                                          "summary": "3 rows"})
    writer.emit("reasoning_summary", "controller",
                {"phase": "attempt", "attempt": 2, "max_attempts": 4, "repairs": ["fix"]})
    writer.emit("reasoning_summary", "verifier",
                {"passed": False, "requirements_missing": ["x"], "scientific_warnings": []})
    assert "→ inspect_dataset" in lines[0]
    assert "✔ inspect_dataset [success] 3 rows" in lines[1]
    assert "試行 2/4" in lines[2] and "修復指示 1 件" in lines[2]
    assert "検証不合格" in lines[3]
