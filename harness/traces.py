"""Trace Normalizer / Writer。

SDK固有のイベントを AgentEvent に正規化し、traces/<run_id>.jsonl へ追記する。
Evolver はこの jsonl を横断的に解析する。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from schemas import AgentEvent


class TraceWriter:
    def __init__(self, traces_dir: Path, run_id: str):
        self.run_id = run_id
        self.path = Path(traces_dir) / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._listeners: list[Callable[[AgentEvent], None]] = []

    def subscribe(self, listener: Callable[[AgentEvent], None]) -> None:
        """イベント発生ごとに呼ばれるリスナーを登録する（CLIの逐次表示等）。"""
        self._listeners.append(listener)

    def emit(self, event_type: str, actor: str, payload: dict[str, Any] | None = None) -> AgentEvent:
        event = AgentEvent(
            run_id=self.run_id,
            event_type=event_type,  # type: ignore[arg-type]
            actor=actor,
            payload=payload or {},
        )
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(event.model_dump_json() + "\n")
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass  # 表示系の失敗で run を止めない
        return event


def read_trace(path: str | Path) -> list[AgentEvent]:
    events = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(AgentEvent.model_validate(json.loads(line)))
    return events


def clip(value: Any, limit: int = 2000) -> str:
    s = "" if value is None else str(value)
    return s if len(s) <= limit else s[:limit] + "... [truncated]"
