"""Task Ledger。

TaskSpec と各試行の状態遷移を workspace/ledger.json に永続化する。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemas import TaskSpec, VerificationResult, utcnow


class TaskLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, Any] = {"task": None, "provider": None, "attempts": [], "status": "pending"}

    def open(self, task: TaskSpec, provider: str) -> None:
        self._data["task"] = task.model_dump()
        self._data["provider"] = provider
        self._data["status"] = "running"
        self._data["opened_at"] = utcnow().isoformat()
        self._save()

    def record_attempt(self, attempt: int, verification: VerificationResult, note: str = "") -> None:
        self._data["attempts"].append({
            "attempt": attempt,
            "verification": verification.model_dump(),
            "note": note,
            "at": utcnow().isoformat(),
        })
        self._save()

    def close(self, status: str) -> None:
        self._data["status"] = status
        self._data["closed_at"] = utcnow().isoformat()
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "TaskLedger":
        ledger = cls(path)
        if ledger.path.exists():
            ledger._data = json.loads(ledger.path.read_text(encoding="utf-8"))
        return ledger
