"""Policy Gate。

Agent が生成したコード・ツール呼び出し・書き込みパスを実行前に検査する。
ブロック判定は ToolResult(status="blocked") として呼び出し元へ返す。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from schemas import ToolResult

# コード実行前に拒否するパターン（ENABLE_UNSAFE_PY_RUNNER=1 で無効化可能）
BLOCKED_CODE_PATTERNS: list[tuple[str, str]] = [
    (r"\bimport\s+socket\b", "raw socket access"),
    (r"\beval\s*\(", "eval()"),
    (r"\bos\.system\s*\(\s*['\"]?\s*rm\b", "os.system rm"),
    (r"rm\s+-rf\s+/", "recursive delete of root"),
    (r"shutil\.rmtree\(\s*['\"]/(?!tmp)", "rmtree outside workspace"),
    (r"\bsubprocess\b.*\b(curl|wget|nc|ssh)\b", "network command via subprocess"),
]

RISK_LEVELS = ("low", "medium", "high")


class PolicyGate:
    def __init__(self, approval: str = "deny", allowed_write_roots: list[Path] | None = None):
        self.approval = approval
        self.allowed_write_roots = [p.resolve() for p in (allowed_write_roots or [])]

    @property
    def unsafe_enabled(self) -> bool:
        return os.environ.get("ENABLE_UNSAFE_PY_RUNNER", "0") == "1"

    def check_code(self, code: str) -> ToolResult | None:
        """問題なければ None、ブロック時は blocked の ToolResult を返す。"""
        if self.unsafe_enabled:
            return None
        for pattern, reason in BLOCKED_CODE_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return ToolResult(
                    status="blocked",
                    summary=f"Blocked by safety policy: {reason}",
                    retryable=False,
                    error_type="policy_violation",
                )
        return None

    def check_write_path(self, path: str | Path) -> ToolResult | None:
        if not self.allowed_write_roots:
            return None
        resolved = Path(path).resolve()
        for root in self.allowed_write_roots:
            if resolved == root or root in resolved.parents:
                return None
        return ToolResult(
            status="blocked",
            summary=f"Write outside allowed paths: {resolved}",
            retryable=False,
            error_type="policy_violation",
        )

    def approve(self, action: str, risk_level: str = "high") -> bool:
        """high リスクアクションの承認。approval 設定に従う。"""
        if risk_level in ("low", "medium"):
            return True
        if self.approval == "allow":
            return True
        if self.approval == "interactive":
            answer = input(f"[approval required] {action} — allow? [y/N] ").strip().lower()
            return answer in ("y", "yes")
        return False
