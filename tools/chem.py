"""共通ツール実装（追加の依存を持たないもの）。

すべて ToolResult を返す。ファイル出力はすべて workspace 配下に限定する。

重い依存が必要なツールはそれぞれ専用環境で実行するモジュールに分かれている:
  - tools/chemenv.py   … RDKit / pandas / scikit-learn 系（正準化・記述子・CV 等）
  - tools/opttddft.py  … 量子化学（HOMO/LUMO・TDDFT・PES スキャン・MI 探索）
  - tools/reactiont5.py … ReactionT5v2 による反応予測
  - tools/aizynth.py   … AiZynthFinder による逆合成経路探索
  - tools/report.py    … 構造式付き HTML レポート
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from schemas import Artifact, ToolResult


def _missing(module: str) -> ToolResult:
    return ToolResult(
        status="failed",
        summary=f"required package `{module}` is not installed in the harness environment",
        retryable=False,
        error_type="missing_dependency",
    )


def _artifact(path: Path) -> Artifact:
    import mimetypes
    return Artifact(
        path=str(path),
        mime=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        bytes=path.stat().st_size,
        kind="figure" if path.suffix in (".png", ".svg", ".pdf") else "data",
    )


def _resolve_input(workspace: Path, path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        for candidate in (workspace / p, Path.cwd() / p):
            if candidate.exists():
                return candidate
    return p


# ---------------------------------------------------------------------------
# 1. inspect_artifact
# ---------------------------------------------------------------------------

def inspect_artifact(workspace: Path, path: str, max_bytes: int = 4000) -> ToolResult:
    f = _resolve_input(workspace, path)
    if not f.exists():
        return ToolResult(status="failed", summary=f"artifact not found: {path}",
                          error_type="input_not_found")
    stat = f.stat()
    data: dict = {"path": str(f), "bytes": stat.st_size, "suffix": f.suffix}
    if f.suffix.lower() in (".csv", ".json", ".txt", ".md", ".xyz", ".log"):
        text = f.read_text(encoding="utf-8", errors="replace")
        data["preview"] = text[:max_bytes]
        data["truncated"] = len(text) > max_bytes
    return ToolResult(status="success", summary=f"{f.name}: {stat.st_size} bytes", data=data)


# ---------------------------------------------------------------------------
# 2. run_python_sandbox  (registry.py で policy + sandbox を束縛して構築)
# ---------------------------------------------------------------------------

def run_python_sandbox(workspace: Path, code: str, *, sandbox, policy) -> ToolResult:
    blocked = policy.check_code(code)
    if blocked is not None:
        return blocked
    result = sandbox.run(code)
    artifacts = [_artifact(Path(p)) for p in result.new_files if Path(p).exists()]
    if result.timed_out:
        return ToolResult(
            status="failed",
            summary=f"execution timed out after {sandbox.config.timeout_sec}s。計算を軽くしてください（基底縮小・分子数削減など）。",
            data={"stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]},
            artifacts=artifacts, retryable=True, error_type="timeout",
        )
    if result.returncode != 0:
        error_type = "runtime_error"
        if "ModuleNotFoundError" in result.stderr:
            error_type = "missing_dependency"
        return ToolResult(
            status="failed",
            summary=f"script exited with code {result.returncode}",
            data={"stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]},
            artifacts=artifacts, retryable=error_type != "missing_dependency",
            error_type=error_type,
        )
    return ToolResult(
        status="success",
        summary="script executed successfully",
        data={"stdout": result.stdout[-8000:], "stderr": result.stderr[-2000:],
              "new_files": result.new_files},
        artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# 3. verify_scientific_result
# ---------------------------------------------------------------------------

def verify_scientific_result(workspace: Path, task_type: str = "generic",
                             expected_outputs: list[str] | None = None) -> ToolResult:
    from harness.verifier import ScientificVerifier
    from schemas import TaskSpec

    task = TaskSpec(description="inline verification", task_type=task_type,  # type: ignore[arg-type]
                    expected_outputs=expected_outputs or [])
    verification = ScientificVerifier().verify(task, workspace)
    return ToolResult(
        status="success" if verification.passed else "partial",
        summary="verification passed" if verification.passed
        else f"verification failed: {len(verification.required_repairs)} repairs required",
        data=verification.model_dump(),
    )


# ---------------------------------------------------------------------------
# 4. search_official_documentation
# ---------------------------------------------------------------------------

def search_official_documentation(workspace: Path, query: str, max_results: int = 5) -> ToolResult:
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return ToolResult(
            status="blocked",
            summary="TAVILY_API_KEY is not set — web search is unavailable in this environment",
            retryable=False, error_type="missing_credentials",
        )
    try:
        import httpx
    except ImportError:
        return _missing("httpx")
    response = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": max_results,
              "include_domains": [], "search_depth": "basic"},
        timeout=30,
    )
    response.raise_for_status()
    results = [
        {"title": r.get("title"), "url": r.get("url"), "content": (r.get("content") or "")[:500]}
        for r in response.json().get("results", [])
    ]
    return ToolResult(status="success", summary=f"{len(results)} results for: {query}",
                      data={"results": results})
