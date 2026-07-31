"""FastAPI サーバ + Web UI。`ahc api` で起動し、http://127.0.0.1:8000/ を開く。

- POST /api/tasks           タスクをバックグラウンド実行し run_id を即返す
- GET  /api/runs            ラン一覧（実行中 + workspaces/ の履歴）
- GET  /api/runs/{id}       ラン詳細（report / ledger / manifest / report.md）
- GET  /api/runs/{id}/trace 正規化イベントの増分取得（?after=N）
- GET  /api/runs/{id}/artifacts/{path}  成果物ファイル配信（画像・HTML報告等）
- POST /api/uploads         入力ファイルのアップロード（python-multipart 必要）
- GET  /api/skills, /api/providers
- GET  /static/{name}       同梱フロントエンドライブラリ（SmilesDrawer 等）

NOTE: `from __future__ import annotations` を付けないこと。エンドポイント内で
定義した Pydantic モデルの注釈が文字列化され、FastAPI が body として解決できなくなる。
"""
import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

from harness.config import HarnessConfig

WEB_DIR = Path(__file__).parent / "web"
VENDOR_DIR = WEB_DIR / "vendor"


def create_app(config: HarnessConfig):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from pydantic import BaseModel

    from harness.controller import HarnessController
    from harness.skill_registry import SkillRegistry
    from schemas import new_id

    app = FastAPI(title="AutoHarnessChem", version="0.2.0")
    controller = HarnessController(config)
    # run_id -> {"status": running|succeeded|failed|error, "error": str|None}
    active: dict[str, dict[str, Any]] = {}

    class TaskRequest(BaseModel):
        request: str
        provider: str | None = None
        task_type: str | None = None
        expected_outputs: list[str] | None = None
        input_paths: list[str] = []

    # ------------------------------------------------------------------ tasks

    async def _execute(run_id: str, body: TaskRequest) -> None:
        try:
            report = await controller.run(
                body.request,
                provider=body.provider or None,
                task_type=body.task_type or None,
                expected_outputs=body.expected_outputs or None,
                copy_inputs=body.input_paths,
                run_id=run_id,
            )
            active[run_id] = {"status": "succeeded" if report.passed else "failed"}
        except Exception as e:
            active[run_id] = {"status": "error", "error": f"{type(e).__name__}: {e}"}

    @app.post("/api/tasks")
    async def submit_task(body: TaskRequest):
        if not body.request.strip():
            raise HTTPException(400, "request must not be empty")
        missing = [p for p in body.input_paths if not Path(p).is_file()]
        if missing:
            raise HTTPException(400, f"input files not found: {missing}")
        run_id = new_id("run")
        active[run_id] = {"status": "running"}
        asyncio.create_task(_execute(run_id, body))
        return {"run_id": run_id, "status": "running"}

    # ------------------------------------------------------------------- runs

    def _run_summary(run_id: str) -> dict[str, Any]:
        workspace = config.paths.workspaces / run_id
        entry: dict[str, Any] = {"run_id": run_id}
        entry.update(active.get(run_id, {}))
        report_path = workspace / "report.json"
        ledger_path = workspace / "ledger.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            entry.setdefault("status", "succeeded" if report.get("passed") else "failed")
            entry.update(
                provider=report.get("provider"),
                passed=report.get("passed"),
                attempts=report.get("attempts"),
                task_type=report.get("task", {}).get("task_type"),
                description=report.get("task", {}).get("description", "")[:120],
                finished_at=report.get("finished_at"),
            )
        elif ledger_path.exists():
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            entry.setdefault("status", ledger.get("status", "running"))
            task = ledger.get("task") or {}
            entry.update(
                provider=ledger.get("provider"),
                task_type=task.get("task_type"),
                description=(task.get("description") or "")[:120],
            )
        else:
            entry.setdefault("status", "running")
        return entry

    @app.get("/api/runs")
    def list_runs():
        on_disk = {
            p.name for p in config.paths.workspaces.glob("run-*") if p.is_dir()
        }
        all_ids = on_disk | set(active)
        runs = [_run_summary(run_id) for run_id in all_ids]

        def sort_key(entry):
            workspace = config.paths.workspaces / entry["run_id"]
            return workspace.stat().st_mtime if workspace.exists() else float("inf")

        runs.sort(key=sort_key, reverse=True)
        return {"runs": runs}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        workspace = config.paths.workspaces / run_id
        if run_id not in active and not workspace.exists():
            raise HTTPException(404, f"run {run_id} not found")
        detail = _run_summary(run_id)
        for name, key in (("report.json", "report"), ("ledger.json", "ledger"),
                          ("manifest.json", "artifacts")):
            path = workspace / name
            if path.exists():
                detail[key] = json.loads(path.read_text(encoding="utf-8"))
        report_md = workspace / "report.md"
        if report_md.exists():
            detail["report_md"] = report_md.read_text(encoding="utf-8")
        return detail

    @app.get("/api/runs/{run_id}/trace")
    def get_trace(run_id: str, after: int = 0):
        path = Path(config.paths.traces) / f"{run_id}.jsonl"
        if not path.exists():
            return {"events": [], "next": after}
        lines = path.read_text(encoding="utf-8").splitlines()
        events = [json.loads(line) for line in lines[after:] if line.strip()]
        return {"events": events, "next": len(lines)}

    @app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
    def get_artifact(run_id: str, artifact_path: str):
        workspace = (config.paths.workspaces / run_id).resolve()
        target = (workspace / artifact_path).resolve()
        if workspace != target and workspace not in target.parents:
            raise HTTPException(403, "path escapes workspace")
        if not target.is_file():
            raise HTTPException(404, f"artifact not found: {artifact_path}")
        return FileResponse(target)

    # ---------------------------------------------------------------- uploads

    if importlib.util.find_spec("multipart") or importlib.util.find_spec("python_multipart"):
        from fastapi import File, UploadFile

        @app.post("/api/uploads")
        async def upload_file(file: UploadFile = File(...)):
            uploads_dir = config.paths.workspaces / "_uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            safe_name = Path(file.filename or "upload.dat").name
            dest = uploads_dir / safe_name
            dest.write_bytes(await file.read())
            return {"path": str(dest), "name": safe_name, "bytes": dest.stat().st_size}

    # ------------------------------------------------------------------- meta

    @app.get("/api/skills")
    def list_skills():
        registry = SkillRegistry(config.paths.skills)
        return {"skills": [registry.get(n).model_dump(exclude={"body"})
                           for n in registry.names()]}

    @app.get("/api/providers")
    def list_providers():
        available = {
            "deepagents": importlib.util.find_spec("deepagents") is not None,
            "claude": importlib.util.find_spec("claude_agent_sdk") is not None,
            "openai": importlib.util.find_spec("agents") is not None,
        }
        return {"available": available, "routing": config.routing,
                "default": config.runtime.provider}

    # ----------------------------------------------------------------- web UI

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return (WEB_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/static/{name}", include_in_schema=False)
    def static_asset(name: str):
        """同梱ライブラリ（SmilesDrawer 等）の配信。CDN を使わないためのもの。"""
        target = (VENDOR_DIR / name).resolve()
        if target.parent != VENDOR_DIR.resolve() or not target.is_file():
            raise HTTPException(404, f"asset not found: {name}")
        media = "text/javascript" if target.suffix == ".js" else None
        return FileResponse(target, media_type=media,
                            headers={"Cache-Control": "public, max-age=86400"})

    return app
