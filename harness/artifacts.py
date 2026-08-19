"""Artifact Manager。

run ごとの workspace ディレクトリを管理し、生成物を manifest.json に記録する。
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path

from schemas import Artifact

_KIND_BY_EXT = {
    ".png": "figure", ".jpg": "figure", ".jpeg": "figure", ".svg": "figure",
    ".pdf": "figure", ".webp": "figure",
    ".csv": "data", ".json": "data", ".parquet": "data", ".xyz": "data",
    ".md": "report", ".txt": "report", ".html": "report",
    ".pkl": "model", ".joblib": "model",
    ".log": "log", ".jsonl": "log",
}

_IGNORED_NAMES = {"manifest.json", "ledger.json", "report.json", "report.md"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ArtifactManager:
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def register(self, path: str | Path) -> Artifact:
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace / p
        stat = p.stat()
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        return Artifact(
            path=str(p),
            mime=mime,
            bytes=stat.st_size,
            sha256=_sha256(p),
            kind=_KIND_BY_EXT.get(p.suffix.lower(), "other"),
        )

    def scan(self) -> list[Artifact]:
        """workspace 配下の全ファイルを列挙し、manifest.json を更新する。"""
        artifacts: list[Artifact] = []
        for p in sorted(self.workspace.rglob("*")):
            if p.is_file() and p.name not in _IGNORED_NAMES and not p.name.startswith("."):
                artifacts.append(self.register(p))
        manifest = self.workspace / "manifest.json"
        manifest.write_text(
            json.dumps([a.model_dump() for a in artifacts], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return artifacts
