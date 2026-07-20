"""Evolver の変更可否ガード（specification.md 段階5）。

変更許可: Skill 本文・付属スクリプト・instructions・tool description・retry policy 等。
変更禁止: Benchmark 正解、Scientific Verifier、Security policy、評価指標、Evolver 自身。
このモジュール自体も変更禁止対象である。
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

ALLOWED_GLOBS = [
    "skills/*/SKILL.md",
    "skills/*/scripts/*",
    "skills/*/references/*",
    "config/instructions*.md",
]

FORBIDDEN_GLOBS = [
    "benchmarks/*",
    "benchmarks/**/*",
    "harness/verifier.py",
    "harness/policy.py",
    "evolver/*",
    "evolver/**/*",
    "config/default.yaml",      # promotion_gate / resource limit を含むため
    "config/production.yaml",
    "skills.lock",
    ".env",
    "pyproject.toml",
]


def classify(path: str | Path, repo_root: Path) -> str:
    """'allowed' / 'forbidden' / 'unlisted' を返す。unlisted も変更不可として扱う。"""
    rel = str(Path(path).resolve().relative_to(Path(repo_root).resolve())) \
        if Path(path).is_absolute() else str(path)
    for pattern in FORBIDDEN_GLOBS:
        if fnmatch.fnmatch(rel, pattern):
            return "forbidden"
    for pattern in ALLOWED_GLOBS:
        if fnmatch.fnmatch(rel, pattern):
            return "allowed"
    return "unlisted"


def assert_allowed(paths: list[str | Path], repo_root: Path) -> None:
    bad = [str(p) for p in paths if classify(p, repo_root) != "allowed"]
    if bad:
        raise PermissionError(f"Evolver はこれらのファイルを変更できません: {bad}")
