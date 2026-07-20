"""Benchmark runner。

同一タスク集合を複数 SDK で実行し、成功率・実行時間を比較する。
結果は benchmarks/results/<label>.json に保存し、task_type ごとの
最良プロバイダから routing 更新案を出力する。
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import yaml

from adapters import AdapterUnavailable
from harness.config import HarnessConfig
from harness.controller import HarnessController

TASKS_PATH = Path(__file__).parent / "tasks.yaml"
RESULTS_DIR = Path(__file__).parent / "results"


def load_tasks(tags: list[str] | None = None, ids: list[str] | None = None) -> list[dict]:
    tasks = yaml.safe_load(TASKS_PATH.read_text(encoding="utf-8"))["tasks"]
    if ids:
        tasks = [t for t in tasks if t["id"] in ids]
    if tags:
        tasks = [t for t in tasks if set(tags) & set(t.get("tags", []))]
    return tasks


async def run_benchmark(
    config: HarnessConfig,
    providers: list[str],
    tags: list[str] | None = None,
    ids: list[str] | None = None,
    label: str = "benchmark",
) -> dict[str, Any]:
    tasks = load_tasks(tags, ids)
    root = config.paths.root
    records: list[dict[str, Any]] = []

    for provider in providers:
        controller = HarnessController(config)
        for task_def in tasks:
            copy_inputs = [str(root / f) for f in task_def.get("input_files", [])]
            started = time.monotonic()
            record: dict[str, Any] = {
                "benchmark_id": task_def["id"],
                "provider": provider,
                "task_type": task_def.get("task_type", "generic"),
            }
            try:
                report = await controller.run(
                    task_def["request"],
                    provider=provider,
                    task_type=task_def.get("task_type"),
                    expected_outputs=task_def.get("expected_outputs"),
                    copy_inputs=copy_inputs,
                )
                record.update(
                    run_id=report.run_id,
                    passed=report.passed,
                    attempts=report.attempts,
                    warnings=len(report.verification.scientific_warnings),
                    usage=report.usage,
                )
            except AdapterUnavailable as e:
                record.update(passed=False, error=str(e), skipped=True)
            except Exception as e:
                record.update(passed=False, error=f"{type(e).__name__}: {e}")
            record["elapsed_sec"] = round(time.monotonic() - started, 2)
            records.append(record)
            print(f"[bench] {task_def['id']} on {provider}: "
                  f"passed={record.get('passed')} ({record['elapsed_sec']}s)")

    summary = summarize(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{label}.json"
    out_path.write_text(
        json.dumps({"records": records, "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[bench] results written to {out_path}")
    return {"records": records, "summary": summary}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_provider: dict[str, dict[str, Any]] = {}
    for record in records:
        stats = by_provider.setdefault(record["provider"], {"n": 0, "passed": 0, "elapsed": 0.0})
        if record.get("skipped"):
            continue
        stats["n"] += 1
        stats["passed"] += int(bool(record.get("passed")))
        stats["elapsed"] += record.get("elapsed_sec", 0.0)
    for stats in by_provider.values():
        stats["success_rate"] = round(stats["passed"] / stats["n"], 3) if stats["n"] else None
        stats["mean_elapsed_sec"] = round(stats["elapsed"] / stats["n"], 2) if stats["n"] else None

    # task_type ごとの最良プロバイダ → routing 更新案（実測ベース）
    by_type: dict[str, dict[str, list[bool]]] = {}
    for record in records:
        if record.get("skipped"):
            continue
        by_type.setdefault(record["task_type"], {}).setdefault(record["provider"], []).append(
            bool(record.get("passed"))
        )
    routing_suggestion = {}
    for task_type, provider_results in by_type.items():
        best = max(provider_results.items(),
                   key=lambda kv: (sum(kv[1]) / len(kv[1]), -len(kv[1])))
        routing_suggestion[task_type] = best[0]

    return {"by_provider": by_provider, "routing_suggestion": routing_suggestion}


def run_benchmark_sync(config: HarnessConfig, providers: list[str], **kwargs) -> dict[str, Any]:
    return asyncio.run(run_benchmark(config, providers, **kwargs))
