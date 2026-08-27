"""採点済み record の集計とレポート出力。

集計の軸は論文と同じ「レベル → 能力次元 → タスク」。加えて ahc 固有の観測値
（Verifier 合格率・試行回数・所要時間・答えの取得元）も出す。ahc の評価では
「答えの正しさ」と「harness がタスクを完了できたか」は別物なので、両方載せる。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from benchmarks_chemeval.catalog import LEVELS, load_catalog
from benchmarks_chemeval.metrics import METRIC_INFO


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate_metrics(records: list[dict]) -> dict[str, float]:
    """record の metrics を集めて平均（squared_error は RMSE、abs_error は MAE）。"""
    buckets: dict[str, list[float]] = {}
    for record in records:
        for name, value in (record.get("metrics") or {}).items():
            if value is None:
                continue
            buckets.setdefault(name, []).append(float(value))
    out: dict[str, float] = {}
    for name, values in buckets.items():
        if name == "squared_error":
            out["rmse"] = round(math.sqrt(sum(values) / len(values)), 4)
        elif name == "abs_error":
            out["mae"] = round(sum(values) / len(values), 4)
        else:
            out[name] = round(sum(values) / len(values), 4)
    return out


def summarize_group(records: list[dict]) -> dict:
    scores = [r["score"] for r in records if r.get("score") is not None]
    elapsed = [r["elapsed_sec"] for r in records if r.get("elapsed_sec") is not None]
    attempts = [r["attempts"] for r in records if r.get("attempts")]
    metrics = aggregate_metrics(records)
    metric_names = sorted({r.get("metric", "") for r in records if r.get("metric")})
    primary = None
    if len(metric_names) == 1:
        info = METRIC_INFO.get(metric_names[0], {})
        primary = info.get("primary")
    return {
        "n": len(records),
        "metric": metric_names[0] if len(metric_names) == 1 else "mixed",
        "primary_metric": primary,
        "primary_value": metrics.get(primary) if primary else None,
        "score": _mean(scores),
        "scored": len(scores),
        "answered_rate": _mean([float(bool(r.get("answered"))) for r in records]),
        "valid_rate": _mean([float(bool(r.get("valid"))) for r in records]),
        "harness_passed_rate": _mean([float(bool(r.get("harness_passed"))) for r in records]),
        "error_count": sum(1 for r in records if r.get("error")),
        "mean_attempts": _mean([float(a) for a in attempts]),
        "mean_elapsed_sec": _mean([float(e) for e in elapsed]),
        "metrics": metrics,
        "answer_sources": _count(records, "answer_source"),
    }


def _count(records: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get(field))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _group(records: list[dict], field: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record.get(field)), []).append(record)
    return grouped


def summarize(records: list[dict], label: str = "chemeval") -> dict:
    """provider ごと・レベルごと・次元ごと・タスクごとの集計をまとめる。"""
    catalog = load_catalog()
    order = {task.id: i for i, task in enumerate(catalog)}
    summary: dict = {"label": label, "n_records": len(records), "providers": {}}
    for provider, provider_records in sorted(_group(records, "provider").items()):
        levels = {}
        for level in LEVELS:
            in_level = [r for r in provider_records if r.get("level") == level]
            if not in_level:
                continue
            dimensions = {
                dim: summarize_group(rows)
                for dim, rows in sorted(_group(in_level, "dimension").items())
            }
            tasks = {}
            for task_id, rows in sorted(_group(in_level, "task_id").items(),
                                        key=lambda kv: order.get(kv[0], 999)):
                task_summary = summarize_group(rows)
                task_summary["dimension"] = rows[0].get("dimension", "")
                task_summary["task_name"] = rows[0].get("task_name", task_id)
                tasks[task_id] = task_summary
            entry = summarize_group(in_level)
            entry.update(dimensions=dimensions, tasks=tasks)
            levels[level] = entry
        overall = summarize_group(provider_records)
        # レベル横断の代表値: 0..1 の score を持つタスクのマクロ平均（尺度の違う回帰は除外）
        task_scores = [t["score"] for level in levels.values() for t in level["tasks"].values()
                       if t["score"] is not None]
        overall["macro_task_score"] = _mean(task_scores)
        overall["scored_tasks"] = len(task_scores)
        summary["providers"][provider] = {"overall": overall, "levels": levels}
    return summary


# --- Markdown -----------------------------------------------------------

def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _metric_cell(group: dict) -> str:
    metrics = group.get("metrics") or {}
    keep = [k for k in ("rmse", "mae", "tanimoto", "bleu4", "edit_similarity",
                        "atom_cosine", "role_f1", "validity", "strict_match",
                        "exact_set_match")
            if k in metrics]
    return ", ".join(f"{k}={_fmt(metrics[k])}" for k in keep) or "-"


def render_markdown(summary: dict) -> str:
    lines = [f"# ChemEval 評価レポート: {summary['label']}", "",
             f"- 対象レコード数: {summary['n_records']}", ""]
    for provider, block in summary["providers"].items():
        overall = block["overall"]
        lines += [
            f"## provider: {provider}", "",
            f"- タスク平均スコア (macro, 0..1): **{_fmt(overall['macro_task_score'])}** "
            f"({overall['scored_tasks']} タスク)",
            f"- 問題平均スコア (micro): {_fmt(overall['score'])} "
            f"({overall['scored']}/{overall['n']} 問が採点対象)",
            f"- 回答率: {_fmt(overall['answered_rate'])} / "
            f"形式が妥当だった率: {_fmt(overall['valid_rate'])}",
            f"- Verifier 合格率: {_fmt(overall['harness_passed_rate'])} / "
            f"平均試行回数: {_fmt(overall['mean_attempts'], 2)} / "
            f"平均所要: {_fmt(overall['mean_elapsed_sec'], 1)}s / "
            f"実行エラー: {overall['error_count']} 件",
            f"- 答えの取得元: {overall['answer_sources']}",
            "",
            "### レベル別", "",
            "| level | n | score | 回答率 | Verifier 合格率 | 主要指標 |",
            "| --- | --: | --: | --: | --: | --- |",
        ]
        for level, level_block in block["levels"].items():
            lines.append(
                f"| {level} | {level_block['n']} | {_fmt(level_block['score'])} "
                f"| {_fmt(level_block['answered_rate'])} "
                f"| {_fmt(level_block['harness_passed_rate'])} "
                f"| {_metric_cell(level_block)} |")
        lines += ["", "### タスク別", "",
                  "| level | dimension | task | metric | n | 主指標 | score | 回答率 | 追加指標 |",
                  "| --- | --- | --- | --- | --: | --: | --: | --: | --- |"]
        for level, level_block in block["levels"].items():
            for task_id, task in level_block["tasks"].items():
                lines.append(
                    f"| {level} | {task.get('dimension') or '-'} | {task_id} | {task['metric']} "
                    f"| {task['n']} | {_fmt(task['primary_value'])} | {_fmt(task['score'])} "
                    f"| {_fmt(task['answered_rate'])} | {_metric_cell(task)} |")
        lines.append("")
    lines += ["## 注記", "",
              "- score は 0..1 で高いほど良い代表値。回帰タスク（RMSE/MAE）は尺度が違うため",
              "  score には含めず「主指標」列に出す。",
              "- judge 系タスクは `--judge` を指定したときだけ採点される（未指定なら score=None）。",
              "- Verifier 合格率は「ahc が答えファイルを作れたか」であり、答えの正しさとは別。",
              ""]
    return "\n".join(lines)


def write_report(records: list[dict], out_dir: Path, label: str = "chemeval") -> dict:
    """metrics.json / report.md / scored.jsonl を書く。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records, label)
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    with (out_dir / "scored.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            slim = {k: v for k, v in record.items() if k != "query"}
            fh.write(json.dumps(slim, ensure_ascii=False) + "\n")
    return summary
