"""採点済み record の集計とレポート出力。

集計の軸は論文と同じ「レベル → 能力次元 → タスク」。加えて ahc 固有の観測値
（Verifier 合格率・試行回数・所要時間・答えの取得元）も出す。ahc の評価では
「答えの正しさ」と「harness がタスクを完了できたか」は別物なので、両方載せる。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from benchmarks_chemeval import baselines
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
        # bare モードは Verifier を通していない（harness_passed=None）。None を 0 と
        # 数えると「Verifier に落ちた」と読めてしまうので、母集団から除いて None を返す
        "harness_passed_rate": _mean([float(bool(r.get("harness_passed"))) for r in records
                                      if r.get("harness_passed") is not None]),
        "error_count": sum(1 for r in records if r.get("error")),
        "mean_attempts": _mean([float(a) for a in attempts]),
        "mean_elapsed_sec": _mean([float(e) for e in elapsed]),
        "metrics": metrics,
        "answer_sources": _count(records, "answer_source"),
        # どのモデルの成績かを明示する（文献値と並べる以上、必須の情報）
        "models": _count([r for r in records if r.get("model")], "model"),
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


def task_summaries(records: list[dict]) -> dict[str, dict]:
    """task_id → 集計（provider をまたいでまとめる。比較列の材料）。"""
    return {task_id: summarize_group(rows)
            for task_id, rows in _group(records, "task_id").items()}


def summarize(records: list[dict], label: str = "chemeval",
              compare_records: list[dict] | None = None,
              compare_label: str | None = None) -> dict:
    """provider ごと・レベルごと・次元ごと・タスクごとの集計をまとめる。

    `compare_records` を渡すと、別 run（素の LLM ベースラインなど）を
    比較列として文献値の表に並べる。
    """
    catalog = load_catalog()
    order = {task.id: i for i, task in enumerate(catalog)}
    summary: dict = {"label": label, "n_records": len(records), "providers": {}}
    other = task_summaries(compare_records) if compare_records else None
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
        all_tasks = {task_id: task for level in levels.values()
                     for task_id, task in level["tasks"].items()}
        summary["providers"][provider] = {
            "overall": overall, "levels": levels,
            # 論文 Table 1 の文献値との突き合わせ（採点には影響しない）
            "baselines": baselines.compare(all_tasks, other_tasks=other,
                                           other_label=compare_label),
        }
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


def _render_baselines(compare: dict) -> list[str]:
    """文献値（論文 Table 1）との比較セクション。"""
    if not compare:
        return []
    source, macro = compare.get("source", {}), compare.get("macro") or {}
    rows = compare.get("tasks") or {}
    lines = ["", "### 文献値との比較（他手法）", "",
             f"- 出典: {source.get('paper', '?')}（{source.get('venue', '?')}） "
             f"{source.get('table', '')}",
             f"- 論文: {source.get('url', '')} / 転記元の図: `{source.get('local_figure', '')}`",
             "- **同条件の比較ではない**: 論文は素の LLM を 0-shot で評価している。ahc は",
             "  「ツール（量子化学計算・反応予測・逆合成）+ Verifier + 再計画ループ」なので、",
             "  差はモデル性能だけでなく harness の寄与も含む。",
             ""]
    column = compare.get("compare") or {}
    if macro.get("n_tasks"):
        lines += [f"指標が一致する {macro['n_tasks']} タスクだけを母集団にしたマクロ平均"
                  "（全手法で同じタスク集合）:", "",
                  "| 手法 | マクロ平均 |", "| --- | --: |",
                  f"| **ahc（この run）** | **{_fmt(macro['ours'])}** |"]
        if column.get("macro") is not None:
            lines.append(f"| 素の LLM（`{column['label']}` / ツールなし 1 ターン、"
                         f"{column['n_tasks']} タスク） | {_fmt(column['macro'])} |")
        for system, value in macro["systems"].items():
            lines.append(f"| {system} | {_fmt(value)} |")
        lines.append("")
    if column.get("macro") is not None:
        gap = None
        if column.get("ours_macro_same_subset") is not None:
            gap = round(column["ours_macro_same_subset"] - column["macro"], 4)
        lines += [f"`{column['label']}` は同じ問題・同じ採点で**ツールも Verifier も再計画も"
                  "使わずに**解かせた結果。ahc との差が harness の寄与にあたる:", "",
                  f"- 同一母集団（{column['n_tasks']} タスク）での ahc: "
                  f"**{_fmt(column.get('ours_macro_same_subset'))}** / "
                  f"素の LLM: **{_fmt(column['macro'])}** / 差: "
                  f"**{'+' if gap is not None and gap > 0 else ''}{_fmt(gap)}**", ""]
        rows_with = [(t, rows[t], v) for t, v in column["tasks"].items() if v is not None]
        if rows_with:
            lines += ["| task | 比較指標 | ahc | 素の LLM | 差 | 文献最高 | その手法 |",
                      "| --- | --- | --: | --: | --: | --: | --- |"]
            for task_id, row, value in sorted(rows_with,
                                              key=lambda x: (x[1]["ours"] or 0) - x[2]):
                diff = round((row["ours"] or 0) - value, 4)
                lines.append(
                    f"| {task_id} | {row.get('ours_field') or 'score'} "
                    f"| {_fmt(row['ours'])} | {_fmt(value)} "
                    f"| {'+' if diff > 0 else ''}{_fmt(diff)} "
                    f"| {_fmt(row['best'])} | {row['best_system'] or '-'} |")
            lines.append("")
    aggregates = [a for a in (compare.get("aggregates") or []) if a["comparable"]]
    if aggregates:
        lines += ["論文が複数タスクを 1 行に集約している項目:", "",
                  "| 論文タスク | 指標 | ahc 側のタスク | ahc | 文献最高 | その手法 | Δ |",
                  "| --- | --- | --- | --: | --: | --- | --: |"]
        for agg in aggregates:
            delta = _fmt(agg["delta"])
            if agg["delta"] is not None and agg["delta"] > 0:
                delta = f"+{delta}"
            lines.append(
                f"| {agg['paper_task']} | {agg['paper_metric']} "
                f"| {agg['n_tasks']} タスクの平均 | {_fmt(agg['ours'])} "
                f"| {_fmt(agg['best'])} | {agg['best_system'] or '-'} | {delta} |")
        lines.append("")
    skipped = [(t, r) for t, r in rows.items() if not r["comparable"]]
    skipped += [(",".join(a["task_ids"]), a) for a in (compare.get("aggregates") or [])
                if not a["comparable"]]
    if skipped:
        lines += ["文献値を `n/a` にしたタスク（論文と指標が違うため、並べると優劣を誤らせる）:", ""]
        for task_id, row in skipped:
            lines.append(f"- `{task_id}` — 論文 {row['paper_task']} は "
                         f"{row['paper_metric']}。{row.get('note', '')}")
        lines.append("")
    return lines


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
            f"- 実行モデル: {overall['models'] or '記録なし（この run より前の実装）'}",
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
        compared = (block.get("baselines") or {}).get("tasks") or {}
        lines += ["", "### タスク別", "",
                  "| level | dimension | task | metric | n | 主指標 | score | 回答率 "
                  "| 比較値(ahc) | 文献最高 | その手法 | Δ | 追加指標 |",
                  "| --- | --- | --- | --- | --: | --: | --: | --: | --- | --: | --- | --: | --- |"]
        for level, level_block in block["levels"].items():
            for task_id, task in level_block["tasks"].items():
                ref = compared.get(task_id) or {}
                if ref.get("comparable"):
                    best, system = _fmt(ref.get("best")), ref.get("best_system") or "-"
                    delta = _fmt(ref.get("delta"))
                    if ref.get("delta") is not None and ref["delta"] > 0:
                        delta = f"+{delta}"
                    # Δ を検算できるように、ahc 側で比べた値と（score 以外なら）その指標名も出す
                    field = ref.get("ours_field") or "score"
                    ours = _fmt(ref.get("ours"))
                    ours_cell = ours if field == "score" else f"{ours} ({field})"
                else:
                    ours_cell, best, system, delta = "-", "n/a", "-", "-"
                lines.append(
                    f"| {level} | {task.get('dimension') or '-'} | {task_id} | {task['metric']} "
                    f"| {task['n']} | {_fmt(task['primary_value'])} | {_fmt(task['score'])} "
                    f"| {_fmt(task['answered_rate'])} | {ours_cell} | {best} | {system} | {delta} "
                    f"| {_metric_cell(task)} |")
        lines += _render_baselines(block.get("baselines") or {})
        lines.append("")
    lines += ["## 注記", "",
              "- score は 0..1 で高いほど良い代表値。回帰タスク（RMSE/MAE）は尺度が違うため",
              "  score には含めず「主指標」列に出す。",
              "- judge 系タスクは `--judge` を指定したときだけ採点される（未指定なら score=None）。",
              "- Verifier 合格率は「ahc が答えファイルを作れたか」であり、答えの正しさとは別。",
              "- 「文献最高」は ChemEval 論文 Table 1（0-shot text）の 13 手法中の最高値を",
              "  0..1 に直したもの。正本は `benchmarks_chemeval/baselines.yaml`。**指標が一致する",
              "  タスクだけ**に入れ、違うものは `n/a`（理由は上の一覧）。Δ = ahc − 文献最高。",
              "- タスクによって ahc 側の比較値は score でなく tanimoto / exact_set_match を使う",
              "  （論文の主指標に合わせる）。どの値を使ったかは metrics.json の",
              "  `baselines.tasks.<task>.ours_field` にある。",
              ""]
    return "\n".join(lines)


def write_report(records: list[dict], out_dir: Path, label: str = "chemeval",
                 compare_records: list[dict] | None = None,
                 compare_label: str | None = None) -> dict:
    """metrics.json / report.md / scored.jsonl を書く。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records, label, compare_records=compare_records,
                        compare_label=compare_label)
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    with (out_dir / "scored.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            slim = {k: v for k, v in record.items() if k != "query"}
            fh.write(json.dumps(slim, ensure_ascii=False) + "\n")
    return summary
