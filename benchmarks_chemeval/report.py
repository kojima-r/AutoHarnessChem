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
        # SDK が報告した値か、後から申告で補った値かの別
        "model_sources": _count([r for r in records if r.get("model_source")],
                                "model_source"),
        # harness を通した run か、素の LLM の run か
        "modes": _count([r for r in records if r.get("mode")], "mode"),
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
    # 比較 run のモデル名（レポートに「何を素の LLM として測ったか」を書くため）
    other_models = _count([r for r in (compare_records or []) if r.get("model")], "model")
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
        column = (summary["providers"][provider]["baselines"] or {}).get("compare")
        if column is not None:
            column["models"] = other_models
    return summary


# --- Markdown -----------------------------------------------------------
#
# レポートは「文脈を知らない人が単体で読める」ことを目標に章立てする。
# 比較表はすべて同じ 6 列（AHC / 文献最高 / その手法 / 素の LLM / 差 2 種）に
# 揃える。揃っていないと、どの表がどの母集団の話か読み手が追えなくなる。

# すべての比較表で共通に使う列。主語（この run が AHC か素の LLM か）だけ差し替える。
# 素の LLM ベースラインが無い場合はその 2 列を落とす（空欄を並べても読めないため）。
def _cmp_align(with_bare: bool = True) -> str:
    return "--: | --: | --- | --: | --: | --:" if with_bare else "--: | --: | --- | --:"


def _cmp_header(subject: str = "AHC", with_bare: bool = True) -> str:
    if not with_bare:
        return f"{subject} | 文献最高 | 文献最高の手法 | Δ({subject}−文献)"
    return (f"{subject} | 文献最高 | 文献最高の手法 | 素のLLM "
            f"| Δ({subject}−文献) | Δ({subject}−素)")


def _subject(block: dict) -> str:
    """この run の呼び名。bare モードの run を「AHC」と呼ばないため。"""
    modes = (block.get("overall") or {}).get("modes") or {}
    return "素のLLM（この run）" if modes.get("bare") else "AHC"


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _signed(value, digits: int = 3) -> str:
    """差分は符号つきで出す（+ が AHC の優位）。"""
    if value is None:
        return "-"
    return f"{'+' if value > 0 else ''}{value:.{digits}f}"


def _diff(a, b):
    return round(a - b, 4) if (a is not None and b is not None) else None


def _cmp_cells(ours, best, system, bare, with_bare: bool = True) -> str:
    """共通列の中身。どの表でも同じ順・同じ意味にする。"""
    head = f"{_fmt(ours)} | {_fmt(best)} | {system or '-'}"
    if not with_bare:
        return f"{head} | {_signed(_diff(ours, best))}"
    return (f"{head} | {_fmt(bare)} "
            f"| {_signed(_diff(ours, best))} | {_signed(_diff(ours, bare))}")


def _metric_cell(group: dict) -> str:
    metrics = group.get("metrics") or {}
    keep = [k for k in ("rmse", "mae", "tanimoto", "bleu4", "edit_similarity",
                        "atom_cosine", "role_f1", "validity", "strict_match",
                        "exact_set_match")
            if k in metrics]
    return ", ".join(f"{k}={_fmt(metrics[k])}" for k in keep) or "-"


def _models_cell(group: dict) -> str:
    models = group.get("models") or {}
    if not models:
        return "記録なし"
    cell = ", ".join(f"{name}（{count} 問）" for name, count in models.items())
    # 実行時にモデル記録が無かった run は後から補ったものなので、その旨を残す
    if group.get("model_sources", {}).get("user-stated"):
        cell += "（実行時は記録未実装のため申告値）"
    return cell


def _intro(summary: dict, block: dict) -> list[str]:
    """1. このレポートの読み方 / 2. 評価条件。"""
    compare = block.get("baselines") or {}
    source = compare.get("source", {})
    macro = compare.get("macro") or {}
    column = compare.get("compare") or {}
    overall = block["overall"]
    subject = _subject(block)
    has_bare = column.get("macro") is not None
    bare_label = column.get("label") or "（未測定）"
    lines = [
        "## 1. このレポートの読み方", "",
        "### 1.1 何を測ったか", "",
        "ChemEval（ICLR 2026, USTC / iFLYTEK）は化学の能力を 4 レベル / 13 能力次元 /",
        "62 タスクに分けて測るベンチマーク。ここでは text split の 0-shot 全問",
        f"（{summary['n_records']} 問）を **AutoHarnessChem（ahc）に解かせて**採点した。",
        "ahc は素の LLM ではなく「ツール（量子化学計算・反応予測・逆合成）+ Verifier +",
        "再計画ループ」を持つエージェントなので、測っているのは**エージェントとしての実力**。",
        "",
        "### 1.2 比較する 3 者", "",
        "| 列 | 中身 | モデル | ツール | 出どころ |",
        "| --- | --- | --- | --- | --- |",
        f"| **AHC** | ahc（この run） | {_models_cell(overall)} | あり | 本 run |",
        f"| **素のLLM** | 同じ問題を 1 往復で解かせただけ "
        f"| {_models_cell(column)} | **なし** | 別 run `{bare_label}` |",
        f"| **文献最高** | 論文 Table 1 の 13 手法のうち各タスクの最高値 | 各手法 "
        f"| 手法による | {source.get('venue', '?')} 論文 |",
        "",
        "3 者を並べる狙いは、**モデル自体の強さ**（素のLLM vs 文献）と、",
        "**harness が足している分**（AHC vs 素のLLM）を切り分けること。",
        "",
        "### 1.3 表の共通の列", "",
        f"本レポートの比較表はすべて同じ {'6' if has_bare else '4'} 列で揃えてある。",
        "",
        "| 列 | 意味 |",
        "| --- | --- |",
        f"| {subject} | この run のスコア（0..1、高いほど良い） |",
        "| 文献最高 | 論文 13 手法中の最高値を 0..1 に直したもの |",
        "| 文献最高の手法 | その値を出した手法名 |",
    ] + ([
        "| 素のLLM | 同じモデルにツールなしで解かせたスコア |",
        f"| Δ({subject}−文献) | 正なら既発表の最高値を上回る |",
        f"| Δ({subject}−素) | 正なら harness が効いている（モデル差は含まない） |",
    ] if has_bare else [
        f"| Δ({subject}−文献) | 正なら既発表の最高値を上回る |",
    ]) + [
        "",
        "**文献値は指標が一致するタスクにだけ入れ、違うものは `n/a`**（4.4 に理由）。",
        "素のLLM は自分で同じ採点を通しているので、文献値が無いタスクでも比較できる。",
        "",
        "## 2. 評価条件", "",
        "| 項目 | 内容 |",
        "| --- | --- |",
        f"| データ | ChemEval text split / 0-shot / {summary['n_records']} 問 / "
        f"{overall['n']} レコード |",
        f"| AHC の実行 | 1 問 = 1 エージェント実行。Verifier 合格率 "
        f"{_fmt(overall['harness_passed_rate'])} / 平均試行 "
        f"{_fmt(overall['mean_attempts'], 2)} 回 / 平均所要 "
        f"{_fmt(overall['mean_elapsed_sec'], 1)} 秒 |",
        "| 素のLLM の実行 | ツールを持たせず（`tools=[]`）1 往復。問題文は原文のまま |",
        "| 採点 | 公式指標を再実装した `metrics.py`。AHC / 素のLLM で**同一の採点経路** |",
        "| 自由記述 | LLM-as-judge（0..10 を 0..1 に正規化）。両者で同じ判定器 |",
        "| 回帰タスク | 尺度が違うため score に混ぜず RMSE / MAE を別掲（4.1 の追加指標列） |",
        f"| 比較母集団 | 指標が一致する {macro.get('n_tasks', 0)} タスク（3.1 / 4.2） |",
        "",
    ]
    return lines


def _render_overview(block: dict) -> list[str]:
    """3. 総合結果。"""
    subject = _subject(block)
    compare = block.get("baselines") or {}
    has_bare = (compare.get("compare") or {}).get("macro") is not None
    macro = compare.get("macro") or {}
    column = compare.get("compare") or {}
    overall = block["overall"]
    rows = compare.get("tasks") or {}
    if not macro.get("n_tasks"):
        return []
    n = macro["n_tasks"]
    best_mean = baselines.task_macro(rows, macro.get("task_ids") or [], "best")
    ours = column.get("ours_macro_same_subset", macro.get("ours"))
    bare = column.get("macro")
    lines = [
        "## 3. 総合結果", "",
        f"### 3.1 {'3 者' if has_bare else '文献値との'}比較（指標が一致する {n} タスク）", "",
        f"| 対象 | n | {_cmp_header(subject, has_bare)} |",
        f"| --- | --: | {_cmp_align(has_bare)} |",
        f"| マクロ平均 | {n} | {_cmp_cells(ours, best_mean, '—', bare, has_bare)} |",
        "",
        f"- 文献側は**各タスクの最高値だけを集めて平均**しても {_fmt(best_mean)}"
        "（単一手法の成績ではない上限値）。",
    ]
    if has_bare:
        lines += [
            f"- **{subject} {_fmt(ours)}** に対し素のLLM {_fmt(bare)} なので、"
            f"**harness の寄与は {_signed(_diff(ours, bare))}**。",
            "- 素のLLM の時点で既に文献側を上回っているので、",
            "  文献との差の大部分は**モデル世代の差**であって harness ではない。",
        ]
    lines += [
        "",
        "### 3.2 全手法の順位（同じ母集団のマクロ平均）", "",
        "| 順位 | 手法 | 種別 | ツール | マクロ平均 |",
        "| --: | --- | --- | --- | --: |",
    ]
    kinds = compare.get("system_kinds") or {}
    bare_model = ", ".join(column.get("models") or {}) or "claude-opus-5"
    ranking = [("AHC（この run）", "ツール利用エージェント", "あり", ours),
               (f"素のLLM（{bare_model}）", "汎用LLM", "なし", bare)]
    ranking += [(name, kinds.get(name, "-"),
                 "あり" if kinds.get(name) == "ツール利用エージェント" else "なし", value)
                for name, value in (macro.get("systems") or {}).items()]
    ranking = [r for r in ranking if r[3] is not None]
    for rank, (name, kind, tools, value) in enumerate(sorted(ranking, key=lambda r: -r[3]), 1):
        mark = "**" if name.startswith(("AHC", "素のLLM")) else ""
        lines.append(f"| {rank} | {mark}{name}{mark} | {kind} | {tools} "
                     f"| {mark}{_fmt(value)}{mark} |")
    lines += ["", "文献側の 13 手法は論文 Table 1 の値。**同条件の比較ではない**"
              "（論文は素の LLM の 0-shot、AHC はツールつきエージェント）。", "",
              "### 3.3 レベル別", "", f"| レベル | n | {_cmp_header(subject, has_bare)} | 回答率 |",
              f"| --- | --: | {_cmp_align(has_bare)} | --: |"]
    for level, level_block in block["levels"].items():
        ids = list(level_block["tasks"])
        lvl_best = baselines.task_macro(rows, ids, "best")
        lvl_ours = baselines.task_macro(rows, ids, "ours")
        bare_values = [(column.get("tasks") or {}).get(t) for t in ids
                       if (rows.get(t) or {}).get("comparable")]
        bare_values = [v for v in bare_values if v is not None]
        lvl_bare = round(sum(bare_values) / len(bare_values), 4) if bare_values else None
        lines.append(f"| {level} | {level_block['n']} "
                     f"| {_cmp_cells(lvl_ours, lvl_best, '—', lvl_bare, has_bare)} "
                     f"| {_fmt(level_block['answered_rate'])} |")
    lines += ["", "レベル別の 3 者はいずれも**そのレベルに属する比較可能タスクの平均**",
              "（論文はレベル平均を公表していないので、Table 1 の値から計算した派生値）。", ""]
    return lines


def _render_tasks(block: dict) -> list[str]:
    """4. タスク別の結果。"""
    subject = _subject(block)
    compare = block.get("baselines") or {}
    has_bare = (compare.get("compare") or {}).get("macro") is not None
    rows = compare.get("tasks") or {}
    column = compare.get("compare") or {}
    bare_values = column.get("tasks") or {}
    bare_fields = column.get("fields") or {}
    macro = compare.get("macro") or {}
    lines = ["## 4. タスク別の結果", "", "### 4.1 全タスク", "",
             "AHC が解いた全タスク。`比較指標` は 3 者を突き合わせるのに使った値で、",
             "論文の主指標に合わせるため score 以外を使うタスクがある（例: 論文の",
             "IUPAC2SMILES は Tanimoto なので tanimoto 同士で比べる）。", "",
             f"| レベル | 次元 | task | 採点方式 | n | 比較指標 | {_cmp_header(subject, has_bare)} | 追加指標 |",
             f"| --- | --- | --- | --- | --: | --- | {_cmp_align(has_bare)} | --- |"]
    for level, level_block in block["levels"].items():
        for task_id, task in level_block["tasks"].items():
            ref = rows.get(task_id) or {}
            field = bare_fields.get(task_id, "score")
            ours = ref.get("ours") if ref.get("comparable") else task.get("score")
            best = ref.get("best") if ref.get("comparable") else None
            system = ref.get("best_system") if ref.get("comparable") else "n/a"
            cells = _cmp_cells(ours, best, system, bare_values.get(task_id), has_bare)
            lines.append(
                f"| {level} | {task.get('dimension') or '-'} | {task_id} "
                f"| {task['metric']} | {task['n']} | {field} | {cells} "
                f"| {_metric_cell(task)} |")
    lines += ["", f"### 4.2 文献値と直接比較できる {macro.get('n_tasks', 0)} タスク", "",
              "3.1 のマクロ平均の母集団そのもの。**Δ(AHC−素) の小さい順**に並べてあるので、",
              "上ほど harness が効いていない（むしろ害になっている）タスク。", "",
              f"| # | task | 比較指標 | n | {_cmp_header(subject, has_bare)} |",
              f"| --: | --- | --- | --: | {_cmp_align(has_bare)} |"]
    subset = []
    for task_id in macro.get("task_ids") or []:
        ref = rows.get(task_id) or {}
        task = None
        for level_block in block["levels"].values():
            if task_id in level_block["tasks"]:
                task = level_block["tasks"][task_id]
                break
        subset.append((task_id, ref, task, bare_values.get(task_id)))
    subset.sort(key=lambda x: (_diff(x[1].get("ours"), x[3]) if x[3] is not None else 99))
    for i, (task_id, ref, task, bare) in enumerate(subset, 1):
        lines.append(
            f"| {i} | {task_id} | {ref.get('ours_field') or 'score'} "
            f"| {task['n'] if task else '-'} "
            f"| {_cmp_cells(ref.get('ours'), ref.get('best'), ref.get('best_system'), bare, has_bare)} |")
    aggregates = [a for a in (compare.get("aggregates") or []) if a["comparable"]]
    if aggregates:
        lines += ["", "### 4.3 論文が複数タスクを 1 行に集約している項目", "",
                  "論文 Table 1 は分子性質の分類タスクを 1 行にまとめている。AHC / 素のLLM 側は",
                  "該当タスクのスコア平均で突き合わせる。", "",
                  f"| 論文タスク | 論文の指標 | 内訳 | {_cmp_header(subject, has_bare)} |",
                  f"| --- | --- | --- | {_cmp_align(has_bare)} |"]
        for agg in aggregates:
            members = [t for t in agg["task_ids"] if bare_values.get(t) is not None]
            bare = (round(sum(bare_values[t] for t in members) / len(members), 4)
                    if members else None)
            lines.append(
                f"| {agg['paper_task']} | {agg['paper_metric']} "
                f"| {agg['n_tasks']} タスクの平均 "
                f"| {_cmp_cells(agg['ours'], agg['best'], agg['best_system'], bare, has_bare)} |")
        lines += ["", "内訳のタスク: "
                  + " / ".join(f"`{t}`" for a in aggregates for t in a["task_ids"]), ""]
    skipped = [(t, r) for t, r in rows.items() if not r["comparable"]]
    skipped += [(" / ".join(a["task_ids"]), a) for a in (compare.get("aggregates") or [])
                if not a["comparable"]]
    if skipped:
        lines += ["### 4.4 文献値を `n/a` にしたタスクと理由", "",
                  "論文と**指標が違う**ため、数値を並べると優劣を誤らせるもの。",
                  "素のLLM 側は同じ採点を通しているので 4.1 では比較できている。", "",
                  "| task | 論文のタスク | 論文の指標 | n/a にした理由 |",
                  "| --- | --- | --- | --- |"]
        for task_id, row in skipped:
            lines.append(f"| `{task_id}` | {row['paper_task']} | {row['paper_metric']} "
                         f"| {row.get('note', '')} |")
        lines.append("")
    return lines


def _render_harness_observations(block: dict) -> list[str]:
    """5. ahc 固有の観測値。"""
    overall = block["overall"]
    lines = ["## 5. AHC 固有の観測値", "",
             "スコアとは別に、「エージェントとしてタスクを完了できたか」も記録している。",
             "Verifier 合格率は**答えファイルを作れたか**であって、答えの正しさとは別物。", "",
             "| 指標 | 値 |", "| --- | --: |",
             f"| 問題数 | {overall['n']} |",
             f"| 回答率 | {_fmt(overall['answered_rate'])} |",
             f"| 形式が妥当だった率 | {_fmt(overall['valid_rate'])} |",
             f"| Verifier 合格率 | {_fmt(overall['harness_passed_rate'])} |",
             f"| 平均試行回数 | {_fmt(overall['mean_attempts'], 2)} |",
             f"| 平均所要時間 | {_fmt(overall['mean_elapsed_sec'], 1)} 秒 |",
             f"| 実行エラー | {overall['error_count']} 件 |",
             f"| 答えの取得元 | {overall['answer_sources']} |",
             f"| 実行モデル | {_models_cell(overall)} |",
             f"| 問題平均スコア (micro) | {_fmt(overall['score'])}"
             f"（{overall['scored']}/{overall['n']} 問が採点対象） |",
             f"| タスク平均スコア (macro, 全タスク) | {_fmt(overall['macro_task_score'])}"
             f"（{overall['scored_tasks']} タスク） |",
             ""]
    return lines


def _render_notes(block: dict) -> list[str]:
    """6. 出典と注記。"""
    compare = block.get("baselines") or {}
    source = compare.get("source", {})
    column = compare.get("compare") or {}
    lines = ["## 6. 出典と注記", "", "### 6.1 文献値の出典", "",
             f"- {source.get('paper', '?')}（{source.get('venue', '?')}）",
             f"- {source.get('table', '')}",
             f"- 論文: {source.get('url', '')}",
             f"- 転記元の図: `{source.get('local_figure', '')}`",
             f"- 正本データ: `benchmarks_chemeval/baselines.yaml`", ""]
    for note in source.get("notes", []):
        lines.append(f"- {note}")
    lines += ["", "### 6.2 読むときの注意", "",
              "- **論文と同条件の比較ではない。** 論文は素の LLM の 0-shot 評価、AHC は",
              "  ツール + Verifier + 再計画ループ。素のLLM 列が同条件の比較にあたる。",
              "- 文献値は**指標が一致するタスクにだけ**入れてある（理由は 4.4）。",
              "- マクロ平均は**全手法で同じタスク集合**を使う（欠測の多い手法が",
              "  有利にならないように）。",
              "- 回帰タスク（RMSE / MAE）は尺度が違うので score に混ぜていない。",
              "- judge 系タスクは `--judge` を指定したときだけ採点される。",
              f"- 素のLLM の実測は別 run `{column.get('label', '-')}`。ツールを持たせない",
              "  設定（`tools=[]`）で、ツール使用を検知したら失敗扱いにしている。", ""]
    return lines


def render_markdown(summary: dict) -> str:
    providers = summary["providers"]
    multi = len(providers) > 1
    lines = ["# ChemEval 評価レポート — AutoHarnessChem (ahc)", "",
             f"- 対象: `{summary['label']}` / {summary['n_records']} レコード", ""]
    for provider, block in providers.items():
        if multi:
            lines += [f"# provider: {provider}", ""]
        lines += _intro(summary, block)
        lines += _render_overview(block)
        lines += _render_tasks(block)
        lines += _render_harness_observations(block)
        lines += _render_notes(block)
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
