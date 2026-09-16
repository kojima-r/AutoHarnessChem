"""ChemEval 論文（ICLR 2026）Table 1 の文献値の読み込みと、ahc の結果との突き合わせ。

正本は `baselines.yaml`。**採点には一切関与しない**（レポートに他手法の値を並べるだけ）。

比較で気をつけていること:

- **指標が一致する行だけを比較対象にする。** 論文が NRMSE（正規化済み）や原子組成の L2
  距離を使っているタスクは、ahc の RMSE / 一致率とは尺度が違うので `comparable: false`
  にして数値を並べない（並べると優劣が逆に見える）。
- **ahc 側で比べる値はタスクごとに指定する**（`ours_field`）。たとえば論文の
  IUPAC2SMILES の主指標は Tanimoto なので、ahc の score（正準 SMILES 完全一致）ではなく
  tanimoto と比べる。
- **同条件の比較ではない**ことを前提にする。論文は素の LLM の 0-shot、ahc は
  「ツール + Verifier + 再計画ループ」。この差はレポートの注記に明示する。
"""
from __future__ import annotations

import functools
from pathlib import Path

import yaml

BASELINES_PATH = Path(__file__).with_name("baselines.yaml")


@functools.lru_cache(maxsize=4)
def load_baselines(path: str | Path | None = None) -> dict:
    target = Path(path) if path else BASELINES_PATH
    if not target.exists():
        return {}
    return yaml.safe_load(target.read_text(encoding="utf-8")) or {}


def _ours(task_summary: dict, field: str) -> float | None:
    """ahc 側の比較値。`score` 以外は metrics から取る。"""
    if field == "score":
        return task_summary.get("score")
    value = (task_summary.get("metrics") or {}).get(field)
    return float(value) if value is not None else None


def _best(values: dict[str, float | None]) -> tuple[float | None, str | None]:
    """文献値の最高とその手法（大きいほど良い指標のみに使う）。"""
    known = [(v, k) for k, v in values.items() if v is not None]
    if not known:
        return None, None
    value, system = max(known)
    return value, system


def task_rows(tasks: dict[str, dict], baselines: dict | None = None) -> dict[str, dict]:
    """task_id → 比較行。`tasks` は summarize() が作るタスク別集計。"""
    baselines = baselines if baselines is not None else load_baselines()
    out: dict[str, dict] = {}
    for entry in baselines.get("tasks", []):
        task_id = entry["task_id"]
        summary = tasks.get(task_id)
        row = {"paper_task": entry["paper_task"], "paper_metric": entry["paper_metric"],
               "comparable": bool(entry.get("comparable")), "note": entry.get("note", ""),
               "ours_field": entry.get("ours_field"), "ours": None,
               "best": None, "best_system": None, "delta": None, "values": {}}
        if row["comparable"]:
            # 論文値は 0..100 スケール。ahc の 0..1 に合わせる
            row["values"] = {k: (v / 100.0 if v is not None else None)
                             for k, v in entry["values"].items()}
            best, system = _best(row["values"])
            row.update(best=best, best_system=system)
            if summary is not None:
                row["ours"] = _ours(summary, entry.get("ours_field", "score"))
            if row["ours"] is not None and best is not None:
                row["delta"] = round(row["ours"] - best, 4)
        else:
            row["values"] = dict(entry["values"])       # 生の値のまま（参照用）
        out[task_id] = row
    return out


def aggregate_rows(tasks: dict[str, dict], baselines: dict | None = None) -> list[dict]:
    """論文が複数タスクを 1 行に集約している項目（MolPC / MolPR）。"""
    baselines = baselines if baselines is not None else load_baselines()
    rows = []
    for entry in baselines.get("aggregates", []):
        members = [tasks[t] for t in entry["task_ids"]
                   if t in tasks and tasks[t].get("score") is not None]
        ours = (round(sum(m["score"] for m in members) / len(members), 4)
                if members else None)
        row = {"paper_task": entry["paper_task"], "paper_metric": entry["paper_metric"],
               "task_ids": entry["task_ids"], "comparable": bool(entry.get("comparable")),
               "note": entry.get("note", ""), "n_tasks": len(members), "ours": ours,
               "best": None, "best_system": None, "delta": None, "values": {}}
        if row["comparable"]:
            row["values"] = {k: (v / 100.0 if v is not None else None)
                             for k, v in entry["values"].items()}
            best, system = _best(row["values"])
            row.update(best=best, best_system=system)
            if ours is not None and best is not None:
                row["delta"] = round(ours - best, 4)
        else:
            row["values"] = dict(entry["values"])
        rows.append(row)
    return rows


def system_macro(rows: dict[str, dict], baselines: dict | None = None) -> dict:
    """比較可能なタスクだけを母集団にした、手法ごとのマクロ平均。

    全手法を**同じタスク集合**で平均する（ahc 側の値がある比較可能タスクのみ）。
    そうしないと欠測の多い手法が有利になる。
    """
    baselines = baselines if baselines is not None else load_baselines()
    systems = list(baselines.get("systems", []))
    usable = [r for r in rows.values()
              if r["comparable"] and r["ours"] is not None
              and all(r["values"].get(s) is not None for s in systems)]
    if not usable:
        return {"n_tasks": 0, "systems": {}, "ours": None}
    per_system = {s: round(sum(r["values"][s] for r in usable) / len(usable), 4)
                  for s in systems}
    return {"n_tasks": len(usable),
            "task_ids": sorted(t for t, r in rows.items() if r in usable),
            "ours": round(sum(r["ours"] for r in usable) / len(usable), 4),
            "systems": dict(sorted(per_system.items(), key=lambda kv: -kv[1]))}


def compare_column(rows: dict[str, dict], macro: dict, other_tasks: dict[str, dict],
                   label: str, all_task_ids: list[str] | None = None) -> dict:
    """別 run（例: 素の LLM ベースライン）を比較列として並べるための値。

    ahc 側と**同じ `ours_field`** で値を取る。文献値と違い、こちらは自分で同じ採点を
    通した結果なので**文献値が無いタスクでも比較できる**（`all_task_ids` を渡すと
    全タスクぶん値を返す）。マクロ平均だけは文献値と同じ 31 タスクの母集団で出し、
    そのうち相手側にも値がある分だけを使う（n を併記して母集団の違いが判るように）。
    """
    subset = list(macro.get("task_ids") or [])
    targets = list(all_task_ids) if all_task_ids is not None else subset
    values: dict[str, float | None] = {}
    fields: dict[str, str] = {}
    for task_id in targets:
        row = rows.get(task_id) or {}
        # 文献値と突き合わせるタスクは論文の主指標に合わせる。それ以外は score
        field = (row.get("ours_field") or "score") if row.get("comparable") else "score"
        fields[task_id] = field
        summary = other_tasks.get(task_id)
        values[task_id] = _ours(summary, field) if summary is not None else None
    usable = [t for t in subset if values.get(t) is not None]
    return {"label": label, "tasks": values, "fields": fields, "n_tasks": len(usable),
            "macro": round(sum(values[t] for t in usable) / len(usable), 4)
            if usable else None,
            # 同じ母集団で比べた ahc 側の平均（相手に欠測があると全体平均とずれるため）
            "ours_macro_same_subset": round(
                sum(rows[t]["ours"] for t in usable) / len(usable), 4) if usable else None}


def task_macro(rows: dict[str, dict], task_ids: list[str], key: str) -> float | None:
    """指定したタスク群の平均（文献最高 / ahc / 素の LLM をレベル単位で見るため）。"""
    values = [rows[t][key] for t in task_ids
              if t in rows and rows[t].get("comparable") and rows[t].get(key) is not None]
    return round(sum(values) / len(values), 4) if values else None


def compare(tasks: dict[str, dict], baselines: dict | None = None,
            other_tasks: dict[str, dict] | None = None,
            other_label: str | None = None) -> dict:
    """レポート用にまとめたもの。"""
    baselines = baselines if baselines is not None else load_baselines()
    if not baselines:
        return {}
    rows = task_rows(tasks, baselines)
    macro = system_macro(rows, baselines)
    out = {"source": baselines.get("source", {}),
           "systems": baselines.get("systems", []),
           "system_kinds": baselines.get("system_kinds", {}),
           "tasks": rows,
           "aggregates": aggregate_rows(tasks, baselines),
           "macro": macro}
    if other_tasks is not None and other_label:
        out["compare"] = compare_column(rows, macro, other_tasks, other_label,
                                        all_task_ids=list(tasks))
    return out
