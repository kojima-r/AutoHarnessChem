"""本実装の採点が公式 ChemBench と一致するかの検証。

文献値と ahc を並べる前提は「**同じ規則で採点している**」ことなので、それを
主張ではなく実測で示すための仕掛け。公開 report には各モデルの回答本文
（`output.text`）と、公式実装がそのとき計算した指標（`metrics.hamming` /
`metrics.mae`）が両方入っている。そこで

  回答本文 → 本実装（`metrics.py`）で採点 → 公式が記録した正誤と比べる

を全問について行う。`evaluate validate` から呼ぶ。

比較の相手は**問題ごとの report に記録された指標**にする（集計 JSON の
`all_correct` ではない）。集計 JSON は別 run の結果を含むことがあり、選択肢の
並び順が違う場合に食い違うため、採点規則の検証には使えない。

refusal（回答拒否）は公式が本文を見ずに 0 点としているので、母集団から外して
件数だけ報告する（本実装は ahc の答えを採点する道具なので拒否検出は持たない）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from benchmarks_chembench.catalog import load_catalog
from benchmarks_chembench.dataset import REPORTS_DIR, _build_item
from benchmarks_chembench.metrics import score_item


def _clean(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def validate_model(model: str, *, limit: int | None = None) -> dict:
    """1 モデルの report を再採点して公式の記録と比べる。"""
    catalog = load_catalog()
    files = sorted((REPORTS_DIR / model).glob("reports/*/*.json"))
    if limit:
        files = files[:limit]
    stats = {"model": model, "n_files": len(files), "compared": 0, "agree": 0,
             "mcq": 0, "mcq_agree": 0, "numeric": 0, "numeric_agree": 0,
             "refusal": 0, "unresolved": 0, "no_stored": 0, "no_completion": 0,
             "disagreements": []}

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        report = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(report, dict):
            continue
        if report.get("triggered_refusal"):
            stats["refusal"] += 1
            continue
        meta = catalog.meta(report.get("name", ""))
        item = _build_item(report, meta, catalog, frozenset())
        if item is None:
            stats["unresolved"] += 1
            continue
        completion = (report.get("output") or {}).get("text") or ""
        if not completion:
            stats["no_completion"] += 1
            continue

        stored = report.get("metrics") or {}
        if item.metric_kind == "mcq":
            hamming = _clean(stored.get("hamming"))
            if hamming is None:
                stats["no_stored"] += 1
                continue
            theirs = int(hamming == 0)
        else:
            mae = _clean(stored.get("mae"))
            theirs = 0 if mae is None else int(mae < item.tolerance)

        record = {"question_name": item.question_name, "metric_kind": item.metric_kind,
                  "score_map": item.score_map, "target": item.target,
                  "tolerance": item.tolerance}
        mine = int(score_item(record, completion)["score"])

        stats["compared"] += 1
        key = "mcq" if item.metric_kind == "mcq" else "numeric"
        stats[key] += 1
        if mine == theirs:
            stats["agree"] += 1
            stats[f"{key}_agree"] += 1
        elif len(stats["disagreements"]) < 10:
            stats["disagreements"].append({
                "question_name": item.question_name, "kind": item.metric_kind,
                "ours": mine, "official": theirs,
                "tail": completion[-120:].replace("\n", " ")})

    stats["agreement"] = (round(stats["agree"] / stats["compared"], 4)
                          if stats["compared"] else None)
    stats["mcq_agreement"] = (round(stats["mcq_agree"] / stats["mcq"], 4)
                              if stats["mcq"] else None)
    stats["numeric_agreement"] = (round(stats["numeric_agree"] / stats["numeric"], 4)
                                  if stats["numeric"] else None)
    return stats


def validate(models: list[str], *, limit: int | None = None) -> dict:
    results = [validate_model(model, limit=limit) for model in models]
    compared = sum(r["compared"] for r in results)
    agree = sum(r["agree"] for r in results)
    return {"models": results, "compared": compared, "agree": agree,
            "agreement": round(agree / compared, 4) if compared else None}


__all__ = ["validate", "validate_model"]
