"""records.jsonl の採点（run とは分離してある）。

run を回さずに採点だけやり直せるようにしてある（`evaluate score --label ...`）。
分子系の指標は rdkit 環境へ一括で投げ、LLM 採点はキャッシュ付きで一括実行する。
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from benchmarks_chemeval import chem_metrics
from benchmarks_chemeval.metrics import CHEM_METRICS, JUDGE_METRICS, score_item

# metric → rdkit 側での扱い
_CHEM_KIND = {"smiles": "smiles", "selfies": "selfies", "reagent_f1": "reagent",
              "reaction_smiles": "reaction"}


def _chem_pairs(records: list[dict]) -> list[dict]:
    pairs = []
    for record in records:
        if record.get("metric") not in CHEM_METRICS:
            continue
        if record.get("answer") is None:
            continue
        pairs.append({
            "id": f"{record['provider']}:{record['item_id']}",
            "pred": str(record["answer"]),
            "gold": str(record.get("target", "")),
            "kind": _CHEM_KIND[record["metric"]],
        })
    return pairs


async def score_records(records: list[dict], *, config=None, judge=None,
                        workspace: Path | None = None, use_chem: bool = True) -> list[dict]:
    """各 record に score / metrics / answered / valid を書き込んで返す。"""
    chem_results: dict[str, dict] = {}
    pairs = _chem_pairs(records)
    if use_chem and pairs and config is not None:
        workspace = Path(workspace or (Path(config.paths.root) / "workspaces" / "chemeval-scoring"))
        computed = chem_metrics.compute(pairs, config.runtime.sandbox, workspace)
        chem_results = computed or {}
        if computed is None:
            print("[chemeval] rdkit が使えないため分子系は文字列一致で採点します")

    judged: dict[str, dict] = {}
    if judge is not None:
        requests = [
            {"item_id": f"{r['provider']}:{r['item_id']}", "task_id": r["task_id"],
             "query": r.get("query", ""), "target": r.get("target", ""),
             "answer": r.get("answer")}
            for r in records
            if r.get("metric") in JUDGE_METRICS and r.get("answer") is not None
        ]
        if requests and (judge.available or judge.cache):
            print(f"[chemeval] LLM-as-judge: {len(requests)} 件を採点します "
                  f"(provider={judge.provider})")
            judged = await judge.judge_many(requests)

    for record in records:
        key = f"{record['provider']}:{record['item_id']}"
        context = {}
        if key in chem_results:
            context["chem"] = chem_results[key]
        if key in judged:
            context["judge"] = judged[key]
        result = score_item(record.get("metric", ""), record.get("answer"),
                            record.get("target", ""), context)
        record.update(score=result["score"], metrics=result["metrics"],
                      answered=result["answered"], valid=result["valid"],
                      score_detail=result["detail"])
    return records


def score_records_sync(records: list[dict], **kwargs) -> list[dict]:
    return asyncio.run(score_records(records, **kwargs))
