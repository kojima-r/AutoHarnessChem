"""records.jsonl の採点（run とは分離してある）。

ChemEval と違って ChemBench は採点に rdkit も LLM-as-judge も要らない
（選択肢の完全一致と数値の 1% 以内だけ）。record が `score_map` / `target` /
`tolerance` を持っているので、**run を回さずに採点だけやり直せる**
（`evaluate score --label ...`）。

`ahc` 側と素の LLM 側で**同じ関数**を通すのが要点。採点経路が分かれると
harness の寄与を測れなくなる。
"""
from __future__ import annotations

from benchmarks_chembench.metrics import score_item


def score_records(records: list[dict]) -> list[dict]:
    """各 record に score / metrics / answered / valid を書き込んで返す。"""
    for record in records:
        try:
            result = score_item(record)
        except ValueError as e:
            # 採点に必要な情報が欠けた record（古い形式など）は落とさず印を付ける
            record.update(score=None, metrics={}, answered=False, valid=False,
                          score_detail={"error": str(e)})
            continue
        record.update(score=result["score"], metrics=result["metrics"],
                      answered=result["answered"], valid=result["valid"],
                      score_detail=result["detail"])
    return records


__all__ = ["score_records"]
