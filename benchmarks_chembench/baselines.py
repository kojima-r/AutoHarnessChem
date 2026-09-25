"""公開 report からの文献値の読み込みと、ahc の結果との突き合わせ。

**採点には一切関与しない**（レポートに他手法の値を並べるだけ）。

ChemEval との一番大きな違いはここで、ChemBench は公式リポジトリが
**問題ごとの正誤（`all_correct`）を 37 モデル分公開している**。そのため

- 論文の表を転記する必要がない（転記ミスも、指標のずれも起きない）
- **ahc が解いた問題だけに母集団を絞って**文献値を再集計できる
  （「同じ問題を解いた上での比較」になる。ChemEval 側は論文のタスク平均しか
  無いので、こちらの方が比較としてずっと素直）

という 2 点が成り立つ。代わりに気をつけるのは

- **どの report が「ツールを使った構成」なのか**を明示する（`kind: agent`）。
  ahc はツールつきエージェントなので、素の LLM と並べるだけでは不公平になる。
  `*-react` と `paper-qa` が同じ土俵の相手。
- **部分集合で比べるときは n を必ず併記する。** モデルごとに解けた問題数が違う
  （refusal や欠測がある）ため、n を出さないと平均の意味が変わる。
- 人間は約 120 問の部分集合しか解いていないので、全体平均と直接は比べられない。
"""
from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import yaml

from benchmarks_chembench.catalog import CHEMBENCH_DIR

BASELINES_PATH = Path(__file__).with_name("baselines.yaml")
REPORTS_DIR = CHEMBENCH_DIR / "reports"
HUMANS_DIR = REPORTS_DIR / "humans"


@functools.lru_cache(maxsize=4)
def load_baselines(path: str | Path | None = None) -> dict:
    target = Path(path) if path else BASELINES_PATH
    if not target.exists():
        return {}
    return yaml.safe_load(target.read_text(encoding="utf-8")) or {}


def _clean(value) -> float | None:
    """`all_correct` を 0/1 に正規化（NaN / None は「無効」として落とす）。"""
    if value is None or isinstance(value, bool):
        return float(bool(value)) if isinstance(value, bool) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _aggregate_file(model: str) -> Path | None:
    """`reports/<model>/<model>.json`（名前が違う場合は直下の唯一の JSON）。"""
    directory = REPORTS_DIR / model
    preferred = directory / f"{model}.json"
    if preferred.exists():
        return preferred
    candidates = [p for p in sorted(directory.glob("*.json"))]
    return candidates[0] if len(candidates) == 1 else None


def _scores_from_aggregate(path: Path) -> dict[str, float]:
    # NaN を含む JSON なので json.loads の既定（NaN 許容）に任せる
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for entry in payload.get("model_scores") or []:
        name = entry.get("question_name")
        value = _clean(entry.get("all_correct"))
        if name and value is not None:
            out[name] = value
    return out


def _scores_from_question_reports(model: str) -> dict[str, float]:
    """集計 JSON が無いモデル用。問題ごとの report に**記録済みの指標**から復元する。

    再採点はしない（`hamming` / `mae` はその run のときに公式実装が計算した値）。
    `all_correct` の定義（MCQ は hamming==0、numeric は mae < 0.01*target）だけを
    当てはめる。
    """
    out: dict[str, float] = {}
    for path in sorted((REPORTS_DIR / model).glob("reports/*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        report = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(report, dict):
            continue
        name = report.get("name")
        metrics = report.get("metrics") or {}
        if not name:
            continue
        if report.get("triggered_refusal"):
            out[name] = 0.0              # 公式も refusal は 0 点として数える
            continue
        hamming = _clean(metrics.get("hamming"))
        if hamming is not None:
            out[name] = float(hamming == 0)
            continue
        mae = _clean(metrics.get("mae"))
        target = report.get("targets_")
        try:
            tolerance = 0.01 * float(target)
        except (TypeError, ValueError):
            continue
        out[name] = 0.0 if mae is None else float(mae < tolerance)
    return out


def _infer_kind(model: str) -> str:
    lowered = model.lower()
    if lowered.startswith("log_prob"):
        return "llm-logprob"
    if lowered.endswith("-react") or lowered == "paper-qa":
        return "agent"
    if lowered.endswith("-t-one"):
        return "llm-t1"
    if lowered == "random_baseline":
        return "random"
    return "llm"


@functools.lru_cache(maxsize=1)
def discover_models() -> dict[str, dict]:
    """成績を読めるモデル → {name, kind, tools, note, scores, source}。

    集計 JSON があればそれを、無ければ問題ごとの report から復元する。
    `reports/humans/` は参加者ごとの JSON なので擬似モデル 2 つにまとめる。
    """
    baselines = load_baselines()
    described = baselines.get("models") or {}
    kinds = baselines.get("kinds") or {}
    excluded = set(baselines.get("exclude") or [])
    found: dict[str, dict] = {}
    if not REPORTS_DIR.exists():
        return found

    for directory in sorted(REPORTS_DIR.iterdir()):
        if not directory.is_dir() or directory.name in excluded:
            continue
        model = directory.name
        if model == "humans":
            found.update(_discover_humans(baselines))
            continue
        aggregate = _aggregate_file(model)
        scores, source = {}, "aggregate"
        if aggregate is not None:
            scores = _scores_from_aggregate(aggregate)
        if not scores:
            scores, source = _scores_from_question_reports(model), "per-question"
        if not scores:
            continue
        meta = described.get(model) or {}
        kind = meta.get("kind") or _infer_kind(model)
        found[model] = {
            "name": meta.get("name") or model,
            "kind": kind, "kind_label": kinds.get(kind, kind),
            "tools": kind in ("agent", "human"),
            "note": meta.get("note", ""), "source": source, "scores": scores,
        }
    return found


def _discover_humans(baselines: dict) -> dict[str, dict]:
    """参加者ごとの JSON → 「人間（ツール使用可 / なし）」の 2 擬似モデル。

    問題ごとに参加者の正答率（0..1）を出してから平均する。参加者数も残す。
    """
    kinds = baselines.get("kinds") or {}
    described = baselines.get("humans") or {}
    buckets: dict[str, dict[str, list[float]]] = {"tool": {}, "notool": {}}
    participants: dict[str, set[str]] = {"tool": set(), "notool": set()}
    if not HUMANS_DIR.exists():
        return {}
    for path in sorted(HUMANS_DIR.glob("*_h_*.json")):
        condition = "notool" if path.stem.endswith("_h_notool") else "tool"
        participant = path.stem.rsplit("_h_", 1)[0]
        try:
            scores = _scores_from_aggregate(path)
        except (json.JSONDecodeError, OSError):
            continue
        if not scores:
            continue
        participants[condition].add(participant)
        for name, value in scores.items():
            buckets[condition].setdefault(name, []).append(value)

    out: dict[str, dict] = {}
    for condition, per_question in buckets.items():
        if not per_question:
            continue
        meta = described.get(condition) or {}
        out[f"humans-{condition}"] = {
            "name": meta.get("name") or f"humans ({condition})",
            "kind": "human", "kind_label": kinds.get("human", "human"),
            "tools": condition == "tool",
            "note": f"{len(participants[condition])} 名の平均",
            "source": "humans", "n_participants": len(participants[condition]),
            "scores": {name: sum(values) / len(values)
                       for name, values in per_question.items()},
        }
    return out


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def full_benchmark_topics(models: dict[str, dict] | None = None) -> dict[str, int]:
    """ベンチマーク全体のトピック別問題数（文献値の母集団そのもの）。

    出題データを読み直す（2,788 個の JSON を開く）のは重いので、公開 report の鍵から
    トピックを引いて数える。文献値の母集団をそのまま使うので、「この run が全体の
    どれだけを覆っているか」を文献値と同じ土俵で言える。

    採用するのは**最も多くのモデルが共有している問題数**を持つ report（= 2,788 問）。
    一部の report は 2,854 問を含む（ベンチマーク改訂前の問題が残っているもの）ので、
    「最も問題数の多い report」を選ぶと母集団が実際より大きくなり、カバー率が
    過小に出てしまう。
    """
    import collections

    from benchmarks_chembench.catalog import load_catalog

    models = models if models is not None else discover_models()
    if not models:
        return {}
    sizes = collections.Counter(len(m["scores"]) for m in models.values())
    common_size = sizes.most_common(1)[0][0]
    reference = next(m for m in models.values() if len(m["scores"]) == common_size)
    catalog = load_catalog()
    counts: dict[str, int] = {}
    for name in reference["scores"]:
        meta = catalog.meta(name)
        topic = meta.topic if meta else "unknown"
        counts[topic] = counts.get(topic, 0) + 1
    return counts


def coverage(records: list[dict], *, models: dict[str, dict] | None = None) -> dict:
    """この run がベンチマーク全体のどれだけを覆っているか、構成比が揃っているか。

    `--limit-per-topic N` はトピックから**均等に**取るので、N 指定の run は
    全体の構成比（chemical_preference と toxicity_and_safety で 6 割）を再現しない。
    公表値と横並びにできるのは全件（`--limit-per-topic 0`）のときだけなので、
    そこを読み手が判断できるように数字で出す。

    併せて `weighted_score` を返す。これは**トピック別の正答率を全体の構成比で
    加重平均したもの**で、部分実行でも公表値と比べられる推定値になる
    （ただしトピックごとの n が小さいと誤差が大きい）。
    """
    full = full_benchmark_topics(models)
    total = sum(full.values())
    by_topic: dict[str, list[dict]] = {}
    for record in records:
        by_topic.setdefault(record.get("topic", "unknown"), []).append(record)

    rows: dict[str, dict] = {}
    weighted: list[tuple[float, float]] = []
    for topic, size in sorted(full.items()):
        rows_in_topic = by_topic.get(topic, [])
        scores = [float(r["score"]) for r in rows_in_topic if r.get("score") is not None]
        share_full = size / total if total else None
        rows[topic] = {
            "n": len(rows_in_topic), "n_full": size,
            "coverage": round(len(rows_in_topic) / size, 4) if size else None,
            "share_run": None, "share_full": round(share_full, 4) if share_full else None,
            "score": _mean(scores),
        }
        if scores and share_full:
            weighted.append((share_full, sum(scores) / len(scores)))
    n_run = sum(r["n"] for r in rows.values())
    for row in rows.values():
        row["share_run"] = round(row["n"] / n_run, 4) if n_run else None

    # 加重平均は「覆えたトピック」だけを母集団にし、その重みで正規化する
    weight_sum = sum(w for w, _ in weighted)
    return {
        "n_run": n_run, "n_full": total,
        "coverage": round(n_run / total, 4) if total else None,
        "is_full": n_run >= total,
        "topics": rows,
        "topics_covered": sum(1 for r in rows.values() if r["n"]),
        "n_topics": len(rows),
        "weighted_score": (round(sum(w * s for w, s in weighted) / weight_sum, 4)
                           if weight_sum else None),
        "weighted_share": round(weight_sum, 4) if weight_sum else None,
        # 構成比が全体と一致しているか（全件なら自明に一致）
        "mix_matches_full": all(
            r["share_run"] is not None and r["share_full"] is not None
            and abs(r["share_run"] - r["share_full"]) < 0.01
            for r in rows.values() if r["n"]),
        # 偏りの原因。`--limit-per-topic N` の均等抽出なのか、全件 run の途中なのかで
        # 読み手の対処が違う（前者は設定を変える / 後者は待てば解消する）
        "equal_sampling": len({r["n"] for r in rows.values() if r["n"]}) == 1
                          and all(r["n"] < r["n_full"] for r in rows.values() if r["n"]),
    }


def model_summary(model: dict, question_names: list[str]) -> dict:
    """指定した問題集合でのその手法の成績（覆えた問題だけを母集団にする）。"""
    scores = model["scores"]
    covered = [scores[name] for name in question_names if name in scores]
    return {
        "name": model["name"], "kind": model["kind"],
        "kind_label": model["kind_label"], "tools": model["tools"],
        "note": model.get("note", ""), "source": model["source"],
        "n": len(covered), "coverage": _mean(
            [1.0 if name in scores else 0.0 for name in question_names]),
        "accuracy": _mean(covered),
        # 母集団の代表性を判断できるよう、ベンチマーク全体での値も出す
        "accuracy_full": _mean(list(scores.values())), "n_full": len(scores),
    }


def compare(records: list[dict], *, other_records: list[dict] | None = None,
            other_label: str | None = None, models: dict[str, dict] | None = None) -> dict:
    """ahc の結果と文献値（と素の LLM）を同じ問題集合で突き合わせる。

    `records` は採点済み（`score` が入っている）ahc の record 列。
    返る dict はレポートがそのまま表にできる形。
    """
    models = models if models is not None else discover_models()
    baselines = load_baselines()
    question_names = sorted({r["question_name"] for r in records if r.get("question_name")})
    if not question_names:
        return {}

    ours = _mean([float(r["score"]) for r in records if r.get("score") is not None])
    systems = {model_id: model_summary(model, question_names)
               for model_id, model in models.items()}

    other = None
    if other_records and other_label:
        # 相手側も**同じ問題だけ**を母集団にする（欠測があると平均が動くため）
        wanted = set(question_names)
        rows = [r for r in other_records
                if r.get("question_name") in wanted and r.get("score") is not None]
        covered = sorted({r["question_name"] for r in rows})
        other = {
            "label": other_label, "n": len(rows),
            "accuracy": _mean([float(r["score"]) for r in rows]),
            "scores": {r["question_name"]: float(r["score"]) for r in rows},
            # 相手に欠測があるときのために、同じ部分集合での ahc 側の値も出す
            "ours_same_subset": _mean(
                [float(r["score"]) for r in records
                 if r.get("question_name") in set(covered) and r.get("score") is not None]),
            "models": _count(rows, "model"),
        }

    return {
        "source": baselines.get("source", {}),
        "kinds": baselines.get("kinds", {}),
        "question_names": question_names,
        "n_questions": len(question_names),
        "ours": ours,
        "systems": dict(sorted(systems.items(),
                               key=lambda kv: -(kv[1]["accuracy"] or -1))),
        "other": other,
    }


def _count(records: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(field)
        if value:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def topic_table(records: list[dict], compared: dict, *,
                other_records: list[dict] | None = None,
                models: dict[str, dict] | None = None) -> dict[str, dict]:
    """トピックごとに ahc / 素の LLM / 文献最高 を並べるための表。"""
    models = models if models is not None else discover_models()
    other_scores = {}
    if other_records:
        other_scores = {r["question_name"]: float(r["score"]) for r in other_records
                        if r.get("question_name") and r.get("score") is not None}
    by_topic: dict[str, list[dict]] = {}
    for record in records:
        by_topic.setdefault(record.get("topic", "unknown"), []).append(record)

    out: dict[str, dict] = {}
    for topic, rows in by_topic.items():
        names = sorted({r["question_name"] for r in rows})
        ours = _mean([float(r["score"]) for r in rows if r.get("score") is not None])
        per_model = {}
        for model_id, model in models.items():
            covered = [model["scores"][n] for n in names if n in model["scores"]]
            if covered:
                per_model[model_id] = {"accuracy": _mean(covered), "n": len(covered)}
        # 文献最高は「その問題集合をすべて解いている手法」の中から選ぶ
        complete = {m: v for m, v in per_model.items() if v["n"] == len(names)}
        pool = complete or per_model
        best_id = max(pool, key=lambda m: pool[m]["accuracy"]) if pool else None
        bare_values = [other_scores[n] for n in names if n in other_scores]
        out[topic] = {
            "n": len(names), "ours": ours,
            "bare": _mean(bare_values), "n_bare": len(bare_values),
            "best": pool[best_id]["accuracy"] if best_id else None,
            "best_system": models[best_id]["name"] if best_id else None,
            "best_is_complete": bool(complete),
            "models": per_model,
        }
    return out


__all__ = ["compare", "coverage", "discover_models", "full_benchmark_topics",
           "load_baselines", "model_summary", "topic_table"]
