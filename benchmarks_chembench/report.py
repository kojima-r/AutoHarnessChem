"""採点済み record の集計と Markdown レポートの出力。

集計の軸は ChemBench 論文と同じ「トピック」に加えて、**採点方式**（選択肢 / 数値）と
**要求能力**（Knowledge / Reasoning / Calculation / Intuition）。後者 2 つは
「harness がどこで効くか」を見るために足したもので、Bash で計算できる問題と
知識を思い出すだけの問題では harness の寄与が違うはず、という見立てを検証する軸。

ahc 固有の観測値（Verifier 合格率・試行回数・所要時間・答えの取得元）も併記する。
ahc の評価では「答えの正しさ」と「harness がタスクを完了できたか」は別物なので、
両方載せないと結果を読み違える。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from benchmarks_chembench import baselines
from benchmarks_chembench.catalog import TOPICS, load_catalog

# 集計に出す要求能力の順（問題数の多い順）
REQUIRES_ORDER = ("Intuition", "Knowledge", "Reasoning", "Knowledge and Reasoning",
                  "Calculation", "Calculation and Reasoning", "Calculation and Knowledge",
                  "Calculation, Knowledge and Reasoning")
METRIC_KIND_LABEL = {"mcq": "選択肢（完全一致）", "numeric": "数値（正解の1%以内）"}


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _count(records: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = record.get(field)
        if key in (None, ""):
            continue
        counts[str(key)] = counts.get(str(key), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _group(records: list[dict], field: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record.get(field) or "unknown"), []).append(record)
    return grouped


def aggregate_metrics(records: list[dict]) -> dict[str, float]:
    """record の metrics を平均（mse は RMSE に直す）。"""
    buckets: dict[str, list[float]] = {}
    for record in records:
        for name, value in (record.get("metrics") or {}).items():
            if value is None:
                continue
            buckets.setdefault(name, []).append(float(value))
    out: dict[str, float] = {}
    for name, values in buckets.items():
        if name == "mse":
            out["rmse"] = round(math.sqrt(sum(values) / len(values)), 4)
        else:
            out[name] = round(sum(values) / len(values), 4)
    return out


def summarize_group(records: list[dict]) -> dict:
    """1 グループ（全体・トピック・採点方式など）の集計。"""
    scores = [float(r["score"]) for r in records if r.get("score") is not None]
    elapsed = [r["elapsed_sec"] for r in records if r.get("elapsed_sec") is not None]
    attempts = [r["attempts"] for r in records if r.get("attempts")]
    kinds = sorted({r.get("metric_kind", "") for r in records if r.get("metric_kind")})
    return {
        "n": len(records),
        "metric_kind": kinds[0] if len(kinds) == 1 else "mixed",
        # ChemBench の代表値は all_correct の平均（= 正答率）
        "score": _mean(scores),
        "scored": len(scores),
        "answered_rate": _mean([float(bool(r.get("answered"))) for r in records]),
        "valid_rate": _mean([float(bool(r.get("valid"))) for r in records]),
        # bare モードは Verifier を通していない（harness_passed=None）。None を 0 と
        # 数えると「Verifier に落ちた」と読めるので母集団から除く
        "harness_passed_rate": _mean([float(bool(r.get("harness_passed"))) for r in records
                                      if r.get("harness_passed") is not None]),
        "error_count": sum(1 for r in records if r.get("error")),
        "mean_attempts": _mean([float(a) for a in attempts]),
        "mean_elapsed_sec": _mean([float(e) for e in elapsed]),
        "metrics": aggregate_metrics(records),
        "answer_sources": _count(records, "answer_source"),
        # どのモデルの成績かを明示する（文献値と並べる以上、必須の情報）
        "models": _count(records, "model"),
        # harness を通した run か、素の LLM の run か
        "modes": _count(records, "mode"),
    }


def _breakdown(records: list[dict], field: str, order: tuple[str, ...]) -> dict[str, dict]:
    grouped = _group(records, field)
    known = [k for k in order if k in grouped]
    rest = sorted(k for k in grouped if k not in order)
    return {key: summarize_group(grouped[key]) for key in [*known, *rest]}


def summarize(records: list[dict], label: str = "chembench",
              compare_records: list[dict] | None = None,
              compare_label: str | None = None) -> dict:
    """provider ごと・トピックごと・採点方式ごと・要求能力ごとの集計。"""
    catalog = load_catalog()
    summary: dict = {"label": label, "n_records": len(records), "providers": {}}
    models = baselines.discover_models()

    for provider, rows in sorted(_group(records, "provider").items()):
        other = None
        if compare_records:
            other = [r for r in compare_records if r.get("score") is not None]
        compared = baselines.compare(rows, other_records=other,
                                     other_label=compare_label, models=models)
        block = {
            "overall": summarize_group(rows),
            "topics": _breakdown(rows, "topic", TOPICS),
            "metric_kinds": _breakdown(rows, "metric_kind", ("mcq", "numeric")),
            "requires": _breakdown(rows, "requires", REQUIRES_ORDER),
            "baselines": compared,
            "coverage": baselines.coverage(rows, models=models),
            "topic_table": baselines.topic_table(rows, compared, other_records=other,
                                                 models=models),
        }
        # 人間との比較は人間が解いた部分集合でしか成立しない
        human_rows = [r for r in rows if r.get("in_human_subset")]
        if human_rows:
            human_other = ([r for r in other if r.get("in_human_subset")]
                           if other else None)
            block["human_subset"] = {
                "overall": summarize_group(human_rows),
                "baselines": baselines.compare(human_rows, other_records=human_other,
                                               other_label=compare_label, models=models),
            }
        block["topic_names"] = {t.id: t.name for t in catalog}
        summary["providers"][provider] = block
    return summary


# --- Markdown -----------------------------------------------------------
#
# レポートは「文脈を知らない人が単体で読める」ことを目標に章立てする。
# 比較表はすべて同じ列（AHC / 文献最高 / その手法 / 素のLLM / 差 2 種）に揃える。

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


def _subject(block: dict) -> str:
    """この run の呼び名（bare モードの run を「AHC」と呼ばないため）。"""
    modes = (block.get("overall") or {}).get("modes") or {}
    return "素のLLM（この run）" if modes.get("bare") else "AHC"


def _cmp_align(with_bare: bool = True) -> str:
    return "--: | --: | --- | --: | --: | --:" if with_bare else "--: | --: | --- | --:"


def _cmp_header(subject: str = "AHC", with_bare: bool = True) -> str:
    if not with_bare:
        return f"{subject} | 文献最高 | 文献最高の手法 | Δ({subject}−文献)"
    return (f"{subject} | 文献最高 | 文献最高の手法 | 素のLLM "
            f"| Δ({subject}−文献) | Δ({subject}−素)")


def _cmp_cells(ours, best, system, bare, with_bare: bool = True) -> str:
    head = f"{_fmt(ours)} | {_fmt(best)} | {system or '-'}"
    if not with_bare:
        return f"{head} | {_signed(_diff(ours, best))}"
    return (f"{head} | {_fmt(bare)} "
            f"| {_signed(_diff(ours, best))} | {_signed(_diff(ours, bare))}")


def _models_cell(group: dict) -> str:
    models = group.get("models") or {}
    if not models:
        return "記録なし"
    return ", ".join(f"{name}（{count} 問）" for name, count in models.items())


def _best_overall(compared: dict, *, agents_only: bool = False) -> tuple:
    """同じ問題を全部解いている手法の中から最高を選ぶ（n の違いで有利にしない）。"""
    systems = compared.get("systems") or {}
    total = compared.get("n_questions") or 0
    pool = {k: v for k, v in systems.items()
            if v["n"] == total and v["accuracy"] is not None
            and (not agents_only or v["kind"] == "agent")}
    if not pool:
        pool = {k: v for k, v in systems.items()
                if v["accuracy"] is not None and (not agents_only or v["kind"] == "agent")}
    if not pool:
        return None, None, False
    best = max(pool, key=lambda k: pool[k]["accuracy"])
    complete = pool[best]["n"] == total
    return pool[best]["accuracy"], systems[best]["name"], complete


def _intro(summary: dict, block: dict) -> list[str]:
    compared = block.get("baselines") or {}
    source = compared.get("source", {})
    other = compared.get("other") or {}
    overall = block["overall"]
    subject = _subject(block)
    has_bare = other.get("accuracy") is not None
    bare_label = other.get("label") or "（未測定）"
    n_systems = len(compared.get("systems") or {})
    is_bare = subject != "AHC"
    # bare モード単独のレポートを「ahc の成績」と読ませないため、1.1 / 1.2 の
    # 主語も差し替える（列見出しだけ直しても本文が嘘になる）
    what = ([f"ここではそのうち **{compared.get('n_questions', 0)} 問**を",
             "**素の LLM（ツールなし・1 往復）に解かせて**採点した。harness を",
             "通していないので、これは文献値と同条件のベースラインであって",
             "ahc の成績ではない（ahc の成績は harness 側の run のレポートを見る）。"]
            if is_bare else
            [f"ここではそのうち **{compared.get('n_questions', 0)} 問**を",
             "**AutoHarnessChem（ahc）に解かせて**採点した。ahc は素の LLM ではなく",
             "「ツール（Bash・RDKit・量子化学計算）+ Verifier + 再計画ループ」を持つ",
             "エージェントなので、測っているのは**エージェントとしての実力**。"])
    subject_row = (f"| **素のLLM** | この run（ツールなし・1 往復） "
                   f"| {_models_cell(overall)} | **なし** | 本 run |" if is_bare else
                   f"| **AHC** | ahc（この run） | {_models_cell(overall)} "
                   f"| あり | 本 run |")
    other_row = (f"| **AHC** | ツール + Verifier + 再計画ループ "
                 f"| {_models_cell(other)} | あり | 別 run `{bare_label}` |" if is_bare else
                 f"| **素のLLM** | 同じ問題を 1 往復で解かせただけ "
                 f"| {_models_cell(other)} | **なし** | 別 run `{bare_label}` |")
    return [
        "## 1. このレポートの読み方", "",
        "### 1.1 何を測ったか", "",
        "ChemBench（*Are large language models superhuman chemists?*,",
        "arXiv:2404.01475, Jablonka ら）は化学・材料の能力を",
        "**9 トピック / 2,788 問**で測る人手作成ベンチマーク。",
        *what, "",
        "### 1.2 比較する 3 者", "",
        "| 列 | 中身 | モデル | ツール | 出どころ |",
        "| --- | --- | --- | --- | --- |",
        subject_row, other_row,
        f"| **文献** | 公式リポジトリが公開している {n_systems} 手法の成績 | 各手法 "
        f"| 手法による | `{source.get('reports_dir', '')}` |",
        "",
        "3 者を並べる狙いは、**モデル自体の強さ**（素のLLM vs 文献）と、",
        "**harness が足している分**（AHC vs 素のLLM）を切り分けること。",
        "",
        "### 1.3 ChemBench の文献値は「同じ問題での比較」になっている", "",
        "ChemEval 版レポート（`benchmarks_chemeval/`）では論文の表を転記した",
        "タスク平均しか無かったが、**ChemBench は問題ごとの正誤を公開している**。",
        "そのため文献値を**ahc が解いた問題だけに絞って再集計**してある。",
        "指標のずれも転記ミスも無く、母集団も揃っている。", "",
        "ただし n（各手法がその問題集合のうち何問解いたか）は手法ごとに違う",
        "（refusal や欠測があるため）。順位表では**全問解いている手法**の中から",
        "最高値を選び、n も併記する。", "",
        "### 1.4 表の共通の列", "",
        f"本レポートの比較表はすべて同じ {'6' if has_bare else '4'} 列で揃えてある。",
        "",
        "| 列 | 意味 |",
        "| --- | --- |",
        f"| {subject} | この run の正答率（0..1、高いほど良い） |",
        "| 文献最高 | 同じ問題を解いた文献手法のうち最高の正答率 |",
        "| 文献最高の手法 | その値を出した手法名 |",
    ] + ([
        "| 素のLLM | 同じモデルにツールなしで解かせた正答率 |",
        f"| Δ({subject}−文献) | 正なら既発表の最高値を上回る |",
        f"| Δ({subject}−素) | 正なら harness が効いている（モデル差は含まない） |",
    ] if has_bare else [
        f"| Δ({subject}−文献) | 正なら既発表の最高値を上回る |",
    ]) + [
        "",
        "## 2. 評価条件", "",
        "| 項目 | 内容 |",
        "| --- | --- |",
        f"| データ | ChemBench {compared.get('n_questions', 0)} 問 / "
        f"{overall['n']} レコード（全 2,788 問からトピックごとに抽出） |",
        "| 出題 | 文献の各モデルが見たプロンプトを**原文のまま**渡した"
        "（選択肢の並び順まで同一） |",
        f"| AHC の実行 | 1 問 = 1 エージェント実行。Verifier 合格率 "
        f"{_fmt(overall['harness_passed_rate'])} / 平均試行 "
        f"{_fmt(overall['mean_attempts'], 2)} 回 / 平均所要 "
        f"{_fmt(overall['mean_elapsed_sec'], 1)} 秒 |",
        "| 素のLLM の実行 | ツールを持たせず（`tools=[]`）1 往復 |",
        "| 採点 | 公式の規則を再実装した `metrics.py`。"
        "選択肢は**完全一致**（部分点なし）、数値は**正解の 1% 以内** |",
        "| 採点経路 | AHC / 素のLLM / 文献値で**同一の規則**"
        "（公開 report で照合済み。6 節参照） |",
        "",
    ]


def _render_coverage(block: dict) -> list[str]:
    """2.1 母集団。文献値の公表条件（全 2,788 問）とどれだけ揃っているか。

    `--limit-per-topic N` はトピックから均等に取るので、N 指定の run は
    ベンチマークの構成比を再現しない。ChemBench は**難しいトピックほど問題数が多い**
    （chemical_preference 36% / toxicity_and_safety 24%）ため、均等抽出すると
    正答率が実際より高く出る。ここを黙っていると公表値と取り違えられる。
    """
    cover = block.get("coverage") or {}
    if not cover.get("n_full"):
        return []
    names = block.get("topic_names") or {}
    subject = _subject(block)
    is_full = cover.get("is_full")
    lines = ["### 2.1 母集団（文献値の公表条件と揃っているか）", "",
             f"| 項目 | 値 |", "| --- | --- |",
             f"| 対象 | 全 {cover['n_full']} 問中 **{cover['n_run']} 問**"
             f"（カバー率 {_fmt(cover.get('coverage'))}） |",
             f"| トピックの構成比 | "
             + ("**全体と一致**（全件実行なので公表値と直接比較できる）" if is_full else
                "**全体と不一致**（トピックから均等に抽出しているため）"
                if cover.get("equal_sampling") else
                "**全体と不一致**（全件 run の途中。進めば解消する）") + " |",
             ""]
    if is_full:
        lines += ["全件を回しているので、この run の正答率は**文献値の公表条件"
                  "（全 2,788 問）と同じ母集団**で、3.2 の「全 2,788 問での正答率」列と"
                  "直接並べられる。", ""]
        return lines

    weighted = cover.get("weighted_score")
    simple = (block.get("overall") or {}).get("score")
    lines += [
        "**この run の単純平均は公表値と並べられない。** ChemBench は",
        "**難しいトピックほど問題数が多い**（`chemical_preference` が全体の "
        f"{_fmt(cover['topics'].get('chemical_preference', {}).get('share_full'), 2)}、"
        f"`toxicity_and_safety` が "
        f"{_fmt(cover['topics'].get('toxicity_and_safety', {}).get('share_full'), 2)}）"
        "ため、",
        ("トピックから均等に取ると簡単なトピックの重みが上がり、正答率が高く出る。"
         if cover.get("equal_sampling") else
         "いま重みの大きいトピックの成績が、全体の正答率を実際より高く見せている。"
         "実行を進めれば構成比は全体に近づく（実行順は構成比の不足順にしてある）。"), "",
        "| 集計のしかた | 値 | 意味 |", "| --- | --: | --- |",
        f"| 単純平均（この run の {cover['n_run']} 問） | {_fmt(simple)} "
        + ("| 均等抽出なので**公表値とは比較不可** |" if cover.get("equal_sampling")
           else "| 構成比が偏っているので**公表値とは比較不可** |"),
        f"| 構成比で加重した推定値 | **{_fmt(weighted)}** "
        "| 全体の構成比に直した値。**公表値と比べるならこちら** |",
        "",
        f"差が {_fmt(_diff(simple, weighted))} あるのは、{subject} が苦手なトピック"
        "（下表でスコアの低いもの）が全体では大きな割合を占めるため。",
        "",
        "| トピック | この run n | 全体 n | この run の重み | 全体の重み "
        f"| {subject} の正答率 |",
        "| --- | --: | --: | --: | --: | --: |",
    ]
    for topic, row in cover["topics"].items():
        if not row["n"]:
            continue
        lines.append(f"| {names.get(topic, topic)} | {row['n']} | {row['n_full']} "
                     f"| {_fmt(row['share_run'])} | {_fmt(row['share_full'])} "
                     f"| {_fmt(row['score'])} |")
    lines += ["",
              "加重推定値はトピックごとの n が小さいと誤差が大きい。**公表値と厳密に"
              "比べるには `--limit-per-topic 0`（全件）で回す**必要がある",
              "（README の「文献値と同じ設定で回す」を参照）。", ""]
    return lines


def _render_overview(block: dict) -> list[str]:
    subject = _subject(block)
    compared = block.get("baselines") or {}
    if not compared:
        return []
    other = compared.get("other") or {}
    has_bare = other.get("accuracy") is not None
    overall = block["overall"]
    n = compared.get("n_questions", 0)
    best, best_system, complete = _best_overall(compared)
    ours = other.get("ours_same_subset") if has_bare else compared.get("ours")
    ours = ours if ours is not None else compared.get("ours")
    bare = other.get("accuracy")
    lines = [
        "## 3. 総合結果", "",
        f"### 3.1 {'3 者' if has_bare else '文献値との'}比較（同一の {n} 問）", "",
        f"| 対象 | n | {_cmp_header(subject, has_bare)} |",
        f"| --- | --: | {_cmp_align(has_bare)} |",
        f"| 全体 | {n} | {_cmp_cells(ours, best, best_system, bare, has_bare)} |",
        "",
    ]
    if not complete:
        lines += ["- 文献最高の手法はこの問題集合を全問解いていない"
                  "（refusal / 欠測あり）。n の差に注意。", ""]
    if has_bare:
        gap = _diff(ours, bare)
        lines += [
            f"- **{subject} {_fmt(ours)}** に対し素のLLM {_fmt(bare)} なので、"
            f"**harness の寄与は {_signed(gap)}**。",
        ]
        # 0.014 のような小さい差を「傾向」と読ませない。1 問 = 1/n なので
        # 差が何問ぶんなのかを必ず併記する（n が小さいほど 1 問の重みが大きい）
        if gap is not None and n:
            questions = abs(gap) * n
            # 差が「何問ぶん」かを必ず併記する。0.005 のような小さい数字だけだと、
            # n が 70 のときと 2,788 のときで意味がまるで違うのに同じに見える
            if questions <= 2:
                note = "**傾向として読める差ではない**（n が小さく 1 問の重みが大きい）。"
            elif abs(gap) < 0.01:
                note = ("全体の 1% 未満なので、**harness による改善も悪化も"
                        "見て取れない**。")
            else:
                note = "より大きな n で確かめる必要がある。"
            lines += [
                f"  この差は **{questions:.0f} 問ぶん**（{n} 問中。1 問 = "
                f"{1 / n:.4f}）。{note}",
            ]
    agent_best, agent_system, _ = _best_overall(compared, agents_only=True)
    if agent_best is not None:
        lines += [
            f"- **同じ「ツールを使うエージェント」同士**では文献側の最高が "
            f"{_fmt(agent_best)}（{agent_system}）で、差は "
            f"{_signed(_diff(ours, agent_best))}。素の LLM と並べるより"
            "こちらが同条件の比較。",
        ]
    cover = block.get("coverage") or {}
    if not cover.get("is_full") and cover.get("weighted_score") is not None:
        lines += [
            f"- 上の表は**この {n} 問だけ**での比較（文献値も同じ {n} 問に絞って"
            "再集計している）なので、3 者の比較としては妥当。",
            f"  ただし {subject} の {_fmt(ours)} は**公表されている全 2,788 問の値とは"
            f"並べられない**（均等抽出のため）。構成比に直した推定値は "
            f"**{_fmt(cover['weighted_score'])}**（2.1 参照）。",
        ]
    lines += ["",
              f"### 3.2 全手法の順位（同じ {n} 問）", "",
              "| 順位 | 手法 | 種別 | ツール | n | 正答率 | 全 2,788 問での正答率 |",
              "| --: | --- | --- | --- | --: | --: | --: |"]
    # この run が harness ありか bare かで、順位表の主語も種別も変わる
    is_bare = subject != "AHC"
    own_model = ", ".join(overall.get("models") or {}) or "同じモデル"
    ranking = [(f"素のLLM（{own_model}、この run）", "汎用LLM", "なし", n, ours, None, True)
               if is_bare else
               ("AHC（この run）", "ツール利用エージェント", "あり", n, ours, None, True)]
    if has_bare:
        bare_model = ", ".join(other.get("models") or {}) or "同じモデル"
        ranking.append((f"素のLLM（{bare_model}）", "汎用LLM", "なし",
                        other.get("n"), bare, None, True))
    for system in (compared.get("systems") or {}).values():
        ranking.append((system["name"], system["kind_label"],
                        "あり" if system["tools"] else "なし", system["n"],
                        system["accuracy"], system["accuracy_full"], False))
    ranking = [r for r in ranking if r[4] is not None]
    for rank, (name, kind, tools, count, value, full, ours_row) in enumerate(
            sorted(ranking, key=lambda r: -r[4]), 1):
        mark = "**" if ours_row else ""
        lines.append(f"| {rank} | {mark}{name}{mark} | {kind} | {tools} | {count} "
                     f"| {mark}{_fmt(value)}{mark} | {_fmt(full)} |")
    lines += ["",
              "「全 2,788 問での正答率」は各手法がベンチマーク全体で出している値",
              "（この run の部分集合が代表的かを判断するため）。", ""]
    if overall["scored"] < overall["n"]:
        lines += [f"AHC 側は {overall['n']} レコードのうち {overall['scored']} 問が"
                  "採点対象（残りは採点に必要な情報が欠けたもの）。", ""]
    return lines


def _render_breakdowns(block: dict) -> list[str]:
    subject = _subject(block)
    compared = block.get("baselines") or {}
    has_bare = (compared.get("other") or {}).get("accuracy") is not None
    table = block.get("topic_table") or {}
    names = block.get("topic_names") or {}
    lines = ["## 4. 内訳", "", "### 4.1 トピック別", "",
             "ChemBench 本来の集計軸。文献最高はそのトピックの問題を全部解いた",
             "手法の中から選んでいる。", "",
             f"| トピック | n | {_cmp_header(subject, has_bare)} |",
             f"| --- | --: | {_cmp_align(has_bare)} |"]
    for topic in [t for t in TOPICS if t in table] + [
            t for t in sorted(table) if t not in TOPICS]:
        row = table[topic]
        lines.append(f"| {names.get(topic, topic)} | {row['n']} "
                     f"| {_cmp_cells(row['ours'], row['best'], row['best_system'], row['bare'], has_bare)} |")

    lines += ["", "### 4.2 採点方式別", "",
              "選択肢問題は**完全一致**（複数正解の問題で 1 つ外すと 0 点）、",
              "数値問題は**正解の 1% 以内**。harness の寄与が出るとすれば",
              "Bash で計算できる数値問題のほうが大きいはず、という見立ての検証。", "",
              "| 採点方式 | n | 正答率 | 回答率 | 形式妥当率 | 追加指標 |",
              "| --- | --: | --: | --: | --: | --- |"]
    for kind, group in (block.get("metric_kinds") or {}).items():
        metrics = group.get("metrics") or {}
        keep = [k for k in ("hamming", "f1", "mae", "rmse", "exact_str_match")
                if k in metrics]
        extra = ", ".join(f"{k}={_fmt(metrics[k])}" for k in keep) or "-"
        lines.append(f"| {METRIC_KIND_LABEL.get(kind, kind)} | {group['n']} "
                     f"| {_fmt(group['score'])} | {_fmt(group['answered_rate'])} "
                     f"| {_fmt(group['valid_rate'])} | {extra} |")

    lines += ["", "### 4.3 要求能力別", "",
              "公式リポジトリの `classified_questions_leaderboard.csv` の `requires` 列。",
              "Calculation を含む問題でこそ harness（Bash での計算）が効くはず。", "",
              "| 要求能力 | n | 正答率 | 回答率 | 平均試行 | 平均所要(秒) |",
              "| --- | --: | --: | --: | --: | --: |"]
    for key, group in (block.get("requires") or {}).items():
        lines.append(f"| {key or '（未分類）'} | {group['n']} | {_fmt(group['score'])} "
                     f"| {_fmt(group['answered_rate'])} "
                     f"| {_fmt(group['mean_attempts'], 2)} "
                     f"| {_fmt(group['mean_elapsed_sec'], 1)} |")
    lines.append("")

    human = block.get("human_subset")
    if human:
        human_compared = human.get("baselines") or {}
        human_other = human_compared.get("other") or {}
        human_systems = human_compared.get("systems") or {}
        lines += ["### 4.4 人間との比較（human subset）", "",
                  "ChemBench は化学者 20 名にも同じ問題の一部を解かせている",
                  "（`scripts/human_subset.csv`）。この run に含まれるその部分集合だけで",
                  "比べたもの。人間側は参加者の平均。", "",
                  "| 対象 | n | 正答率 |", "| --- | --: | --: |",
                  f"| **{subject}** | {human['overall']['n']} "
                  f"| **{_fmt(human['overall']['score'])}** |"]
        if human_other.get("accuracy") is not None:
            other_name = "AHC" if subject != "AHC" else "素のLLM"
            lines.append(f"| {other_name} | {human_other.get('n')} "
                         f"| {_fmt(human_other['accuracy'])} |")
        for key in ("humans-tool", "humans-notool"):
            if key in human_systems:
                system = human_systems[key]
                lines.append(f"| {system['name']}（{system['note']}） | {system['n']} "
                             f"| {_fmt(system['accuracy'])} |")
        lines.append("")
    return lines


def _render_harness_observations(block: dict) -> list[str]:
    overall = block["overall"]
    return ["## 5. AHC 固有の観測値", "",
            "正答率とは別に、「エージェントとしてタスクを完了できたか」も記録している。",
            "Verifier 合格率は**答えファイルを作れたか**であって、答えの正しさとは別物。", "",
            "| 指標 | 値 |", "| --- | --: |",
            f"| 問題数 | {overall['n']} |",
            f"| 正答率（all_correct の平均） | {_fmt(overall['score'])} |",
            f"| 回答率（答えを取り出せた率） | {_fmt(overall['answered_rate'])} |",
            f"| 形式が妥当だった率 | {_fmt(overall['valid_rate'])} |",
            f"| Verifier 合格率 | {_fmt(overall['harness_passed_rate'])} |",
            f"| 平均試行回数 | {_fmt(overall['mean_attempts'], 2)} |",
            f"| 平均所要時間 | {_fmt(overall['mean_elapsed_sec'], 1)} 秒 |",
            f"| 実行エラー | {overall['error_count']} 件 |",
            f"| 答えの取得元 | {overall['answer_sources']} |",
            f"| 実行モデル | {_models_cell(overall)} |",
            ""]


def _render_notes(block: dict) -> list[str]:
    compared = block.get("baselines") or {}
    source = compared.get("source", {})
    other = compared.get("other") or {}
    lines = ["## 6. 出典と注記", "", "### 6.1 文献値の出典", "",
             f"- {source.get('paper', '?')}（{source.get('arxiv', '')}）",
             f"- マルチモーダル版: {source.get('multimodal_paper', '')}",
             f"- リーダーボード: {source.get('leaderboard', '')}",
             f"- 値の実体: `{source.get('reports_dir', '')}` の問題ごとの "
             "`all_correct`（論文の表の転記ではない）",
             "- メタ情報（表示名・種別）: `benchmarks_chembench/baselines.yaml`", ""]
    for note in source.get("notes", []):
        lines.append(f"- {note}")
    lines += ["", "### 6.2 読むときの注意", "",
              "- **文献値の大半は素の LLM（1 往復・ツールなし）。** AHC はツールつき",
              "  エージェントなので、素の LLM と並べた差には harness の寄与が入る。",
              "  同条件で比べたいときは 3.1 の「ツールを使うエージェント同士」の行と、",
              "  自分で測った素のLLM 列を見る。",
              "- **問題は同一だが n は手法ごとに違う**（refusal・欠測）。順位表は",
              "  全問解いた手法の中から最高を選び、n を併記している。",
              "- 選択肢問題は**部分点なし**（hamming=0 のみ 1 点）。複数正解の問題で",
              "  1 つ取りこぼすと 0 点になるので、f1 も併記している（4.2）。",
              "- 数値問題の許容差は公式と同じ `0.01 × 正解`（相対 1% ではなく絶対量）。",
              "  **正解が 0 以下の問題は原理的に不正解**になるが、文献値も同じ規則で",
              "  採点されているので揃えてある。",
              "- 採点の一致は公開 report で確認済み（`evaluate validate`）。",
              "  公式が記録した正誤と本実装の判定が numeric で 100%、"
              "MCQ で 99.6% 一致する。",
              "- 人間の成績は約 120 問の部分集合のみ（4.4）。全体平均とは比べられない。",
              f"- 素のLLM の実測は別 run `{other.get('label', '-')}`。ツールを持たせない",
              "  設定（`tools=[]`）で、ツール使用を検知したら失敗扱いにしている。",
              "- ChemBench のデータは**評価専用**（学習に使わないこと）。", ""]
    return lines


def render_markdown(summary: dict) -> str:
    providers = summary["providers"]
    multi = len(providers) > 1
    lines = ["# ChemBench 評価レポート — AutoHarnessChem (ahc)", "",
             f"- 対象: `{summary['label']}` / {summary['n_records']} レコード", ""]
    for provider, block in providers.items():
        if multi:
            lines += [f"# provider: {provider}", ""]
        lines += _intro(summary, block)
        lines += _render_coverage(block)
        lines += _render_overview(block)
        lines += _render_breakdowns(block)
        lines += _render_harness_observations(block)
        lines += _render_notes(block)
    return "\n".join(lines)


def write_report(records: list[dict], out_dir: Path, label: str = "chembench",
                 compare_records: list[dict] | None = None,
                 compare_label: str | None = None) -> dict:
    """metrics.json / report.md / scored.jsonl を書く。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records, label, compare_records=compare_records,
                        compare_label=compare_label)
    # 問題ごとの一覧・文献値は問題数ぶん膨らむので metrics.json には残さない
    # （集計値だけを残す。生の正誤は scored.jsonl と公式 report 側にある）
    slim = json.loads(json.dumps(summary, ensure_ascii=False, default=str))
    for block in slim.get("providers", {}).values():
        for compared in (block.get("baselines"), (block.get("human_subset") or {}).get("baselines")):
            if not compared:
                continue
            compared.pop("question_names", None)
            for system in (compared.get("systems") or {}).values():
                system.pop("scores", None)
            if compared.get("other"):
                compared["other"].pop("scores", None)
    (out_dir / "metrics.json").write_text(
        json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    with (out_dir / "scored.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summary


__all__ = ["render_markdown", "summarize", "summarize_group", "write_report"]
