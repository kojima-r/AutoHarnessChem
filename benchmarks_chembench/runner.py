"""ChemBench の問題を ahc（HarnessController）に解かせる実行ループ。

1 問 = 1 run。ChemBench のプロンプトを**原文のまま**渡し（ベンチマークを改変しない）、
答えの保存先だけを足す（`chembench_answer.json`）。task_type はカタログの
`ahc_task_type`（既定は `generic`）を使う。ツールは `build_default_registry` が
task_type に関係なく全部登録するので、エージェントは Bash・RDKit・量子化学計算を
必要に応じて使える。

出力は results/<label>/records.jsonl（1 行 1 問、追記）。採点に必要な情報
（`score_map` / `target` / `tolerance`）も record に入れるので、**run を回さずに
採点だけやり直せる**。同じ label で再実行すると記録済みの問題は飛ばす
（--overwrite で無効化）ので、途中で止めても再開できる。
"""
from __future__ import annotations

import asyncio
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks_chembench.dataset import ChemBenchItem
from benchmarks_chembench.extract import ANSWER_FILE, collect_answer

PROMPT_TEMPLATE = """以下は化学ベンチマーク ChemBench の問題です（トピック: {topic_name}）。

回答手順:
1. 必要ならツール（Bash での数値計算・RDKit・量子化学計算など）で確かめてよい。
2. **最後に必ず** workspace 直下の `{answer_file}` に、次の 1 行 JSON で答えを保存する。
   {{"answer": "<答え>"}}
   - 選択肢問題: 選んだ選択肢の**文字だけ**を入れる（例 {{"answer": "C"}}、
     複数選ぶ問題なら {{"answer": "A,C"}}）。選択肢の本文や説明は入れない。
   - 数値問題: **数値だけ**を入れる（単位・記号・桁区切り・説明を入れない。
     例 {{"answer": "42292.49"}}）。小数点はドット。
3. 最終メッセージにも、問題文の指示どおり `[ANSWER]...[/ANSWER]` の形で答えを書く。
4. 答えが決まらない場合も、最良の推定を入れて `{answer_file}` を必ず作る。

---- 問題文（ここから下は ChemBench の原文。指示に従って解答すること）----
{prompt}
"""


@dataclass
class RunnerConfig:
    label: str = "chembench"
    providers: tuple[str, ...] = ("claude",)
    concurrency: int = 2
    item_timeout_sec: int = 600
    max_replans: int = 1
    cleanup_workspaces: bool = False
    overwrite: bool = False


def build_request(item: ChemBenchItem) -> str:
    return PROMPT_TEMPLATE.format(topic_name=item.topic_name, answer_file=ANSWER_FILE,
                                  prompt=item.prompt)


def run_id_for(label: str, provider: str, question_name: str) -> str:
    raw = f"{label}-{provider}-{question_name}"
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in raw)
    return f"chembench-{safe}"[:120]


def base_record(item: ChemBenchItem, provider: str, run_id: str) -> dict:
    """採点とレポートに必要な情報を全部持つ record の雛形。

    `score_map` / `target` / `tolerance` を入れておくのが肝で、これがあるので
    records.jsonl だけで採点をやり直せる（出題元の report を読み直さなくてよい）。
    """
    return {
        "question_name": item.question_name,
        "topic": item.topic, "topic_name": item.topic_name,
        "requires": item.requires, "difficulty": item.difficulty,
        "metric_kind": item.metric_kind, "ahc_task_type": item.ahc_task_type,
        "in_human_subset": item.in_human_subset,
        "score_map": item.score_map or None,
        "target": item.target, "tolerance": item.tolerance,
        "n_options": len(item.options) or None,
        "n_correct_options": len(item.correct_letters) or None,
        "provider": provider, "run_id": run_id,
        "answer": None, "answer_source": "none",
        # SDK が実際に使ったモデル。文献値と並べるので「どのモデルの成績か」は必須
        "model": None,
        "harness_passed": None, "attempts": 0, "error": None,
    }


def load_done(records_path: Path) -> set[tuple[str, str]]:
    """既に記録済みの (provider, question_name)。"""
    done: set[tuple[str, str]] = set()
    if not records_path.exists():
        return done
    with records_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("question_name"):
                done.add((record.get("provider", ""), record["question_name"]))
    return done


def read_records(records_path: Path) -> list[dict]:
    records = []
    if Path(records_path).exists():
        with Path(records_path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


PURGE_COUNTS_FILE = "purge_counts.json"


#: これより短く終わった未回答の試行は「実際には解かせていない」とみなす。
#: 利用枠が切れると SDK は即エラーを返し、1 問 2 秒・0 ターンで未回答になる
#: （枠切れの瞬間に残り全問がこの形で記録される。実測: 残り 2,557 問が一斉に潰れた）。
#: これを再試行回数に数えると、枠切れが数回起きただけで全問が「確定失敗」になる。
NOT_ATTEMPTED_SEC = 15.0


def purge_failed(records_path: Path, *, max_retries: int = 3,
                 not_attempted_sec: float = NOT_ATTEMPTED_SEC,
                 not_attempted_cap: int = 12) -> dict:
    """答えが取れていない record を落として、再開時の実行対象に戻す。

    全 2,788 問を回すと必ず利用枠をまたぐ。枠切れやタイムアウトで失敗した record を
    残したまま再開すると `load_done` が「実行済み」と見なして**永久に飛ばす**ので、
    再開前にこれを通す必要がある（ChemEval の運用で実際に踏んだ罠）。

    判定は「エラー文言」ではなく **answer が無いこと**で行う。SDK は枠切れを例外では
    なく本文で返すことがあり、文言での判定は当てにならないため。

    同じ問題を無限に回さないよう、落とした回数を `purge_counts.json` に持ち越し、
    `max_retries` を超えたものは**確定失敗として残す**（0 点として集計に入る）。
    ただし **`not_attempted_sec` 未満で未回答に終わった試行は通常の回数に数えない**
    （枠切れで一斉に潰れた分まで数えると、数回の枠切れで全問が確定失敗になる）。

    その代わり「速い失敗」も別カウンタで数え、`not_attempted_cap` に達したら
    諦める。**速い拒否と速い枠切れは見分けが付かない**ためで、これが無いと
    モデルが安全上の理由で即座に断る問題（実例: TATP の合成に使う薬品を問う設問）が
    永久に再実行され続けてループが終わらない。上限は枠切れの回数より十分大きく取る。
    """
    records_path = Path(records_path)
    if not records_path.exists():
        return {"total": 0, "purged": 0, "given_up": 0, "kept": 0,
                "not_attempted": 0}

    counts_path = records_path.parent / PURGE_COUNTS_FILE
    counts: dict[str, int] = {}
    fast: dict[str, int] = {}
    if counts_path.exists():
        try:
            stored = json.loads(counts_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stored = {}
        if isinstance(stored, dict) and "retries" in stored:
            counts = dict(stored.get("retries") or {})
            fast = dict(stored.get("fast") or {})
        else:                       # 旧形式（{name: 回数} のフラットな dict）
            counts = {k: int(v) for k, v in (stored or {}).items()}

    records = read_records(records_path)
    kept: list[dict] = []
    purged = given_up = not_attempted = 0
    for record in records:
        name = record.get("question_name") or ""
        answered = record.get("answer") is not None
        if answered:
            kept.append(record)
            continue
        elapsed = record.get("elapsed_sec")
        if elapsed is not None and float(elapsed) < not_attempted_sec:
            # 枠切れ等で実際には解かせていない。通常の再試行回数には数えない。
            # ただし「速い拒否」と区別が付かないので別カウンタで上限を設ける
            seen_fast = int(fast.get(name, 0))
            if seen_fast >= not_attempted_cap:
                record["purge_given_up"] = seen_fast
                record["purge_given_up_reason"] = "not_attempted_cap"
                kept.append(record)
                given_up += 1
                continue
            fast[name] = seen_fast + 1
            not_attempted += 1
            purged += 1
            continue
        seen = int(counts.get(name, 0))
        if seen >= max_retries:
            # 何度やっても答えが出ない問題。確定失敗として残し、前へ進める
            record["purge_given_up"] = seen
            kept.append(record)
            given_up += 1
            continue
        counts[name] = seen + 1
        purged += 1

    # 諦めた record には印を付けるので、purged が 0 でも書き戻す必要がある
    if purged or given_up:
        with records_path.open("w", encoding="utf-8") as fh:
            for record in kept:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    if purged:                          # どちらかのカウンタを増やしたら書き戻す
        counts_path.write_text(
            json.dumps({"retries": counts, "fast": fast}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    return {"total": len(records), "purged": purged, "given_up": given_up,
            "kept": len(kept), "not_attempted": not_attempted}


def balance_pending(items: list[ChemBenchItem],
                    done_topics: dict[str, int]) -> list[ChemBenchItem]:
    """未実行の items を、トピックの構成比の**不足が大きい順**に並べ替える。

    全件（2,788 問）は利用枠をまたぐので、必ず途中経過でレポートを出すことになる。
    そのとき部分集合が母集団の縮小版になっていないと、正答率を公表値と比べられない。

    `dataset.load_items` は全件のとき層化順序で返すが、**再開時には既に実行済みの
    偏り**（前回どこまで進んだか）を埋め合わせる必要がある。そこで「目標の構成比に
    対していま何問足りないか」を毎回見て、最も足りていないトピックから 1 問ずつ
    取り出す（貪欲法）。目標の構成比は `items` 自身（= 今回の対象全体）から取るので、
    トピックを絞った run でもそのまま使える。

    並べ替えても再開判定は question_name で行うため、結果には影響しない。
    """
    if not items:
        return []
    by_topic: dict[str, list[ChemBenchItem]] = {}
    for item in items:
        by_topic.setdefault(item.topic, []).append(item)

    # 目標の構成比は **run 全体**（既に実行済み + 未実行）から取る。未実行だけで
    # 割合を出すと、前回の偏りがそのまま目標に化けてしまう（例: analytical_chemistry を
    # 全部終えた状態で再開すると、そのトピックが目標から消えて偏りが固定される）。
    # 実行済みトピックも母数に入れるため、done_topics 側のキーも拾う。
    target = {topic: len(group) for topic, group in by_topic.items()}
    for topic, count in done_topics.items():
        target[topic] = target.get(topic, 0) + int(count)
    total_target = sum(target.values()) or 1
    placed = {topic: int(done_topics.get(topic, 0)) for topic in target}

    ordered: list[ChemBenchItem] = []
    queues = {topic: list(group) for topic, group in by_topic.items()}
    while any(queues.values()):
        total_placed = sum(placed.values()) or 1
        # 「目標の割合 − 現在の割合」が最大のトピックから 1 問ずつ取る
        topic = max((t for t, q in queues.items() if q),
                    key=lambda t: (target[t] / total_target - placed[t] / total_placed,
                                   target[t]))
        ordered.append(queues[topic].pop(0))
        placed[topic] += 1
    return ordered


def topic_counts(records: list[dict]) -> dict[str, int]:
    """記録済みレコードのトピック別件数（再開時の不足計算に使う）。"""
    counts: dict[str, int] = {}
    for record in records:
        topic = record.get("topic")
        if topic:
            counts[str(topic)] = counts.get(str(topic), 0) + 1
    return counts


async def run_items(config, items: list[ChemBenchItem], runner_config: RunnerConfig,
                    records_path: Path, controller=None) -> list[dict]:
    """items を各 provider で実行し、records.jsonl へ追記しながら結果を返す。

    controller を渡すとそれを使う（テストではスタブを渡す）。
    """
    from adapters import AdapterUnavailable
    from harness.controller import HarnessController

    # 1 問ごとに締切を設け、時間切れは延長せず打ち切る（評価が止まらないように）
    config.runtime.attempt_timeout_sec = runner_config.item_timeout_sec
    config.runtime.on_attempt_timeout = "stop"
    config.runtime.max_replans = runner_config.max_replans

    records_path = Path(records_path)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    if runner_config.overwrite and records_path.exists():
        records_path.unlink()
    done = load_done(records_path)

    controller = controller or HarnessController(config)
    semaphore = asyncio.Semaphore(max(1, runner_config.concurrency))
    write_lock = asyncio.Lock()
    produced: list[dict] = []
    total = len(items) * len(runner_config.providers)
    counter = {"n": 0}

    async def write(record: dict) -> None:
        async with write_lock:
            with records_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            produced.append(record)
            counter["n"] += 1
            print(f"[chembench] ({counter['n']}/{total}) {record['question_name']} "
                  f"@{record['provider']} answered={record['answer'] is not None} "
                  f"passed={record.get('harness_passed')} ({record['elapsed_sec']}s)")

    async def one(provider: str, item: ChemBenchItem) -> None:
        async with semaphore:
            run_id = run_id_for(runner_config.label, provider, item.question_name)
            workspace = config.paths.workspaces / run_id
            record = base_record(item, provider, run_id)
            started = time.monotonic()
            try:
                report = await controller.run(
                    build_request(item),
                    provider=provider,
                    task_type=item.ahc_task_type,
                    expected_outputs=[ANSWER_FILE],
                    run_id=run_id,
                )
                record.update(
                    model=getattr(report, "model", None),
                    harness_passed=bool(report.passed),
                    attempts=report.attempts,
                    timeout_extensions=report.timeout_extensions,
                    usage=report.usage,
                    **collect_answer(workspace, report.final_message, item.metric_kind),
                )
            except AdapterUnavailable as e:
                record.update(error=str(e), skipped=True)
            except asyncio.CancelledError:
                raise
            except Exception as e:                    # 1 問の失敗で全体を止めない
                record.update(error=f"{type(e).__name__}: {e}")
            record["elapsed_sec"] = round(time.monotonic() - started, 2)
            await write(record)
            if runner_config.cleanup_workspaces and workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)

    # 途中経過のレポートを母集団の縮小版に保つため、実行順を構成比の不足順にする
    already = topic_counts(read_records(records_path))
    tasks = [one(provider, item)
             for provider in runner_config.providers
             for item in balance_pending(
                 [i for i in items if (provider, i.question_name) not in done], already)]
    skipped = total - len(tasks)
    if skipped:
        print(f"[chembench] 記録済みの {skipped} 件をスキップします（--overwrite で再実行）")
    if tasks:
        await asyncio.gather(*tasks)
    return produced
