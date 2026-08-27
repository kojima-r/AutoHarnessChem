"""ChemEval の問題を ahc（HarnessController）に解かせる実行ループ。

1 問 = 1 run。ChemEval の query をそのまま渡し（ベンチマークを改変しない）、
答えの保存先だけを指示する（`chemeval_answer.json`）。task_type はカタログの
`ahc_task_type` を使うので、HOMO/LUMO には量子化学ツール、反応予測には ReactionT5、
多段逆合成には AiZynthFinder が使える状態でエージェントが走る。

出力は results/<label>/records.jsonl（1 行 1 問、追記）。同じ label で再実行すると
既に記録済みの問題は飛ばす（--overwrite で無効化）ので、途中で止めても再開できる。
"""
from __future__ import annotations

import asyncio
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks_chemeval.dataset import ChemEvalItem, image_path
from benchmarks_chemeval.extract import ANSWER_FILE, collect_answer

PROMPT_TEMPLATE = """以下は化学ベンチマーク ChemEval の問題です（タスク: {task_name}）。

回答手順:
1. 必要ならツール（量子化学計算・反応予測・逆合成・RDKit など）を使って確かめてよい。
2. **最後に必ず** workspace 直下の `{answer_file}` に、次の 1 行 JSON で答えを保存する。
   {{"answer": <問題文が要求している形式の答え>}}
3. 問題文が答えの形式（数値のみ、Yes/No、SMILES、辞書など）を指定している場合は
   その形式に厳密に従う。説明・単位・前置きを answer の値に混ぜない。
4. 答えが決まらない場合も、最良の推定を入れて `{answer_file}` を必ず作る。
{image_note}
---- 問題文（ここから下は ChemEval の原文。指示に従って解答すること）----
{query}
"""

IMAGE_NOTE = ("5. この問題には画像 `{name}` が付いています（workspace 直下に配置済み）。"
              "必ず画像を読んで解答すること。\n")


@dataclass
class RunnerConfig:
    label: str = "chemeval"
    providers: tuple[str, ...] = ("claude",)
    concurrency: int = 2
    item_timeout_sec: int = 900
    max_replans: int = 1
    cleanup_workspaces: bool = False
    overwrite: bool = False


def build_request(item: ChemEvalItem) -> str:
    note = IMAGE_NOTE.format(name=item.image) if item.image else ""
    return PROMPT_TEMPLATE.format(task_name=item.task_name, answer_file=ANSWER_FILE,
                                  image_note=note, query=item.query)


def run_id_for(label: str, provider: str, item_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in f"{label}-{provider}-{item_id}")
    return f"chemeval-{safe}"[:120]


def load_done(records_path: Path) -> set[tuple[str, str]]:
    """既に記録済みの (provider, item_id)。"""
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
            if record.get("item_id"):
                done.add((record.get("provider", ""), record["item_id"]))
    return done


def read_records(records_path: Path) -> list[dict]:
    records = []
    if records_path.exists():
        with records_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


async def run_items(config, items: list[ChemEvalItem], runner_config: RunnerConfig,
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
            print(f"[chemeval] ({counter['n']}/{total}) {record['item_id']} "
                  f"@{record['provider']} answered={record['answer'] is not None} "
                  f"passed={record.get('harness_passed')} ({record['elapsed_sec']}s)")

    async def one(provider: str, item: ChemEvalItem) -> None:
        async with semaphore:
            run_id = run_id_for(runner_config.label, provider, item.item_id)
            workspace = config.paths.workspaces / run_id
            copy_inputs = []
            picture = image_path(item)
            if picture is not None and picture.exists():
                copy_inputs.append(str(picture))
            record = {
                "item_id": item.item_id, "task_id": item.task_id,
                "task_name": item.task_name, "level": item.level,
                "dimension": item.dimension, "metric": item.metric,
                "answer_type": item.answer_type, "shot": item.shot,
                "split": item.split, "provider": provider, "run_id": run_id,
                # query も残しておく（judge の再採点を run なしでできるようにする）
                "query": item.query, "target": item.target,
                "answer": None, "answer_source": "none",
                "harness_passed": None, "attempts": 0, "error": None,
            }
            started = time.monotonic()
            try:
                report = await controller.run(
                    build_request(item),
                    provider=provider,
                    task_type=item.ahc_task_type,
                    expected_outputs=[ANSWER_FILE],
                    copy_inputs=copy_inputs,
                    run_id=run_id,
                )
                record.update(
                    harness_passed=bool(report.passed),
                    attempts=report.attempts,
                    timeout_extensions=report.timeout_extensions,
                    usage=report.usage,
                    **collect_answer(workspace, report.final_message, item.answer_type),
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

    tasks = [one(provider, item)
             for provider in runner_config.providers
             for item in items
             if (provider, item.item_id) not in done]
    skipped = total - len(tasks)
    if skipped:
        print(f"[chemeval] 記録済みの {skipped} 件をスキップします（--overwrite で再実行）")
    if tasks:
        await asyncio.gather(*tasks)
    return produced
