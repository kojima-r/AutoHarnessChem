"""素の LLM ベースライン（harness を通さず 1 ターンで解かせる）。

ChemBench の公開 report（= 文献値）は**素の LLM に 1 往復で**解かせた結果で、
ツールも検証ループも無い。一方 `runner.py` は「ツール + Verifier + 再計画ループ」で
解かせるので、両者を並べただけではモデル世代の差と harness の寄与が混ざる。
そこで文献と同条件の列を自分で測るためのモードを用意する。

文献の条件に合わせるため、ここでは:

- ChemBench のプロンプトを**原文のまま**投げる（答えの保存先の指示も足さない。
  プロンプト自身が `[ANSWER]...[/ANSWER]` 形式を指定している）
- **ツールを一切持たせない**（`tools=[]`）。`allowed_tools=[]` は「許可リストを
  指定しない」の意味でツールは無効にならないため、ツールセット自体を空にする。
  念のため ToolUseBlock が来たら失敗として扱う
- 1 往復で終える。答えは応答テキストから `extract.collect_answer` で拾う
- 採点は harness 側と**同じ `metrics.py`**（= 公式と同じ規則）を通す

出力は `runner.py` と同じ形の records.jsonl（`mode: "bare"` が入る）。同じ label で
再実行すると記録済みは飛ばすので、利用枠をまたいで再開できる。
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from adapters.base import real_model_name
from benchmarks_chembench.dataset import ChemBenchItem
from benchmarks_chembench.extract import collect_answer
from benchmarks_chembench.runner import (RunnerConfig, balance_pending, base_record,
                                          load_done, read_records, topic_counts)

SYSTEM_PROMPT = ("You are a chemistry expert. Answer the question exactly in the output "
                 "format the question specifies, with no extra explanation.")

# 利用枠切れのとき、SDK は例外ではなく
# `You've hit your session limit · resets 4:50am (Asia/Tokyo)` という**本文**を返す。
# これを答えとして記録すると「回答済みだが不正解」になり、枠切れの検知も再実行も
# できなくなる（ChemEval の bare 実行で 2,210 問中 1,346 問がこれで潰れた実績がある）。
_LIMIT_MESSAGE = re.compile(r"hit your (session|usage|weekly) limit", re.IGNORECASE)


class UsageLimitReached(RuntimeError):
    """利用枠切れ。答えではなく失敗として記録し、枠が戻ってから解き直す。"""


class ToolUseDetected(RuntimeError):
    """素のはずのベースラインでツールが使われた（= ベースラインとして無効）。

    SDK の既定が変わってツールが復活したときに、黙って「素の LLM」を名乗る結果が
    混ざるのを防ぐための番人。
    """


class BareModel:
    """claude-agent-sdk 経由の 1 ターン問い合わせ（ツールなし）。"""

    def __init__(self, model: str | None = None):
        self.model = model
        # --model で固定したならそれが実行モデル。SDK の報告が届く前に終わった問でも
        # モデル名が欠けないよう既定値にしておく（並列実行のため）
        self.reported_model: str | None = real_model_name(model)

    async def ask(self, prompt: str) -> str:
        from claude_agent_sdk import ClaudeAgentOptions, query

        options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            model=self.model,
            tools=[],                  # ツールセットを空にする（文献と同条件）
            allowed_tools=[],
            # 思考だけで数ターン使う問題があるため余裕を持たせる。ツールが無い以上
            # 往復が増えるだけで、外部の計算に頼ることはない
            max_turns=6,
        )
        text = ""
        try:
            async for message in query(prompt=prompt, options=options):
                reported = real_model_name(getattr(message, "model", None))
                if reported:
                    self.reported_model = reported
                kind = type(message).__name__
                if kind == "AssistantMessage":
                    for block in getattr(message, "content", []) or []:
                        block_kind = type(block).__name__
                        if block_kind == "ToolUseBlock":
                            raise ToolUseDetected(str(getattr(block, "name", "")))
                        if block_kind == "TextBlock":
                            text += getattr(block, "text", "")
                elif kind == "ResultMessage":
                    text = getattr(message, "result", "") or text
        except ToolUseDetected:
            raise                      # ツールを使ったなら答えの中身によらず無効
        except Exception:
            # ターン上限などで打ち切られても、受け取った答えは捨てない
            if not text:
                raise
        if _LIMIT_MESSAGE.search(text):
            raise UsageLimitReached(text.strip()[:200])
        return text


async def run_items_bare(items: list[ChemBenchItem], runner_config: RunnerConfig,
                         records_path: Path, model: str | None = None,
                         asker: BareModel | None = None) -> list[dict]:
    """items を素の LLM に解かせ、records.jsonl へ追記しながら結果を返す。

    `asker` を渡すとそれを使う（テストではスタブを渡す）。
    """
    records_path = Path(records_path)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    if runner_config.overwrite and records_path.exists():
        records_path.unlink()
    done = load_done(records_path)
    asker = asker or BareModel(model)

    semaphore = asyncio.Semaphore(max(1, runner_config.concurrency))
    write_lock = asyncio.Lock()
    produced: list[dict] = []
    counter = {"n": 0}
    # harness 側と同じ順序規則（構成比の不足順）で回す。途中で止めても
    # 両者の部分集合が同じ母集団の縮小版になり、比較がずれない
    already = topic_counts(read_records(records_path))
    targets = [(provider, item) for provider in runner_config.providers
               for item in balance_pending(
                   [i for i in items if (provider, i.question_name) not in done], already)]
    total = len(targets)

    async def write(record: dict) -> None:
        async with write_lock:
            with records_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            produced.append(record)
            counter["n"] += 1
            print(f"[chembench:bare] ({counter['n']}/{total}) {record['question_name']} "
                  f"@{record['provider']} answered={record['answer'] is not None} "
                  f"({record['elapsed_sec']}s)")

    async def one(provider: str, item: ChemBenchItem) -> None:
        async with semaphore:
            record = base_record(item, provider, f"bare-{provider}-{item.question_name}")
            # harness を通していないので Verifier も再計画も無い
            record.update(mode="bare", attempts=1)
            started = time.monotonic()
            try:
                text = await asyncio.wait_for(asker.ask(item.prompt),
                                              timeout=runner_config.item_timeout_sec)
                record.update(model=asker.reported_model,
                              **collect_answer(None, text, item.metric_kind))
            except asyncio.TimeoutError:
                record.update(error="timeout")
            except asyncio.CancelledError:
                raise
            except Exception as e:                    # 1 問の失敗で全体を止めない
                record.update(error=f"{type(e).__name__}: {e}")
            record["elapsed_sec"] = round(time.monotonic() - started, 2)
            await write(record)

    skipped = len(items) * len(runner_config.providers) - total
    if skipped:
        print(f"[chembench:bare] 記録済みの {skipped} 件をスキップします（--overwrite で再実行）")
    if targets:
        await asyncio.gather(*(one(provider, item) for provider, item in targets))
    return produced
