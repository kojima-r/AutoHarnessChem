"""エージェントの出力から ChemBench の答えを取り出す。

ChemBench のプロンプトは答えを `[ANSWER]...[/ANSWER]` で囲むよう指示していて、
文献の各モデルの答えもその正規表現で拾われている。ahc はエージェント実行なので、
答えは

  1. workspace の `chembench_answer.json`（この評価スクリプトが指示する保存先）
  2. 最終メッセージ中の `[ANSWER]...[/ANSWER]`
  3. それも無ければ最終メッセージ全文

のいずれかにある。1 → 2 → 3 の順に探し、どこから取れたかを `answer_source` に残す。

**拾った後の解釈は `metrics.py` に任せる**（公式と同じ正規表現で選択肢の文字 /
数値にする）。ここで「`C. 選択肢名` の先頭文字を取る」ような救済はしない
―― 文献値は救済なしで採点されているので、こちら側だけ甘くすると比較が崩れる。
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

ANSWER_FILE = "chembench_answer.json"

_ANSWER_TAG = re.compile(r"\[ANSWER\](.*?)\[/?ANSWER\]", re.DOTALL)


def loads_loose(text: str) -> Any:
    """JSON → だめなら python リテラルとして読む。"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def _answer_from_obj(obj: Any) -> tuple[Any, bool]:
    """dict なら `answer` 相当のキーを取り出す。"""
    if isinstance(obj, dict):
        for key in obj:
            if isinstance(key, str) and key.strip().strip('"').lower() == "answer":
                return obj[key], True
        return obj, False
    return obj, False


def read_answer_file(workspace: Path) -> tuple[Any, bool]:
    """workspace の `chembench_answer.json` を読む（サブディレクトリも探す）。"""
    workspace = Path(workspace)
    paths = [workspace / ANSWER_FILE, *sorted(workspace.rglob(ANSWER_FILE))]
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        parsed = loads_loose(text)
        if parsed is None:
            return text, True            # JSON として壊れていても中身は答えとして使う
        answer, _ = _answer_from_obj(parsed)
        return answer, True
    return None, False


def extract_from_text(text: str) -> tuple[Any, bool]:
    """最終メッセージから `[ANSWER]...[/ANSWER]` を取り出す（後方を優先）。"""
    if not text:
        return None, False
    matches = _ANSWER_TAG.findall(text)
    if matches:
        return matches[-1].strip(), True
    return None, False


def normalize(answer: Any, metric_kind: str) -> Any:
    """採点側の前提（文字列 or 数値）に合わせて整形する。"""
    if answer is None:
        return None
    if isinstance(answer, bool):
        return str(answer)
    if isinstance(answer, (int, float)):
        return answer
    if isinstance(answer, (list, tuple)):
        # MCQ で ["A", "B"] と書かれた場合。numeric では先頭だけを使う
        values = [str(a).strip() for a in answer if str(a).strip()]
        if metric_kind == "numeric":
            return values[0] if values else None
        return ", ".join(values)
    if isinstance(answer, dict):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer).strip()


def collect_answer(workspace: Path | None, final_message: str,
                   metric_kind: str = "mcq") -> dict:
    """答え・取得元・生テキストをまとめて返す。"""
    answer: Any = None
    source = "none"
    if workspace is not None:
        answer, found = read_answer_file(workspace)
        if found:
            source = ANSWER_FILE
    if source == "none":
        answer, found = extract_from_text(final_message or "")
        if found:
            source = "answer_tag"
        elif (final_message or "").strip():
            answer, source = final_message.strip(), "final_message_text"
    return {
        "answer": normalize(answer, metric_kind),
        "answer_source": source,
        "answer_raw": None if answer is None else str(answer)[:4000],
    }


__all__ = ["ANSWER_FILE", "collect_answer", "extract_from_text", "loads_loose",
           "normalize", "read_answer_file"]
