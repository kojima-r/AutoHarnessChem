"""エージェントの出力から ChemEval の答え（`{"answer": ...}`）を取り出す。

ChemEval の query 自身が `{"answer": "..."}` 形式での出力を指示しているため、
基本はその JSON を拾うだけでよい。ただし ahc はエージェント実行なので、答えは

  1. workspace の `chemeval_answer.json`（この評価スクリプトが指示する保存先）
  2. 最終メッセージ中の JSON（コードフェンス内 / 素の `{...}`）
  3. それも無ければ最終メッセージ全文

のいずれかにある。ここでは 1 → 2 → 3 の順に探し、どこから取れたかを
`source` として記録する（採点結果の信頼度を後から追えるようにする）。
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

ANSWER_FILE = "chemeval_answer.json"


def loads_loose(text: str) -> Any:
    """JSON → だめなら python リテラル（ChemEval の gold も dict の repr 形式）。"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def balanced_json_blocks(text: str) -> list[str]:
    """`{` … `}` の対応が取れた部分文字列をすべて返す（出現順）。"""
    blocks: list[str] = []
    depth = 0
    start = -1
    in_string = False
    quote = ""
    escaped = False
    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in "\"'":
            in_string, quote = True, char
            continue
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                blocks.append(text[start:i + 1])
                start = -1
    return blocks


def _answer_from_obj(obj: Any) -> tuple[Any, bool]:
    """dict なら `answer` 相当のキーを取り出す。"""
    if isinstance(obj, dict):
        for key in obj:
            if isinstance(key, str) and key.strip().strip('"').lower() == "answer":
                return obj[key], True
        return obj, False
    return obj, False


def extract_from_text(text: str) -> tuple[Any, bool]:
    """テキスト中の JSON から答えを取り出す。後方の JSON を優先する。"""
    if not text:
        return None, False
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    candidates = [*balanced_json_blocks(text), *fenced]
    for block in reversed(candidates):
        parsed = loads_loose(block.strip())
        if parsed is None:
            continue
        answer, found = _answer_from_obj(parsed)
        if found:
            return answer, True
    # `answer: xxx` / `最終回答: xxx` のような素の書き方
    match = re.search(r'"?answer"?\s*[:=]\s*"?([^"\n}]+)"?', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(), True
    return None, False


def read_answer_file(workspace: Path) -> tuple[Any, bool]:
    """workspace の `chemeval_answer.json` を読む（サブディレクトリも探す）。"""
    workspace = Path(workspace)
    paths = [workspace / ANSWER_FILE, *sorted(workspace.rglob(ANSWER_FILE))]
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        parsed = loads_loose(text)
        if parsed is None:
            answer, found = extract_from_text(text)
            if found:
                return answer, True
            return text or None, bool(text)
        answer, found = _answer_from_obj(parsed)
        # {"answer": ...} でなくても、ファイルがある以上その中身が答え
        return answer, True if found else parsed is not None
    return None, False


def normalize(answer: Any, answer_type: str) -> Any:
    """answer_type に合わせて整形する（採点側の前提を揃える）。"""
    if answer is None:
        return None
    if answer_type == "dict":
        if isinstance(answer, dict):
            return answer
        parsed = loads_loose(str(answer).strip()) if isinstance(answer, str) else None
        return parsed if isinstance(parsed, dict) else answer
    if answer_type == "list":
        if isinstance(answer, list):
            return [str(a) for a in answer]
        if isinstance(answer, str):
            parsed = loads_loose(answer.strip())
            if isinstance(parsed, list):
                return [str(a) for a in parsed]
        return answer if isinstance(answer, (int, float)) else str(answer)
    if isinstance(answer, list):
        return ", ".join(str(a) for a in answer)
    if isinstance(answer, dict):
        return json.dumps(answer, ensure_ascii=False)
    return answer


def collect_answer(workspace: Path | None, final_message: str,
                   answer_type: str = "text") -> dict:
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
            source = "final_message_json"
        elif (final_message or "").strip():
            answer, source = final_message.strip(), "final_message_text"
    return {
        "answer": normalize(answer, answer_type),
        "answer_source": source,
        "answer_raw": None if answer is None else str(answer)[:4000],
    }
