"""ChemBench の採点を純 python で再実装したもの。

公式実装との対応（`benchmarks_chembench/chembench/src/chembench/`）:

| ここ | 公式 |
| --- | --- |
| `MCQ_REGEX` / `FLOATQ_REGEX` / `NUM_REGEX` | `constant.py` の同名定数（そのまま） |
| `prepare_mcq_answer` | `prompter.py:prepare_mcq_answer`（LLM 抽出の分岐を除く） |
| `extract_mcq_letters` | `prompter.py` の `run_regex(create_multiple_choice_regex(...))` |
| `parse_number` | `utils.py:find_numbers` + `convert_to_number`（pint を使わず純 python） |
| `classification_scores` | `metrics.py:classification_scores` |
| `score_item` の `score` | `prompter.py:_calculate_metrics` の `all_correct` |

採点の要点（**文献値もこの規則で計算されている**ので、緩めず厳密に踏襲する）:

- **MCQ**: `all_correct = (hamming == 0)`。hamming は `(取りこぼし + 余計) / 正解数` なので、
  選んだ文字の集合が正解の集合と**完全一致**したときだけ 1 点。部分点は無い
  （部分的な出来は `f1` / `multiple_choice_grade` に出る）。得点 0.5 の選択肢は
  「正解」に数えない（公式が `v == 1` で判定しているため）。
- **numeric**: `all_correct = (|答え − 正解| < 0.01 * 正解)`。相対 1% ではなく
  「正解の 1% を絶対量として比較」で、正解が 0 以下だと許容差も 0 以下になり
  原理的に不正解になる。**公式の挙動なのでそのまま**（README に注記）。
- 答えが取れなかった問題は 0 点（`answered=False`）。集計から外さない
  ―― 「形式どおりに答えられない」ことも能力の一部として測る、というのが
  ChemBench の立場（公式も refusal を 0 点として数える）。

公式は正規表現で拾えないとき LLM に答えを抽出させる経路を持つ（`llm_extraction_count`）。
ここでは使わない代わりに、ahc には `chembench_answer.json` へ書かせて確実に拾う
（`extract.py`）。どちらも「モデルの答えを取りこぼさない」ための仕掛けで、
どこから拾ったかは `answer_source` に残す。
"""
from __future__ import annotations

import math
import re

# --- 公式 constant.py からそのまま持ってきた正規表現 --------------------
MCQ_REGEX = (r"(?:\[ANSWER\]|\[ANS\]|<ANS>|<ANSWER>|<ans>)\s*"
             r"([A-Z](?:,\s*[A-Z])*)\s*(?:\..*?|,.*?)?"
             r"(?:\[/?ANSWER\]|\[/ANS\]|</ANS>|</ANSWER>|</ans>)")
FLOATQ_REGEX = r"\[ANSWER\][\s\n]*(.*?)[\s\n]*\[/?ANSWER\]"
NUM_REGEX = (r"(?<!\w)([+-]?[0-9]*\.?[0-9]+\s?)?([\^]([-+]?[0-9]+))?"
             r"(?:(?:[eE]|exp|EXP)[-+]?[0-9]+)?"
             r"(?:([*x]|\\times|\\u00d7)\s?[0-9]+[\^]([-+]?[0-9]+|{([-+]?[0-9]+)}))?(?!\w)")

ALPHABET = tuple(chr(i) for i in range(ord("A"), ord("Z") + 1))

METRIC_KINDS = ("mcq", "numeric")

# レポートで「この採点方式の主指標は何か」を出すため
METRIC_INFO = {
    "mcq": {"primary": "hamming", "label": "選択肢の完全一致（hamming=0）",
            "extra": ("f1", "precision", "recall", "multiple_choice_grade")},
    "numeric": {"primary": "mae", "label": "数値が正解の 1% 以内",
                "extra": ("mae", "mse", "exact_str_match")},
}


def _first_nonempty(pattern: str, text: str) -> str | None:
    """`run_regex_iterator(..., return_first=True)` と同じ（空マッチは飛ばす）。"""
    for match in re.finditer(pattern, text):
        if match.group(0) != "":
            return match.group(0)
    return None


# --- 数値の取り出し -----------------------------------------------------

def convert_to_number(text: str) -> float | None:
    """`utils.convert_to_number` の純 python 版（pint を使わない）。

    公式は pint で単位つきの式も解釈するが、`NUM_REGEX` が先に数値トークンだけを
    切り出すため、実際に必要なのは「素の数値」「指数表記」「a × 10^b」の 3 形だけ。
    公開 report の numeric 244 問 × 5 モデルで公式の採点結果と完全一致することを
    確認済み（`evaluate validate` で再現できる）。
    """
    value = str(text).replace("x", "*").replace("\\times", "*").replace("\\u00d7", "*")
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass
    # 1.5 * 10^-3 / 10^{5} など
    match = re.fullmatch(r"([+-]?[0-9]*\.?[0-9]+)?\s*\*?\s*10\s*\^\s*\{?([+-]?[0-9]+)\}?",
                         value)
    if match:
        mantissa = float(match.group(1)) if match.group(1) else 1.0
        return mantissa * 10 ** int(match.group(2))
    # e-10 のように指数だけ書かれている場合（公式も "1" を足して読む）
    match = re.fullmatch(r"([+-]?[0-9]*\.?[0-9]+)?\s*[eE]([+-]?[0-9]+)", value)
    if match:
        mantissa = float(match.group(1)) if match.group(1) else 1.0
        return mantissa * 10 ** int(match.group(2))
    return None


def parse_number(text: str) -> float | None:
    """`utils.find_numbers` 相当（NUM_REGEX の最初の非空マッチを数値にする）。"""
    if not text:
        return None
    match = _first_nonempty(NUM_REGEX, str(text))
    return convert_to_number(match) if match is not None else None


def extract_number_answer(completion: str) -> float | None:
    """`prompter.prepare_general_answer` 相当（`[ANSWER]..[/ANSWER]` 優先）。"""
    if not completion:
        return None
    tagged = _first_nonempty(FLOATQ_REGEX, str(completion))
    if tagged is not None:
        return parse_number(tagged)
    return None


# --- 選択肢の取り出し ---------------------------------------------------

def prepare_mcq_answer(text: str, alphabet: tuple[str, ...] = ALPHABET) -> str | None:
    """`prompter.prepare_mcq_answer` 相当（LLM 抽出の分岐は持たない）。"""
    text = "" if text is None else str(text)
    matches = sorted(set(re.findall(MCQ_REGEX, text, re.DOTALL)))
    if matches:
        return str(matches)
    if text in alphabet:
        return text
    for separator in (",", " "):
        if separator in text:
            parts = text.split(separator)
            letters = [p for p in parts if p.strip() in alphabet]
            return separator.join(letters) if len(letters) == len(parts) else ""
    return None


def extract_mcq_letters(completion: str, letters) -> list[str] | None:
    """答えのテキスト → 選んだ選択肢の文字（公式と同じ 2 段構え）。"""
    prepared = prepare_mcq_answer(completion)
    if not prepared:
        return None
    pattern = "(?:" + "|".join(re.escape(letter) for letter in letters) + ")"
    found = re.findall(pattern, prepared, re.IGNORECASE)
    return [f.upper() for f in found] or None


# --- 指標 ---------------------------------------------------------------

def classification_scores(score_map: dict[str, float], found) -> dict[str, float]:
    """`metrics.classification_scores` と同一。"""
    expected = {k for k, v in score_map.items() if v == 1}
    found_set = set(found or [])
    correct = len(expected & found_set)
    extra = len(found_set - expected)
    missed = len(expected - found_set)
    precision = correct / (correct + extra) if (correct + extra) else 0.0
    recall = correct / (correct + missed) if (correct + missed) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "correct_classes": float(correct), "incorrect_classes": float(extra),
        "missed_classes": float(missed), "extra_classes": float(extra),
        "precision": precision, "recall": recall, "f1": f1,
        # 正解が 1 つも無い問題は dataset 側で落としているので 0 除算は起きない
        "hamming": (missed + extra) / len(expected),
    }


def multiple_choice_grade(score_map: dict[str, float], found) -> float:
    """`metrics.multiple_choice_grade`（選んだ選択肢の得点の合計）。"""
    return float(sum(score_map.get(letter, 0.0) for letter in (found or [])))


def _round(value, digits: int = 6):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)


def score_mcq(score_map: dict[str, float], answer) -> dict:
    """選択肢問題の採点。`answer` は文字列（生の答え）か文字のリスト。"""
    letters = list(score_map)
    if isinstance(answer, (list, tuple)):
        found = [str(a).strip().upper() for a in answer if str(a).strip()]
        found = [f for f in found if f in score_map] or None
    else:
        found = extract_mcq_letters("" if answer is None else str(answer), letters)
    scores = classification_scores(score_map, found)
    grade = multiple_choice_grade(score_map, found)
    return {
        "score": float(scores["hamming"] == 0),
        "answered": found is not None,
        # 許された文字だけを選べているか（形式の妥当性）
        "valid": bool(found) and all(f in score_map for f in found),
        "metrics": {k: _round(v) for k, v in scores.items()} | {
            "multiple_choice_grade": _round(grade)},
        "detail": {"parsed": found, "expected": sorted(
            k for k, v in score_map.items() if v == 1)},
    }


def score_numeric(target: float, tolerance: float | None, answer) -> dict:
    """数値問題の採点（`all_correct = mae < tolerance`）。"""
    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        found = float(answer)
    else:
        text = "" if answer is None else str(answer)
        # `[ANSWER]..[/ANSWER]` が無い場合（答えファイル由来など）は素の数値として読む
        found = extract_number_answer(text)
        if found is None:
            found = parse_number(text)
    if tolerance is None:
        tolerance = 0.01 * float(target)
    if found is None:
        return {"score": 0.0, "answered": False, "valid": False,
                "metrics": {"mae": None, "mse": None, "exact_str_match": 0.0},
                "detail": {"parsed": None, "expected": target, "tolerance": tolerance}}
    absolute_error = abs(found - float(target))
    return {
        "score": float(absolute_error < tolerance),
        "answered": True, "valid": True,
        "metrics": {"mae": _round(absolute_error), "mse": _round(absolute_error ** 2),
                    "exact_str_match": float(str(found).strip() == str(target).strip())},
        "detail": {"parsed": found, "expected": float(target), "tolerance": tolerance},
    }


def score_item(record: dict, answer=None) -> dict:
    """record（`metric_kind` / `score_map` / `target`）と答えから採点結果を返す。

    `answer` を省略すると `record["answer"]` を使う。返る dict は
    `score`（0/1 = all_correct） / `answered` / `valid` / `metrics` / `detail`。
    """
    if answer is None:
        answer = record.get("answer")
    kind = record.get("metric_kind")
    if kind == "mcq":
        score_map = {str(k): float(v) for k, v in (record.get("score_map") or {}).items()}
        if not score_map:
            raise ValueError(f"{record.get('question_name')}: score_map がありません")
        return score_mcq(score_map, answer)
    if kind == "numeric":
        target = record.get("target")
        if target is None:
            raise ValueError(f"{record.get('question_name')}: target がありません")
        return score_numeric(float(target), record.get("tolerance"), answer)
    raise ValueError(f"未知の metric_kind: {kind!r}")


__all__ = ["ALPHABET", "FLOATQ_REGEX", "MCQ_REGEX", "METRIC_INFO", "METRIC_KINDS",
           "NUM_REGEX", "classification_scores", "convert_to_number",
           "extract_mcq_letters", "extract_number_answer", "multiple_choice_grade",
           "parse_number", "prepare_mcq_answer", "score_item", "score_mcq",
           "score_numeric"]
