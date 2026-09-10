"""ChemEval の採点（1 問ごと）。

ChemEval 公式の採点コード（`benchmarks_chemeval/ChemEval/Textual/code evaluate/`）は
Windows 前提のパス・中国語のファイル名分岐・rdkit / nltk / Levenshtein 依存で構成されて
いる。ここでは**指標の定義を公式に合わせたまま**、harness 本体の環境（重い依存なし）で
動くよう純 python で再実装する。分子構造の正準化と Tanimoto だけは rdkit が要るので、
`chem_metrics.py` が専用環境（sandbox の named_env `rdkit`）で計算した結果を
`context["chem"]` として受け取る（無ければ文字列一致に退避する）。

公式との対応:
  choice          選択肢集合の完全一致                （2_Extract の find_last_ABCD_letter 相当）
  true_false      正誤の一致                          （classification.calculate_accuracy）
  yes_no          Yes/No の一致                       （classification.calculate_accuracy）
  contains        gold が pred に含まれるか            （classification.calculate_accuracy2）
  entity_f1       カンマ区切り集合の F1                （entity_extraction.calculate_f1_score）
  relation_f1     `(…)` タプル集合の F1                （relation_extraction.calculate_f1_score）
  reagent_f1      試薬集合の F1                        （reagent_selection.calculate_f1_score）
  sider           20 ラベルの一致率                    （sider.sider）
  regression      RMSE / MAE（範囲は中央値）           （regression_new.calculate_rmse）
  range_overlap   範囲の重なり / 和                    （Reaction_Rate_Prediction.calculate_overlap）
  smiles          正準 SMILES 一致 + Tanimoto          （molecule_design_S.calculate_smiles_metrics）
  reaction_smiles 反応 SMILES の役割ごとの一致           （smiles_canonicalization.canonicalize_reaction_smiles 相当）
  text_exact      記法を揃えた完全一致（LaTeX 出力用）
  formula         原子組成の一致 + cos/L1/L2 類似度     （molecule_design_F.calculate_formula_metrics）
  iupac           小文字一致 + BLEU-4 + 編集距離        （molecule_design_I）
  selfies         正準化後の一致                        （molecule_design_S 相当）
  judge           LLM-as-judge の 0..1                 （LLM evaluate/*.py 相当）

score は「0..1 で高いほど良い代表値」。回帰系は score=None にして RMSE/MAE を別に集計する
（尺度が違うものを 1 つの平均に混ぜない）。
"""
from __future__ import annotations

import ast
import math
import re
import unicodedata
from collections import Counter
from typing import Any

# metric 名 → 集計時の主指標と向き。report.py が見る
METRIC_INFO: dict[str, dict[str, Any]] = {
    "choice":        {"primary": "accuracy", "higher_is_better": True},
    "true_false":    {"primary": "accuracy", "higher_is_better": True},
    "yes_no":        {"primary": "accuracy", "higher_is_better": True},
    "contains":      {"primary": "accuracy", "higher_is_better": True},
    "entity_f1":     {"primary": "f1", "higher_is_better": True},
    "relation_f1":   {"primary": "f1", "higher_is_better": True},
    "reagent_f1":    {"primary": "f1", "higher_is_better": True},
    "sider":         {"primary": "label_accuracy", "higher_is_better": True},
    "regression":    {"primary": "rmse", "higher_is_better": False},
    "range_overlap": {"primary": "overlap", "higher_is_better": True},
    "smiles":        {"primary": "exact_match", "higher_is_better": True},
    "formula":       {"primary": "exact_match", "higher_is_better": True},
    "iupac":         {"primary": "exact_match", "higher_is_better": True},
    "selfies":       {"primary": "exact_match", "higher_is_better": True},
    "judge":         {"primary": "judge_score", "higher_is_better": True},
    "text_exact":    {"primary": "exact_match", "higher_is_better": True},
    "reaction_smiles": {"primary": "exact_match", "higher_is_better": True},
}

# 集計は report.aggregate_metrics が行う。squared_error は sqrt(mean) で RMSE、
# abs_error は mean で MAE になり、それ以外は単純平均。

SIDER_LABELS = (
    "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Eye disorders",
    "Musculoskeletal and connective tissue disorders", "Gastrointestinal disorders",
    "Immune system disorders", "Reproductive system and breast disorders",
    "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
    "Endocrine disorders", "Vascular disorders", "Blood and lymphatic system disorders",
    "Skin and subcutaneous tissue disorders", "Congenital, familial and genetic disorders",
    "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders",
    "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions",
    "Ear and labyrinth disorders", "Cardiac disorders", "Nervous system disorders",
)

_NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_TRUE_WORDS = ("correct", "true", "正确", "正しい", "yes")
_FALSE_WORDS = ("incorrect", "false", "错误", "誤り", "no")


# --- 共通ヘルパ ---------------------------------------------------------

_DASHES = {ord(c): "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uff0d"}


def parse_literal(value: Any) -> Any:
    """`"[\'O\', \'O\']"` のような python/JSON リテラル文字列を実体に戻す。

    ChemEval の target は「リストの文字列表現」で入っていることがある
    （合成反应产物抽取 / 底物抽取 / 反应底物推荐 など 5 タスク）。一方 answer 側は
    `extract.loads_loose` が既にリストへパースしている。両者を揃えないと
    「完全正答なのに f1=0」になるため、target 側でも同じパースを行う。
    正規化の統一であって、判定を緩めるものではない。
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if len(text) < 2 or text[0] not in "[(" or text[-1] not in "])":
        return value
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return value
    return parsed if isinstance(parsed, (list, tuple)) else value


def as_text(value: Any) -> str:
    value = parse_literal(value)
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(as_text(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}: {as_text(v)}" for k, v in value.items())
    # NFKC で互換文字を畳む。ChemEval の target には `℃`(U+2103) や下付き数字が
    # そのまま入っており、`°C` や `H2O` と書いた答えが不一致になるため
    # （表記の違いであって内容の違いではない）。SMILES は ASCII なので影響しない。
    text = unicodedata.normalize("NFKC", str(value))
    # ダッシュ類を ASCII ハイフンへ。範囲表記（`120–180 ℃` と `120-180 °C`）の
    # 不一致を防ぐほか、負号 U+2212 を regression が数値として読めるようにする。
    return text.translate(_DASHES)


def norm_text(value: Any) -> str:
    """小文字化 + 空白正規化 + 端の記号除去。"""
    text = as_text(value).lower().replace("　", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .,;:!?\"'`*【】[]()（）")


def parse_number(value: Any) -> float | None:
    """数値を取り出す。範囲（180-370 / 5~10 / 3 to 4）は中央値にする。"""
    text = as_text(value).replace(",", "").replace("×10^", "e").replace("*10^", "e")
    if not text:
        return None
    span = re.search(rf"({_NUMBER})\s*(?:-|–|—|~|～|to|、)\s*({_NUMBER})", text)
    if span:
        low, high = float(span.group(1)), float(span.group(2))
        return (low + high) / 2
    found = re.search(_NUMBER, text)
    return float(found.group()) if found else None


def parse_range(value: Any) -> tuple[float, float] | None:
    """公式 Reaction_Rate_Prediction.parse_range と同じ扱い（'-' を区切りとみなす）。"""
    text = as_text(value).replace(",", "")
    if not text:
        return None
    numbers = re.findall(r"[+-]?[0-9]*\.?[0-9]+", text.replace("-", " "))
    if len(numbers) < 2:
        return None
    low, high = float(numbers[0]), float(numbers[1])
    return (min(low, high), max(low, high))


def split_entities(value: Any) -> set[str]:
    """カンマ等で区切られた集合にする（公式は空白除去 + ',' 分割）。"""
    text = as_text(value).lower()
    text = re.sub(r"\s+and\s+", ",", text)
    parts = re.split(r"[,;、；/\n]| \+ ", text)
    return {re.sub(r"\s+", "", p).strip(".") for p in parts if re.sub(r"\s+", "", p).strip(".")}


def set_f1(gold: set[str], pred: set[str]) -> float:
    if not gold or not pred:
        return 0.0
    hit = len(gold & pred)
    precision = hit / len(pred)
    recall = hit / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[0-9a-z]+|[^\sa-z0-9]", text.lower())


def sentence_bleu4(reference: str, hypothesis: str) -> float:
    """BLEU-4（重み 0.25 均等、スムージングなし = nltk の既定と同じ挙動）。"""
    ref, hyp = _tokens(reference), _tokens(hypothesis)
    if not hyp or not ref:
        return 0.0
    log_sum = 0.0
    for n in range(1, 5):
        ref_ngrams = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        hyp_ngrams = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        total = sum(hyp_ngrams.values())
        if total == 0:
            return 0.0
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
        if overlap == 0:
            return 0.0
        log_sum += 0.25 * math.log(overlap / total)
    brevity = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / len(hyp))
    return brevity * math.exp(log_sum)


def parse_formula(formula: Any) -> dict[str, int]:
    """分子式 → 原子数（公式 molecule_design_F.parse_molecular_formula と同じ規則）。"""
    counts: Counter[str] = Counter()
    for element, count in re.findall(r"([A-Z][a-z]*)(\d*)", as_text(formula)):
        counts[element] += int(count) if count else 1
    return dict(counts)


def _vector_similarities(pred: dict[str, int], gold: dict[str, int]) -> dict[str, float]:
    atoms = sorted(set(pred) | set(gold))
    v1 = [pred.get(a, 0) for a in atoms]
    v2 = [gold.get(a, 0) for a in atoms]
    dot = sum(x * y for x, y in zip(v1, v2))
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(y * y for y in v2))
    cosine = dot / (n1 * n2) if n1 and n2 else 0.0
    l1 = sum(abs(x - y) for x, y in zip(v1, v2))
    l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(v1, v2)))
    return {"atom_cosine": cosine, "atom_l1_similarity": 1 / (1 + l1),
            "atom_l2_similarity": 1 / (1 + l2)}


def _result(score: float | None, metrics: dict[str, float], *,
            answered: bool, valid: bool = True, detail: str = "") -> dict:
    return {"score": score, "metrics": metrics, "answered": answered,
            "valid": valid, "detail": detail}


# --- metric ごとの採点 ---------------------------------------------------

def score_choice(answer: Any, target: str, context: dict) -> dict:
    gold = set(re.findall(r"[A-Z]", as_text(target).upper()))
    text = as_text(answer).strip()
    letters = set(re.findall(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])", text))
    if not letters and re.fullmatch(r"[A-Da-d]{1,4}", text):
        letters = set(text)
    pred = {c.upper() for c in letters}
    correct = bool(pred) and pred == gold
    return _result(float(correct), {"accuracy": float(correct)},
                   answered=bool(text), valid=bool(pred),
                   detail=f"pred={sorted(pred)} gold={sorted(gold)}")


def _truthiness(text: str) -> bool | None:
    lowered = norm_text(text)
    if not lowered:
        return None
    for word in _FALSE_WORDS:          # incorrect は correct を含むので先に見る
        if word in lowered:
            return False
    for word in _TRUE_WORDS:
        if word in lowered:
            return True
    return None


def score_true_false(answer: Any, target: str, context: dict) -> dict:
    gold = _truthiness(target)
    pred = _truthiness(as_text(answer))
    correct = pred is not None and pred == gold
    return _result(float(correct), {"accuracy": float(correct)},
                   answered=bool(as_text(answer)), valid=pred is not None,
                   detail=f"pred={pred} gold={gold}")


def score_yes_no(answer: Any, target: str, context: dict) -> dict:
    gold = norm_text(target)
    text = norm_text(answer)
    pred = ""
    if text in ("yes", "no"):
        pred = text
    else:
        found = re.findall(r"\b(yes|no)\b", text)
        if found:
            pred = found[0]
    correct = bool(pred) and pred == gold
    return _result(float(correct), {"accuracy": float(correct)},
                   answered=bool(text), valid=bool(pred),
                   detail=f"pred={pred!r} gold={gold!r}")


def score_contains(answer: Any, target: str, context: dict) -> dict:
    """gold が pred に含まれるか（公式 classification.calculate_accuracy2 相当）。

    ただし単語境界を要求する。素の部分文字列一致だと
    gold=`organic chemistry` が pred=`inorganic chemistry` に含まれてしまい、
    **誤答が正解になる**（偽陽性）。判定を緩めないための境界条件。
    """
    gold, pred = norm_text(target), norm_text(answer)
    if not gold or not pred:
        return _result(0.0, {"accuracy": 0.0}, answered=bool(pred), valid=bool(pred))
    correct = pred == gold or re.search(rf"(?<!\w){re.escape(gold)}(?!\w)", pred) is not None
    return _result(float(correct), {"accuracy": float(correct)},
                   answered=bool(pred), valid=bool(pred))


def score_entity_f1(answer: Any, target: str, context: dict) -> dict:
    gold, pred = split_entities(target), split_entities(answer)
    f1 = set_f1(gold, pred)
    return _result(f1, {"f1": f1, "exact_set_match": float(bool(gold) and gold == pred)},
                   answered=bool(pred), valid=bool(pred))


def score_relation_f1(answer: Any, target: str, context: dict) -> dict:
    def tuples(value: Any) -> set[str]:
        text = re.sub(r"\s+", "", as_text(value).lower())
        return set(re.findall(r"\(.*?\)", text))

    gold, pred = tuples(target), tuples(answer)
    if not gold:                       # 括弧形式でない gold は集合 F1 に退避
        return score_entity_f1(answer, target, context)
    f1 = set_f1(gold, pred)
    return _result(f1, {"f1": f1}, answered=bool(as_text(answer)), valid=bool(pred))


def score_reagent_f1(answer: Any, target: str, context: dict) -> dict:
    gold, pred = split_entities(target), split_entities(answer)
    f1 = set_f1(gold, pred)
    metrics = {"f1": f1, "exact_set_match": float(bool(gold) and gold == pred)}
    chem = (context or {}).get("chem")
    if chem and chem.get("exact_match") is not None:
        metrics["canonical_exact_match"] = float(chem["exact_match"])
    return _result(f1, metrics, answered=bool(pred), valid=bool(pred))


def score_sider(answer: Any, target: str, context: dict) -> dict:
    from benchmarks_chemeval.extract import loads_loose   # 循環 import を避けて遅延

    gold_obj = loads_loose(as_text(target)) if not isinstance(target, dict) else target
    pred_obj = answer if isinstance(answer, dict) else loads_loose(as_text(answer))
    if not isinstance(gold_obj, dict):
        return _result(None, {}, answered=bool(answer), valid=False,
                       detail="gold を dict として解釈できません")
    labels = list(gold_obj) or list(SIDER_LABELS)
    if not isinstance(pred_obj, dict):
        return _result(0.0, {"label_accuracy": 0.0, "strict_match": 0.0},
                       answered=bool(as_text(answer)), valid=False,
                       detail="answer が dict ではありません")
    lowered = {norm_text(k): v for k, v in pred_obj.items()}
    hit = sum(1 for label in labels
              if norm_text(lowered.get(norm_text(label))) == norm_text(gold_obj[label]))
    accuracy = hit / len(labels) if labels else 0.0
    return _result(accuracy, {"label_accuracy": accuracy,
                              "strict_match": float(hit == len(labels))},
                   answered=True, valid=True, detail=f"{hit}/{len(labels)} labels")


def score_regression(answer: Any, target: str, context: dict) -> dict:
    gold = parse_number(target)
    pred = parse_number(answer)
    if gold is None:
        return _result(None, {}, answered=bool(as_text(answer)), valid=False,
                       detail="gold を数値として解釈できません")
    if pred is None:
        return _result(None, {}, answered=bool(as_text(answer)), valid=False,
                       detail="answer を数値として解釈できません")
    error = pred - gold
    return _result(None, {"squared_error": error ** 2, "abs_error": abs(error)},
                   answered=True, valid=True, detail=f"pred={pred} gold={gold}")


def score_range_overlap(answer: Any, target: str, context: dict) -> dict:
    gold_range = parse_range(target)
    pred_range = parse_range(answer)
    if gold_range is None:
        return _result(None, {}, answered=bool(as_text(answer)), valid=False,
                       detail="gold を範囲として解釈できません")
    if pred_range is None:
        return _result(0.0, {"overlap": 0.0}, answered=bool(as_text(answer)), valid=False,
                       detail="answer を範囲として解釈できません")
    intersection = max(0.0, min(pred_range[1], gold_range[1]) - max(pred_range[0], gold_range[0]))
    union = max(pred_range[1], gold_range[1]) - min(pred_range[0], gold_range[0])
    overlap = intersection / union if union else 0.0
    return _result(overlap, {"overlap": overlap}, answered=True, valid=True,
                   detail=f"pred={pred_range} gold={gold_range}")


def score_smiles(answer: Any, target: str, context: dict) -> dict:
    """正準 SMILES の完全一致（+ Tanimoto）。rdkit が無ければ文字列一致に退避。"""
    chem = (context or {}).get("chem") or {}
    text = as_text(answer).strip()
    if chem.get("exact_match") is None:
        exact = float(text == as_text(target).strip())
        return _result(exact, {"exact_match": exact, "rdkit": 0.0},
                       answered=bool(text), valid=bool(text),
                       detail="rdkit なし: 文字列一致で採点")
    metrics = {"exact_match": float(chem["exact_match"]), "rdkit": 1.0,
               "validity": float(bool(chem.get("valid")))}
    if chem.get("tanimoto") is not None:
        metrics["tanimoto"] = float(chem["tanimoto"])
    return _result(metrics["exact_match"], metrics, answered=bool(text),
                   valid=bool(chem.get("valid")), detail=chem.get("detail", ""))


def score_selfies(answer: Any, target: str, context: dict) -> dict:
    chem = (context or {}).get("chem") or {}
    text = as_text(answer).strip()
    if chem.get("exact_match") is not None:
        return score_smiles(answer, target, context)
    exact = float(re.sub(r"\s+", "", text) == re.sub(r"\s+", "", as_text(target)))
    return _result(exact, {"exact_match": exact, "rdkit": 0.0},
                   answered=bool(text), valid=bool(text),
                   detail="selfies/rdkit なし: 文字列一致で採点")


def score_formula(answer: Any, target: str, context: dict) -> dict:
    text = as_text(answer).strip()
    gold_atoms = parse_formula(target)
    pred_atoms = parse_formula(text)
    exact = float(bool(pred_atoms) and pred_atoms == gold_atoms)
    metrics = {"exact_match": exact}
    metrics.update(_vector_similarities(pred_atoms, gold_atoms) if pred_atoms
                   else {"atom_cosine": 0.0, "atom_l1_similarity": 0.0,
                         "atom_l2_similarity": 0.0})
    return _result(exact, metrics, answered=bool(text), valid=bool(pred_atoms))


def score_iupac(answer: Any, target: str, context: dict) -> dict:
    pred = as_text(answer).strip()
    gold = as_text(target).strip()
    if not pred:
        return _result(0.0, {"exact_match": 0.0, "bleu4": 0.0, "edit_similarity": 0.0,
                             "edit_distance": float(len(gold))},
                       answered=False, valid=False)
    exact = float(pred.lower() == gold.lower())
    distance = levenshtein(pred, gold)
    similarity = 1 - distance / max(len(pred), len(gold), 1)
    metrics = {"exact_match": exact, "bleu4": sentence_bleu4(gold, pred),
               "edit_similarity": similarity, "edit_distance": float(distance)}
    return _result(exact, metrics, answered=True, valid=True)


def score_text_exact(answer: Any, target: str, context: dict) -> dict:
    """記法の揺れ（空白・`$`・LaTeX の細かい差）を落とした完全一致。

    分子式・反応式の LaTeX 出力（multimodal）向け。編集距離も出しておく。
    """
    def canon(value: Any) -> str:
        text = as_text(value).lower()
        # 矢印は先に統一する（`\right` の除去で `\rightarrow` が壊れるため順序が重要）
        for arrow, unified in (("\\rightarrow", ">"), ("\\longrightarrow", ">"),
                               ("\\to", ">"), ("->", ">"), ("→", ">")):
            text = text.replace(arrow, unified)
        for junk in ("$", "\\left", "\\right", "\\,", "\\;", "\\!", "{", "}", " "):
            text = text.replace(junk, "")
        return text.strip()

    pred, gold = canon(answer), canon(target)
    exact = float(bool(pred) and pred == gold)
    distance = levenshtein(pred, gold)
    return _result(exact, {"exact_match": exact,
                           "edit_similarity": 1 - distance / max(len(pred), len(gold), 1)},
                   answered=bool(pred), valid=bool(pred))


def score_reaction_smiles(answer: Any, target: str, context: dict) -> dict:
    """反応 SMILES（`reactants>conditions>products`）の一致。

    rdkit があれば役割ごとに正準化して集合一致を見る（chem_metrics の kind="reaction"）。
    無ければ役割ごとの文字列集合 F1 に退避する。
    """
    chem = (context or {}).get("chem") or {}
    if chem.get("exact_match") is not None:
        metrics = {"exact_match": float(chem["exact_match"]),
                   "role_f1": float(chem.get("role_f1") or 0.0), "rdkit": 1.0}
        return _result(metrics["role_f1"], metrics, answered=bool(as_text(answer)),
                       valid=bool(chem.get("valid")), detail=chem.get("detail", ""))

    def roles(value: Any) -> list[set[str]]:
        parts = as_text(value).split(">")
        parts += [""] * (3 - len(parts))
        return [{c for c in re.split(r"[.\s]+", part) if c} for part in parts[:3]]

    pred_roles, gold_roles = roles(answer), roles(target)
    scores = [set_f1(g, p) if (g or p) else 1.0 for p, g in zip(pred_roles, gold_roles)]
    role_f1 = sum(scores) / len(scores)
    exact = float(pred_roles == gold_roles and any(gold_roles))
    return _result(role_f1, {"exact_match": exact, "role_f1": role_f1, "rdkit": 0.0},
                   answered=bool(as_text(answer)), valid=bool(as_text(answer)),
                   detail="rdkit なし: 文字列集合で採点")


def score_judge(answer: Any, target: str, context: dict) -> dict:
    """LLM-as-judge。judge が無い場合は集計対象外（score=None）にする。"""
    judge = (context or {}).get("judge")
    answered = bool(as_text(answer).strip())
    if not judge or judge.get("score") is None:
        return _result(None, {}, answered=answered, valid=False,
                       detail="judge 未実施（--judge を指定すると採点されます）")
    score = max(0.0, min(1.0, float(judge["score"])))
    return _result(score, {"judge_score": score}, answered=answered, valid=True,
                   detail=str(judge.get("reason", ""))[:300])


METRICS = {
    "choice": score_choice,
    "true_false": score_true_false,
    "yes_no": score_yes_no,
    "contains": score_contains,
    "entity_f1": score_entity_f1,
    "relation_f1": score_relation_f1,
    "reagent_f1": score_reagent_f1,
    "sider": score_sider,
    "regression": score_regression,
    "range_overlap": score_range_overlap,
    "smiles": score_smiles,
    "selfies": score_selfies,
    "formula": score_formula,
    "iupac": score_iupac,
    "text_exact": score_text_exact,
    "reaction_smiles": score_reaction_smiles,
    "judge": score_judge,
}

# 分子構造として正準化・Tanimoto を計算する metric（rdkit 環境へ回す対象）
CHEM_METRICS = ("smiles", "selfies", "reagent_f1", "reaction_smiles")
# LLM-as-judge が必要な metric
JUDGE_METRICS = ("judge",)


def score_item(metric: str, answer: Any, target: str, context: dict | None = None) -> dict:
    """metric 名で採点する。未知の metric は例外にせず score=None で返す。"""
    scorer = METRICS.get(metric)
    if scorer is None:
        return _result(None, {}, answered=bool(as_text(answer)), valid=False,
                       detail=f"未知の metric: {metric}")
    if answer is None or as_text(answer).strip() == "":
        # 無回答は「答えられなかった」として 0 点（回帰系は score=None のまま）
        empty = scorer(answer, target, context or {})
        empty["answered"] = False
        return empty
    return scorer(answer, target, context or {})
