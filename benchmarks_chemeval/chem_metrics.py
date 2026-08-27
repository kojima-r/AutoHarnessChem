"""分子構造としての採点（正準 SMILES 一致 / Tanimoto / 妥当性）。

rdkit（と SELFIES）は harness 本体の環境に入れない方針なので、tools/envrun.py の
「自己完結スクリプト + JSON 入出力」で専用環境（`sandbox.named_envs` の `rdkit`）へ投げる。
環境が無い場合は None を返し、metrics 側が文字列一致へ退避する。
"""
from __future__ import annotations

from pathlib import Path

from tools.envrun import EnvScript, run_env_script
from tools.sandbox import create_sandbox

# rdkit の import はコア数に比例した領域を要求するため上限を大きく取る（CLAUDE.md 参照）
MEMORY_LIMIT_MB = 16384

_SCRIPT = r'''
"""ChemEval の分子系採点（rdkit 環境で実行）。"""
import json

with open("__INPUT_JSON__", encoding="utf-8") as fh:
    spec = json.load(fh)

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog("rdApp.*")

try:
    import selfies as sf
except ImportError:
    sf = None


def to_mol(text, kind):
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    if kind == "selfies" and sf is not None and text.startswith("["):
        try:
            text = sf.decoder(text)
        except Exception:
            return None
    try:
        return Chem.MolFromSmiles(text)
    except Exception:
        return None


def canonical(text, kind):
    mol = to_mol(text, kind)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def fingerprint(text, kind):
    mol = to_mol(text, kind)
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return None


def split_multi(text):
    """'A and B' / 'A,B' / 'A.B' のような複数分子の答えを分解する。"""
    parts = []
    for chunk in str(text).replace(" and ", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


results = {}
pairs = spec.get("pairs", [])
for index, pair in enumerate(pairs):
    kind = pair.get("kind", "smiles")
    pred_raw, gold_raw = pair.get("pred"), pair.get("gold")
    entry = {"canonical_pred": None, "canonical_gold": None, "exact_match": None,
             "tanimoto": None, "role_f1": None, "valid": False, "detail": ""}
    if kind == "reagent":
        # 集合として一致を見る（順序・表記ゆれを正準 SMILES で吸収する）
        pred_set = {canonical(p, "smiles") for p in split_multi(pred_raw or "")}
        gold_set = {canonical(g, "smiles") for g in split_multi(gold_raw or "")}
        pred_set.discard(None)
        gold_set.discard(None)
        entry["valid"] = bool(pred_set)
        entry["exact_match"] = bool(gold_set) and pred_set == gold_set
        entry["canonical_pred"] = ".".join(sorted(pred_set)) or None
        entry["canonical_gold"] = ".".join(sorted(gold_set)) or None
    elif kind == "reaction":
        # 反応 SMILES: `reactants>conditions>products` を役割ごとに正準化して集合比較
        def role_sets(text):
            parts = str(text or "").split(">")
            parts += [""] * (3 - len(parts))
            out = []
            for part in parts[:3]:
                canon_set = set()
                for piece in part.replace("~", ".").split("."):
                    piece = piece.strip()
                    if not piece:
                        continue
                    canon_set.add(canonical(piece, "smiles") or piece)
                out.append(canon_set)
            return out

        pred_roles, gold_roles = role_sets(pred_raw), role_sets(gold_raw)

        def f1(gold, pred):
            if not gold and not pred:
                return 1.0
            if not gold or not pred:
                return 0.0
            hit = len(gold & pred)
            if not hit:
                return 0.0
            precision, recall = hit / len(pred), hit / len(gold)
            return 2 * precision * recall / (precision + recall)

        scores = [f1(g, p) for p, g in zip(pred_roles, gold_roles)]
        entry["role_f1"] = sum(scores) / len(scores)
        entry["exact_match"] = pred_roles == gold_roles and any(gold_roles)
        entry["valid"] = any(pred_roles)
        entry["canonical_pred"] = ">".join(".".join(sorted(r)) for r in pred_roles)
        entry["canonical_gold"] = ">".join(".".join(sorted(r)) for r in gold_roles)
    elif kind == "selfies" and sf is None and (
            str(pred_raw or "").startswith("[") or str(gold_raw or "").startswith("[")):
        # SELFIES を SMILES として読むと別分子になってしまうため、
        # selfies が無い環境では文字列一致に退避する
        entry["exact_match"] = (str(pred_raw or "").replace(" ", "")
                                == str(gold_raw or "").replace(" ", ""))
        entry["valid"] = bool(str(pred_raw or "").strip())
        entry["detail"] = "selfies 未インストール: 文字列一致で採点"
    else:
        pred_canon = canonical(pred_raw, kind)
        gold_canon = canonical(gold_raw, kind)
        entry["canonical_pred"] = pred_canon
        entry["canonical_gold"] = gold_canon
        entry["valid"] = pred_canon is not None
        if gold_canon is None:
            # gold が rdkit で読めない（SELFIES の片方向など）ときは文字列一致
            entry["exact_match"] = str(pred_raw or "").strip() == str(gold_raw or "").strip()
            entry["detail"] = "gold を rdkit で解釈できないため文字列一致"
        else:
            entry["exact_match"] = pred_canon is not None and pred_canon == gold_canon
            fp_pred = fingerprint(pred_raw, kind)
            fp_gold = fingerprint(gold_raw, kind)
            if fp_pred is not None and fp_gold is not None:
                entry["tanimoto"] = float(DataStructs.TanimotoSimilarity(fp_pred, fp_gold))
            else:
                entry["tanimoto"] = 0.0
    results[pair["id"]] = entry
    if (index + 1) % 100 == 0:
        with open("__PARTIAL_JSON__", "w", encoding="utf-8") as fh:
            json.dump({"results": results, "done": index + 1}, fh)

with open("__OUTPUT_JSON__", "w", encoding="utf-8") as fh:
    json.dump({"results": results, "done": len(pairs),
               "selfies_available": sf is not None}, fh)
'''


def compute(pairs: list[dict], sandbox_config, workspace: Path,
            *, timeout_sec: int = 1800) -> dict[str, dict] | None:
    """pairs = [{"id", "pred", "gold", "kind"}] を採点する。

    kind は "smiles" / "selfies" / "reagent" / "reaction"。
    環境が無い等で計算できない場合は None。
    """
    if not pairs:
        return {}
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    sandbox = create_sandbox(sandbox_config, workspace, env="rdkit")
    script = EnvScript(
        body=_SCRIPT,
        script_name="chemeval_chem_metrics.py",
        input_json="chemeval_chem_input.json",
        output_json="chemeval_chem_output.json",
        partial_json="chemeval_chem_partial.json",
    )
    run = run_env_script(
        sandbox, workspace, script, {"pairs": pairs},
        timeout_sec=timeout_sec, memory_limit_mb=MEMORY_LIMIT_MB,
        timeout_hint="採点対象を分割してください（--limit-per-task を下げる）。",
    )
    if run.payload is None:
        summary = run.error.summary if run.error else "unknown error"
        print(f"[chemeval] rdkit 採点をスキップします: {summary}")
        return None
    results = run.payload.get("results", {})
    if run.partial:
        print(f"[chemeval] rdkit 採点は途中で打ち切られました "
              f"({len(results)}/{len(pairs)} 件): {run.interrupted_reason}")
    return results
