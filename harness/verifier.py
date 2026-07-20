"""Scientific Verifier。

終了条件は特定の文字列ではなく、この構造化判定（VerificationResult）で決める。
Evolver による変更は禁止対象（評価基準を自己改善で甘くさせないため）。
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from schemas import TaskSpec, VerificationResult

# 軌道エネルギーの物理的に妥当な範囲 (eV)
ORBITAL_EV_RANGE = (-60.0, 30.0)
HARTREE_TO_EV = 27.211386245988


class ScientificVerifier:
    def verify(self, task: TaskSpec, workspace: Path) -> VerificationResult:
        workspace = Path(workspace)
        satisfied: list[str] = []
        missing: list[str] = []
        warnings: list[str] = []
        repairs: list[str] = []

        # 1) 期待される出力ファイルの存在チェック（glob 可）
        for pattern in task.expected_outputs:
            if list(workspace.rglob(pattern)):
                satisfied.append(f"output exists: {pattern}")
            else:
                missing.append(f"output missing: {pattern}")
                repairs.append(f"必要な出力 `{pattern}` を workspace 直下に生成してください。")

        # 2) ドメイン別の科学的妥当性チェック
        if task.task_type == "orbital_calculation":
            warnings += self._check_orbitals(workspace, repairs)
        elif task.task_type == "molecular_regression":
            warnings += self._check_regression(workspace, repairs)
        elif task.task_type == "reaction_prediction":
            warnings += self._check_reaction_prediction(workspace, repairs)

        passed = not missing and not repairs
        return VerificationResult(
            passed=passed,
            requirements_satisfied=satisfied,
            requirements_missing=missing,
            scientific_warnings=warnings,
            required_repairs=repairs,
        )

    # --- domain checks -------------------------------------------------

    def _check_orbitals(self, workspace: Path, repairs: list[str]) -> list[str]:
        warnings: list[str] = []
        files = list(workspace.rglob("orbital_features.csv")) + list(workspace.rglob("orbitals*.csv"))
        for f in files:
            try:
                rows = list(csv.DictReader(f.open(encoding="utf-8")))
            except Exception as e:
                repairs.append(f"{f.name} が読み込めません: {e}")
                continue
            if not rows:
                repairs.append(f"{f.name} が空です。")
                continue
            for i, row in enumerate(rows):
                homo = _float_field(row, ("homo_ev", "homo", "HOMO"))
                lumo = _float_field(row, ("lumo_ev", "lumo", "LUMO"))
                if homo is None or lumo is None:
                    warnings.append(f"{f.name} row {i}: homo/lumo 列が見つかりません")
                    continue
                if not (homo < lumo):
                    repairs.append(
                        f"{f.name} row {i}: HOMO({homo}) >= LUMO({lumo}) — 占有軌道の割当を確認してください。"
                    )
                for label, v in (("HOMO", homo), ("LUMO", lumo)):
                    if not (ORBITAL_EV_RANGE[0] <= v <= ORBITAL_EV_RANGE[1]):
                        warnings.append(
                            f"{f.name} row {i}: {label}={v} eV が妥当範囲 {ORBITAL_EV_RANGE} 外。"
                            "単位 (Hartree/eV) の変換を確認してください。"
                        )
        return warnings

    def _check_regression(self, workspace: Path, repairs: list[str]) -> list[str]:
        warnings: list[str] = []
        metric_files = list(workspace.rglob("cv_metrics.json")) + list(workspace.rglob("cv_metrics.csv"))
        for f in metric_files:
            metrics = _read_metrics(f)
            if metrics is None:
                repairs.append(f"{f.name} が読み込めません。")
                continue
            r2 = metrics.get("r2")
            if r2 is None:
                warnings.append(f"{f.name}: r2 が記録されていません")
                continue
            if not math.isfinite(r2) or r2 > 1.0:
                repairs.append(f"{f.name}: r2={r2} は不正な値です。")
            elif r2 > 0.999:
                warnings.append(f"{f.name}: r2={r2} — リーク（target が特徴量に混入）を疑ってください。")
            elif r2 < -1.0:
                warnings.append(f"{f.name}: r2={r2} — モデルが機能していません。特徴量を確認してください。")
        return warnings


    def _check_reaction_prediction(self, workspace: Path, repairs: list[str]) -> list[str]:
        warnings: list[str] = []
        for f in workspace.rglob("reactiont5_predictions.csv"):
            try:
                rows = list(csv.DictReader(f.open(encoding="utf-8")))
            except Exception as e:
                repairs.append(f"{f.name} が読み込めません: {e}")
                continue
            if not rows:
                repairs.append(f"{f.name} が空です。")
                continue
            for i, row in enumerate(rows):
                if "predicted_yield" in row and row["predicted_yield"] not in ("", None):
                    value = _safe_float(row["predicted_yield"])
                    if value is None or not (0.0 <= value <= 100.0):
                        warnings.append(
                            f"{f.name} row {i}: predicted_yield={row['predicted_yield']} が "
                            "0–100% の範囲外です。入力形式（REACTANT:/REAGENT:/PRODUCT:）を確認してください。"
                        )
                if "prediction" in row and not (row.get("prediction") or "").strip():
                    warnings.append(f"{f.name} row {i}: prediction が空です。")
        return warnings


def _float_field(row: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in row and row[key] not in ("", None):
            try:
                return float(row[key])
            except ValueError:
                return None
    return None


def _read_metrics(path: Path) -> dict | None:
    try:
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        if not rows:
            return None
        # csv の場合は metric,value 形式または1行のワイド形式を許容
        if {"metric", "value"} <= set(rows[0]):
            return {r["metric"]: _safe_float(r["value"]) for r in rows}
        return {k: _safe_float(v) for k, v in rows[0].items()}
    except Exception:
        return None


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
