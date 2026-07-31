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
# 分子の電子遷移として妥当な波長範囲 (nm)。この外は単位取り違え・状態の取り違えを疑う
WAVELENGTH_NM_RANGE = (50.0, 2000.0)
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
            warnings += self._check_spectrum(workspace, repairs)
        elif task.task_type == "molecular_design":
            warnings += self._check_molecular_design(workspace, repairs)
        elif task.task_type == "pes_scan":
            warnings += self._check_pes_scan(workspace, repairs)
        elif task.task_type == "molecular_regression":
            warnings += self._check_regression(workspace, repairs)
        elif task.task_type == "reaction_prediction":
            warnings += self._check_reaction_prediction(workspace, repairs)
        elif task.task_type == "retrosynthesis_planning":
            warnings += self._check_retrosynthesis(workspace, repairs)

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
            for i, row in enumerate(_read_rows(f, repairs)):
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

    def _check_spectrum(self, workspace: Path, repairs: list[str]) -> list[str]:
        """TDDFT スペクトル: 波長が妥当な範囲か、振動子強度が非負か。"""
        warnings: list[str] = []
        for f in workspace.rglob("tddft_spectrum*.csv"):
            rows = _read_rows(f, repairs)
            for i, row in enumerate(rows):
                wavelength = _float_field(row, ("wavelength_nm", "wavelength", "lambda_nm"))
                if wavelength is None:
                    warnings.append(f"{f.name} row {i}: wavelength_nm 列が見つかりません")
                    continue
                if not (WAVELENGTH_NM_RANGE[0] <= wavelength <= WAVELENGTH_NM_RANGE[1]):
                    repairs.append(
                        f"{f.name} row {i}: wavelength_nm={wavelength} が妥当範囲 "
                        f"{WAVELENGTH_NM_RANGE} 外です。"
                        "励起エネルギー (eV) → 波長 (nm) の変換 (1240/E) を確認してください。"
                    )
                strength = _float_field(row, ("oscillator_strength", "f"))
                if strength is not None and strength < 0:
                    warnings.append(
                        f"{f.name} row {i}: oscillator_strength={strength} が負です。")
        return warnings

    def _check_molecular_design(self, workspace: Path, repairs: list[str]) -> list[str]:
        """分子設計（探索）: 有効な trial があり、最良分子と波長が記録されているか。"""
        warnings: list[str] = []
        for f in workspace.rglob("optimization_summary.json"):
            summary = _read_json(f)
            if not isinstance(summary, dict):
                repairs.append(f"{f.name} が読み込めません（dict の JSON が必要）。")
                continue
            completed = summary.get("n_completed_trials")
            if completed is not None and completed < 1:
                repairs.append(
                    f"{f.name}: n_completed_trials=0 — 有効な分子が1つも得られていません。"
                    "置換基候補・基底関数を見直し、探索予算を増やして再試行してください。"
                )
                continue
            if not (summary.get("best_smiles") or "").strip():
                repairs.append(f"{f.name}: best_smiles が空です。")
            wavelength = _safe_float(summary.get("best_wavelength_nm"))
            if wavelength is None:
                warnings.append(f"{f.name}: best_wavelength_nm が記録されていません。")
            elif not (WAVELENGTH_NM_RANGE[0] <= wavelength <= WAVELENGTH_NM_RANGE[1]):
                repairs.append(
                    f"{f.name}: best_wavelength_nm={wavelength} が妥当範囲外です。")
            target = _safe_float(summary.get("target_wavelength_nm"))
            difference = _safe_float(summary.get("difference_from_target_nm"))
            if None not in (target, difference, wavelength) and (
                    abs(abs(wavelength - target) - difference) > 1.0):
                warnings.append(
                    f"{f.name}: difference_from_target_nm={difference} が "
                    f"|{wavelength} - {target}| と一致しません。目的関数の定義を確認してください。"
                )
        return warnings

    def _check_pes_scan(self, workspace: Path, repairs: list[str]) -> list[str]:
        """PES スキャン: 点数・距離の単調性・S1 > S0（励起状態が基底状態より高い）。"""
        warnings: list[str] = []
        for f in list(workspace.rglob("esipt_scan_results.csv")) + \
                list(workspace.rglob("pes_scan*.csv")):
            rows = _read_rows(f, repairs)
            if not rows:
                continue
            if len(rows) < 3:
                repairs.append(
                    f"{f.name}: スキャン点が {len(rows)} 点しかありません。"
                    "PES の形が判断できないため、範囲・刻み幅を見直して 3 点以上計算してください。"
                )
            distances = [_float_field(r, ("distance_angstrom", "Distance_Angstrom",
                                          "distance")) for r in rows]
            if any(d is None for d in distances):
                warnings.append(f"{f.name}: distance_angstrom 列が読めない行があります。")
            elif any(b <= a for a, b in zip(distances, distances[1:])):
                warnings.append(f"{f.name}: 距離が単調増加していません。")
            for i, row in enumerate(rows):
                s0 = _float_field(row, ("s0_energy_hartree", "S0_Energy_Hartree"))
                s1 = _float_field(row, ("s1_energy_hartree", "S1_Energy_Hartree"))
                if s0 is None or s1 is None:
                    continue
                if s1 <= s0:
                    repairs.append(
                        f"{f.name} row {i}: S1({s1}) <= S0({s0}) — 励起状態が基底状態より "
                        "低くなっています。S1 = S0 + 励起エネルギー の計算を確認してください。"
                    )
        return warnings

    def _check_retrosynthesis(self, workspace: Path, repairs: list[str]) -> list[str]:
        """逆合成経路探索: 経路が抽出でき、段数・前駆体・solved 判定が整合しているか。"""
        warnings: list[str] = []
        for f in workspace.rglob("retrosynthesis_routes.json"):
            payload = _read_json(f)
            entries = payload.get("targets") if isinstance(payload, dict) else payload
            if not isinstance(entries, list) or not entries:
                repairs.append(
                    f"{f.name}: 経路情報が空です。targets ごとに routes を持つ JSON "
                    "を出力してください。"
                )
                continue
            n_solved = 0
            for entry in entries:
                if not isinstance(entry, dict):
                    warnings.append(f"{f.name}: target エントリの形式が不正です。")
                    continue
                target = entry.get("target", "?")
                routes = entry.get("routes") or []
                if not routes:
                    warnings.append(
                        f"{f.name}: {target} は経路が 0 件です。"
                        "iteration_limit / time_limit_sec を増やすと解ける可能性があります。"
                    )
                    continue
                n_solved += bool(entry.get("solved"))
                for route in routes:
                    if not isinstance(route, dict):
                        continue
                    n_steps = _safe_float(route.get("n_steps"))
                    if n_steps is not None and n_steps < 1:
                        warnings.append(
                            f"{f.name}: {target} の経路に反応段数 0 のものがあります。")
                    if route.get("solved") and not all(route.get("precursors_in_stock") or [False]):
                        repairs.append(
                            f"{f.name}: {target} の経路が solved なのに stock 外の前駆体が "
                            "残っています。solved 判定（全葉が in_stock）を見直してください。"
                        )
            if entries and n_solved == 0:
                warnings.append(
                    f"{f.name}: どの target も stock まで到達していません（未解決）。"
                    "探索予算を増やすか、stock を変更してください。"
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
            for i, row in enumerate(_read_rows(f, repairs)):
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


def _read_rows(path: Path, repairs: list[str]) -> list[dict]:
    """CSV を dict の列として読む。読めない/空なら repairs に追記して [] を返す。"""
    try:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
    except Exception as e:
        repairs.append(f"{path.name} が読み込めません: {e}")
        return []
    if not rows:
        repairs.append(f"{path.name} が空です。")
    return rows


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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
