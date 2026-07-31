"""OptTDDFT（`tools/OptTDDFT` の `opt_tddft` パッケージ）を用いた量子化学ツール群。

harness 本体のプロセスに pyscf を入れず、`SandboxConfig.named_envs["opttddft"]`
（既定: conda env `pyscf`）で自己完結スクリプトとして実行する。以前 harness 内で
直接 pyscf を叩いていた `calculate_orbitals` は、この OptTDDFT ベースの実装に
置き換わっている。

提供するツール:
  - calculate_orbitals             … DFT/HF の SCF で HOMO/LUMO/gap（TDDFT なし・軽量）
  - calculate_tddft_spectrum       … TDDFT で励起波長・振動子強度（UV-Vis スペクトル）
  - optimize_absorption_wavelength … Optuna で目標吸収波長に近い分子を探索（MI）
  - scan_esipt_pes                 … ESIPT の Relaxed PES スキャン（S0/S1 曲面）

いずれも opt_tddft の実装（構造生成・SolverConfig・TDDFTSolver・PESScanner・
OptunaOptimizer・可視化/レポート）を再利用し、charge/spin の明示指定と
実行上限（timeout/threads）の制御だけを足している。
"""
from __future__ import annotations

from pathlib import Path

from schemas import ToolResult
from tools.envrun import EnvScript, artifact, run_env_script

# tools/OptTDDFT — editable install が無い環境でも import できるよう sys.path へ渡す
OPT_TDDFT_ROOT = Path(__file__).resolve().parent / "OptTDDFT"

# geomeTRIC は構造最適化（use_geom_opt / ESIPT の拘束付き最適化）でのみ必要
_GEOMETRIC_FAILURE = (
    r"No module named 'geometric'", "missing_dependency", False,
    "構造最適化には geomeTRIC が必要です（`conda run -n pyscf pip install geometric`。"
    "numpy を 1.26.4 に固定したまま入れること）。agent 側では修復不能で、"
    "scan_esipt_pes では必須、calculate_orbitals / calculate_tddft_spectrum では "
    "use_geom_opt=false にすれば回避できます。",
)
_PYSCF_FAILURES = (
    (r"Basis set .* not found|BasisNotFoundError", "invalid_input", False,
     "指定した基底関数がその元素に存在しません。`def2-svp` など広い基底に変更してください。"),
    (r"Electron number .* and spin", "invalid_input", False,
     "charge と spin の組み合わせが電子数と矛盾しています。分子に合わせて指定してください。"),
)


# ---------------------------------------------------------------------------
# 専用環境で実行されるスクリプト（prelude + 各モードの body）
# ---------------------------------------------------------------------------

_PRELUDE = r'''
import csv
import dataclasses
import json
import os
import sys
from pathlib import Path

spec = json.loads(Path("__INPUT_JSON__").read_text(encoding="utf-8"))

# PySCF のバックエンド (OpenBLAS/MKL) が全コアを占有しないようにする。
# pyscf の import より前に設定する必要がある（OptTDDFT README の推奨事項）。
_threads = str(int(spec.get("threads", 4)))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = _threads
os.environ.setdefault("MPLBACKEND", "Agg")  # 画像は必ずヘッドレスで描画する

_root = spec.get("opt_tddft_root")
if _root and Path(_root).exists() and _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
from pyscf import dft, gto, scf

from opt_tddft.core.quantum_solver import (CalculationTimeoutError, MoleculeBuildError,
                                           SCFConvergenceError, SolverConfig, TDDFTSolver)

HARTREE_TO_EV = 27.211386245988


def build_solver_config(s):
    return SolverConfig(
        basis=s["basis"], functional=s["functional"], max_cycle=s["max_cycle"],
        nstates=s["nstates"], solvent_model=s["solvent_model"],
        solvent_eps=s["solvent_eps"], use_geom_opt=s["use_geom_opt"],
        opt_max_steps=s["opt_max_steps"], timeout_seconds=s["timeout_seconds"],
    )


class Solver(TDDFTSolver):
    """opt_tddft の TDDFTSolver に電荷・スピンの明示指定を足したサブクラス。

    構造生成 (_generate_xyz_from_smiles) と DFT/TDDFT の設定は opt_tddft 側の
    実装をそのまま使う。run_calculation の結果は振動子強度の再利用のために保持する。
    """

    def __init__(self, config, charge=0, spin=0):
        super().__init__(config)
        self.charge = charge
        self.spin = spin
        self.last_results = {}

    def _build_pyscf_molecule(self, xyz_string):
        # charge/spin が既定 (0,0) のときだけ opt_tddft と同じスピン多重度スイープを行う
        spins = [self.spin] if (self.charge or self.spin) else [0, 1, 2]
        errors = []
        for spin in spins:
            try:
                mol = gto.Mole()
                mol.atom = xyz_string
                mol.basis = self.config.basis
                mol.charge = self.charge
                mol.spin = spin
                mol.build()
                return mol
            except Exception as e:
                errors.append("spin=%d: %s" % (spin, e))
        raise MoleculeBuildError("failed to build molecule (" + "; ".join(errors) + ")")

    def run_calculation(self, smiles):
        result = super().run_calculation(smiles)
        self.last_results[smiles] = result
        return result


def make_mf(mol, method, config):
    """method='HF' なら Hartree-Fock、それ以外は DFT 汎関数名として扱う。"""
    if method.upper() == "HF":
        mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
    else:
        mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        mf.xc = method
    mf.max_cycle = config.max_cycle
    if config.solvent_model == "pcm":
        mf = mf.PCM()
        mf.with_solvent.method = "IEF-PCM"
        mf.with_solvent.eps = config.solvent_eps
    if config.use_geom_opt:
        from pyscf.geomopt.geometric_solver import optimize
        mol_eq = optimize(mf, maxsteps=config.opt_max_steps)
        return make_mf(mol_eq, method, dataclasses.replace(config, use_geom_opt=False))
    return mf


def run_with_timeout(func, seconds):
    """OptTDDFT と同じ SIGALRM ベースのタイムアウト。"""
    import signal

    def _handler(signum, frame):
        raise CalculationTimeoutError("calculation timed out after %ds" % seconds)

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        return func()
    finally:
        signal.alarm(0)


def homo_lumo_ev(mf):
    """占有数から HOMO/LUMO を取り出して eV で返す（開殻は α/β を統合して判定）。"""
    energies, occupations = mf.mo_energy, mf.mo_occ
    if getattr(energies, "ndim", 1) == 2:
        energies = np.concatenate(energies)
        occupations = np.concatenate(occupations)
        order = np.argsort(energies)
        energies, occupations = energies[order], occupations[order]
    occupied = [e for e, o in zip(energies, occupations) if o > 0]
    virtual = [e for e, o in zip(energies, occupations) if o == 0]
    if not occupied or not virtual:
        raise SCFConvergenceError("could not identify HOMO/LUMO from occupations")
    return float(max(occupied)) * HARTREE_TO_EV, float(min(virtual)) * HARTREE_TO_EV


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finish(payload):
    Path("__OUTPUT_JSON__").write_text(
        json.dumps(payload, ensure_ascii=False, default=float), encoding="utf-8")
'''

_ORBITALS_BODY = r'''
config = build_solver_config(spec)
solver = Solver(config, charge=spec["charge"], spin=spec["spin"])
method = spec["method"]
rows, failures = [], []

for smiles in spec["smiles"]:
    try:
        xyz = solver._generate_xyz_from_smiles(smiles)   # opt_tddft: 多コンフォマー + UFF
        mol = solver._build_pyscf_molecule(xyz)
        mf = make_mf(mol, method, config)
        energy = run_with_timeout(mf.kernel, config.timeout_seconds)
        if not mf.converged:
            raise SCFConvergenceError("SCF did not converge")
        homo_ev, lumo_ev = homo_lumo_ev(mf)
        rows.append({
            "smiles": smiles,
            "method": method,
            "basis": config.basis,
            "total_energy_hartree": float(energy),
            "homo_ev": homo_ev,
            "lumo_ev": lumo_ev,
            "gap_ev": lumo_ev - homo_ev,
            "charge": mol.charge,
            "spin": mol.spin,
            "solvent_model": config.solvent_model or "gas",
            "engine": "opt_tddft",
        })
        print("[orbitals] %s: HOMO=%.3f eV LUMO=%.3f eV" % (smiles, homo_ev, lumo_ev))
    except Exception as e:
        failures.append({"smiles": smiles, "error": "%s: %s" % (type(e).__name__, e)})
        print("[orbitals] %s FAILED: %s" % (smiles, e))

write_csv(spec["output_csv"], rows)
finish({"results": rows, "failures": failures,
        "output_csv": spec["output_csv"] if rows else None})
'''

_SPECTRUM_BODY = r'''
config = build_solver_config(spec)
solver = Solver(config, charge=spec["charge"], spin=0)
state_rows, orbital_rows, failures, images = [], [], [], []

for smiles in spec["smiles"]:
    try:
        # SCF のタイムアウトは opt_tddft 側が SIGALRM で処理する（TDDFT 部分は
        # sandbox の timeout が上限になる）
        result = solver.run_calculation(smiles)
    except Exception as e:
        failures.append({"smiles": smiles, "error": "%s: %s" % (type(e).__name__, e)})
        print("[tddft] %s FAILED: %s" % (smiles, e))
        continue

    wavelengths = [float(w) for w in result["wavelengths_nm"]]
    strengths = [float(f) for f in result["oscillator_strengths"]]
    if not wavelengths:
        failures.append({"smiles": smiles, "error": "no converged excited states"})
        continue

    for index, (wavelength, strength) in enumerate(zip(wavelengths, strengths), start=1):
        state_rows.append({
            "smiles": smiles,
            "state_index": index,               # 1 = 最低励起状態 (S1)
            "wavelength_nm": wavelength,
            "excitation_energy_ev": 1240.0 / wavelength,
            "oscillator_strength": strength,
            "matrix_type": result["matrix_type"],
            "functional": config.functional,
            "basis": config.basis,
        })

    strongest = max(zip(wavelengths, strengths), key=lambda pair: pair[1])[0]
    orbital_rows.append({
        "smiles": smiles,
        "method": config.functional,
        "basis": config.basis,
        "homo_ev": float(result["homo_ev"]),
        "lumo_ev": float(result["lumo_ev"]),
        "gap_ev": float(result["gap_ev"]),
        "max_wavelength_nm": max(wavelengths),
        "strongest_wavelength_nm": strongest,
        "n_states": len(wavelengths),
        "matrix_type": result["matrix_type"],
        "solvent_model": config.solvent_model or "gas",
        "engine": "opt_tddft",
    })
    print("[tddft] %s: %d states, lambda_max=%.1f nm (f_max at %.1f nm)"
          % (smiles, len(wavelengths), max(wavelengths), strongest))

    if spec["plot"]:
        from opt_tddft.postprocess.visualizer import SpectrumVisualizer
        image = "%s_%d.png" % (spec["plot_prefix"], len(orbital_rows))
        SpectrumVisualizer.plot_spectrum(wavelengths, strengths, image,
                                         stdev=spec["plot_stdev"])
        images.append(image)

write_csv(spec["output_csv"], state_rows)
write_csv(spec["orbital_csv"], orbital_rows)
finish({"states": state_rows, "molecules": orbital_rows, "failures": failures,
        "images": images})
'''

_OPTUNA_BODY = r'''
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
from opt_tddft.optimize.optuna_runner import OptunaOptimizer


class Optimizer(OptunaOptimizer):
    """opt_tddft の OptunaOptimizer に探索の打ち切り時間（timeout）を足したもの。"""

    def run(self, study_name, n_trials=100, timeout=None):
        study = optuna.create_study(study_name=study_name,
                                    storage="sqlite:///%s.db" % study_name,
                                    direction="minimize", load_if_exists=True)
        study.optimize(self._objective, n_trials=n_trials, timeout=timeout,
                       catch=(Exception,))
        return study


config = build_solver_config(spec)
solver = Solver(config)
search_space = {
    "scaffold": spec["scaffold"],
    "side_chains": {"pos1": spec["side_chains_pos1"], "pos2": spec["side_chains_pos2"]},
    "target_wavelength_nm": spec["target_wavelength_nm"],
}
study = Optimizer(solver, search_space).run(
    study_name=spec["study_name"], n_trials=spec["n_trials"],
    timeout=spec["search_timeout_sec"])

rows = []
for trial in study.trials:
    wavelengths = trial.user_attrs.get("wavelengths_nm") or []
    rows.append({
        "trial": trial.number,
        "state": trial.state.name,
        "objective_nm": None if trial.value is None else float(trial.value),
        "pos1": trial.params.get("pos1"),
        "pos2": trial.params.get("pos2"),
        "smiles": trial.user_attrs.get("smiles", ""),
        "max_wavelength_nm": max(wavelengths) if wavelengths else None,
        "homo_ev": trial.user_attrs.get("homo_ev"),
        "lumo_ev": trial.user_attrs.get("lumo_ev"),
        "n_states": len(wavelengths),
    })
write_csv(spec["output_csv"], rows)

completed = [t for t in study.trials if t.state.name == "COMPLETE"]
completed.sort(key=lambda t: t.value)
summary = {
    "study_name": spec["study_name"],
    "storage": "%s.db" % spec["study_name"],
    "target_wavelength_nm": spec["target_wavelength_nm"],
    "scaffold": spec["scaffold"],
    "n_trials_requested": spec["n_trials"],
    "n_trials_total": len(study.trials),
    "n_completed_trials": len(completed),
    "n_pruned_trials": sum(1 for t in study.trials if t.state.name == "PRUNED"),
    "functional": config.functional,
    "basis": config.basis,
    "nstates": config.nstates,
    "solvent_model": config.solvent_model or "gas",
    "best_smiles": None,
    "best_wavelength_nm": None,
    "difference_from_target_nm": None,
    "best_params": None,
    "top_trials": [],
}
if completed:
    best = completed[0]
    best_wavelengths = best.user_attrs.get("wavelengths_nm") or []
    summary.update({
        "best_trial": best.number,
        "best_smiles": best.user_attrs.get("smiles", ""),
        "best_wavelength_nm": max(best_wavelengths) if best_wavelengths else None,
        "difference_from_target_nm": float(best.value),
        "best_params": dict(best.params),
        "best_homo_ev": best.user_attrs.get("homo_ev"),
        "best_lumo_ev": best.user_attrs.get("lumo_ev"),
    })
    seen, top = set(), []
    for trial in completed:
        smiles = trial.user_attrs.get("smiles", "")
        if smiles and smiles not in seen:
            seen.add(smiles)
            top.append(trial)
        if len(top) >= 10:
            break
    summary["top_trials"] = [
        {"trial": t.number, "smiles": t.user_attrs.get("smiles", ""),
         "difference_from_target_nm": float(t.value),
         "max_wavelength_nm": max(t.user_attrs.get("wavelengths_nm") or [0]) or None}
        for t in top
    ]

reports = []
if spec["generate_report"] and completed:
    from opt_tddft.postprocess.report_generator import (ExcelReporter, JsonReporter,
                                                        PowerPointReporter)
    from opt_tddft.postprocess.visualizer import SpectrumVisualizer

    results_data, images = [], []
    for trial in top:
        smiles = trial.user_attrs.get("smiles", "")
        wavelengths = trial.user_attrs.get("wavelengths_nm") or []
        # 振動子強度は user_attrs に残らないので、計算時に保持した結果から引く
        cached = solver.last_results.get(smiles, {})
        strengths = [float(f) for f in cached.get("oscillator_strengths", [])]
        if len(strengths) != len(wavelengths):
            strengths = [0.5] * len(wavelengths)
        image = "spectrum_trial_%d.png" % trial.number
        SpectrumVisualizer.plot_spectrum(wavelengths, strengths, image,
                                        stdev=spec["plot_stdev"])
        images.append(image)
        results_data.append({
            "trial_id": trial.number, "smiles": smiles,
            "wavelengths_nm": wavelengths,
            "homo_ev": trial.user_attrs.get("homo_ev", ""),
            "lumo_ev": trial.user_attrs.get("lumo_ev", ""),
            "gap_ev": (trial.user_attrs.get("lumo_ev", 0)
                       - trial.user_attrs.get("homo_ev", 0)),
        })
    JsonReporter.create_report(results_data, "tddft_top_trials.json")
    ExcelReporter.create_report(results_data, "tddft_optimization_summary.xlsx")
    PowerPointReporter.create_report(results_data, images,
                                     "tddft_optimization_summary.pptx")
    reports = ["tddft_top_trials.json", "tddft_optimization_summary.xlsx",
               "tddft_optimization_summary.pptx", *images]

Path(spec["output_json"]).write_text(
    json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print("[optuna] completed=%d/%d best=%s"
      % (len(completed), len(study.trials), summary["best_smiles"]))
finish({"summary": summary, "trials": rows, "reports": reports})
'''

_ESIPT_BODY = r'''
from opt_tddft.core.pes_scanner import ConstraintManager, PESScanner
from opt_tddft.postprocess.pes_visualizer import PESVisualizer

HARTREE_TO_KCAL = 627.509

config = build_solver_config(spec)
scanner = PESScanner(config)
idx1, idx2 = spec["atom_idx_1"], spec["atom_idx_2"]
distances = np.arange(spec["start_dist"],
                      spec["end_dist"] + spec["step_size"] / 2.0,
                      spec["step_size"])

rows, failures = [], []
current_xyz = spec["initial_xyz"].strip()

for distance in distances:
    try:
        adjusted = ConstraintManager.adjust_distance(current_xyz, idx1, idx2, float(distance))
        result = scanner.run_constrained_point(adjusted, distances=[(idx1, idx2)])
        e_s0 = float(result["total_energy_hartree"])
        wavelengths = [float(w) for w in result["wavelengths_nm"]]
        e_s1 = None
        if wavelengths:
            # S1 = S0 + 最低励起エネルギー（波長は低エネルギー順なので先頭が S1）
            e_s1 = e_s0 + (1240.0 / wavelengths[0]) / HARTREE_TO_EV
        rows.append({
            "distance_angstrom": float(distance),
            "s0_energy_hartree": e_s0,
            "s1_energy_hartree": e_s1,
            "s1_excitation_nm": wavelengths[0] if wavelengths else None,
        })
        current_xyz = result["optimized_xyz"]   # Relaxed scan: 前点の最適構造を引き継ぐ
        print("[esipt] d=%.2f A  S0=%.6f Eh  S1=%s"
              % (distance, e_s0, "n/a" if e_s1 is None else "%.6f Eh" % e_s1))
    except Exception as e:
        failures.append({"distance_angstrom": float(distance),
                         "error": "%s: %s" % (type(e).__name__, e)})
        print("[esipt] d=%.2f A FAILED: %s" % (distance, e))
        break   # 収束しなくなった時点で打ち切る（部分結果は保存する）

if not rows:
    raise SystemExit("no PES point could be calculated: %s" % failures)


def relative_kcal(values):
    usable = [v for v in values if v is not None]
    if not usable:
        return []
    lowest = min(usable)
    return [None if v is None else (v - lowest) * HARTREE_TO_KCAL for v in values]


s1_values = [r["s1_energy_hartree"] for r in rows]
s0_rel = relative_kcal([r["s0_energy_hartree"] for r in rows])
s1_rel = relative_kcal(s1_values)
for row, s0r, s1r in zip(rows, s0_rel, s1_rel or [None] * len(rows)):
    row["s0_relative_kcal"] = s0r
    row["s1_relative_kcal"] = s1r
write_csv(spec["output_csv"], rows)

series = [{"label": "S0 (Ground State)",
           "x": [r["distance_angstrom"] for r in rows],
           "y": [r["s0_energy_hartree"] for r in rows]}]
if all(v is not None for v in s1_values):
    series.append({"label": "S1 (Excited State)",
                   "x": [r["distance_angstrom"] for r in rows], "y": s1_values})
PESVisualizer.plot_energy_surface(
    data_series=series, output_path=spec["output_png"],
    x_label="Proton Transfer Distance (Angstrom)",
    title="ESIPT Potential Energy Surface")

summary = {
    "n_points": len(rows),
    "atom_idx_1": idx1,
    "atom_idx_2": idx2,
    "distance_range_angstrom": [rows[0]["distance_angstrom"], rows[-1]["distance_angstrom"]],
    "functional": config.functional,
    "basis": config.basis,
    "solvent_model": config.solvent_model or "gas",
    "s0_barrier_kcal": max(s0_rel) if s0_rel else None,
    "s1_barrier_kcal": max(s1_rel) if s1_rel else None,
    "s0_minimum_distance_angstrom": min(
        (r for r in rows if r["s0_relative_kcal"] is not None),
        key=lambda r: r["s0_relative_kcal"])["distance_angstrom"] if s0_rel else None,
    "s1_minimum_distance_angstrom": min(
        (r for r in rows if r["s1_relative_kcal"] is not None),
        key=lambda r: r["s1_relative_kcal"])["distance_angstrom"] if s1_rel else None,
    "failures": failures,
}
Path(spec["output_json"]).write_text(
    json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
finish({"summary": summary, "points": rows, "failures": failures})
'''


def _script(body: str, name: str) -> EnvScript:
    return EnvScript(body=_PRELUDE + body, script_name=f"_opttddft_{name}.py",
                     input_json=f"_opttddft_{name}_input.json",
                     output_json=f"_opttddft_{name}_output.json")


def _solver_spec(*, functional: str, basis: str, max_cycle: int, nstates: int,
                 solvent_model: str | None, solvent_eps: float, use_geom_opt: bool,
                 opt_max_steps: int, timeout_seconds: int, threads: int) -> dict:
    """スクリプトへ渡す SolverConfig 相当の共通部分。"""
    return {
        "opt_tddft_root": str(OPT_TDDFT_ROOT),
        "functional": functional,
        "basis": basis,
        "max_cycle": max_cycle,
        "nstates": nstates,
        "solvent_model": "pcm" if (solvent_model or "").lower() == "pcm" else None,
        "solvent_eps": solvent_eps,
        "use_geom_opt": bool(use_geom_opt),
        "opt_max_steps": opt_max_steps,
        "timeout_seconds": timeout_seconds,
        "threads": threads,
    }


def _failed(summary: str, error_type: str, retryable: bool = False) -> ToolResult:
    return ToolResult(status="failed", summary=summary, retryable=retryable,
                      error_type=error_type)


# ---------------------------------------------------------------------------
# 1. calculate_orbitals（旧 tools/chem.py の pyscf 実装の置き換え）
# ---------------------------------------------------------------------------

def calculate_orbitals(
    workspace: Path,
    smiles: list[str],
    method: str = "HF",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    solvent_model: str | None = None,
    solvent_eps: float = 4.7113,
    use_geom_opt: bool = False,
    opt_max_steps: int = 50,
    max_cycle: int = 200,
    output_csv: str = "orbital_features.csv",
    timeout_sec: int | None = None,
    threads: int = 4,
    *,
    sandbox,
) -> ToolResult:
    """OptTDDFT の構造生成 + PySCF SCF で HOMO/LUMO/gap を計算する（TDDFT なし）。"""
    if not smiles:
        return _failed("smiles is empty", "invalid_input")

    workspace = Path(workspace)
    per_molecule_timeout = timeout_sec or sandbox.config.timeout_sec
    spec = {
        "smiles": list(smiles),
        "method": method,
        "charge": int(charge),
        "spin": int(spin),
        "output_csv": output_csv,
        **_solver_spec(functional=method, basis=basis, max_cycle=max_cycle, nstates=0,
                       solvent_model=solvent_model, solvent_eps=solvent_eps,
                       use_geom_opt=use_geom_opt, opt_max_steps=opt_max_steps,
                       timeout_seconds=per_molecule_timeout, threads=threads),
    }
    run = run_env_script(
        sandbox, workspace, _script(_ORBITALS_BODY, "orbitals"), spec,
        timeout_sec=timeout_sec, threads=threads,
        extra_failures=(_GEOMETRIC_FAILURE, *_PYSCF_FAILURES),
        timeout_hint="分子数を分割するか、基底関数を小さくしてください。",
    )
    if run.error is not None:
        return run.error

    rows = run.payload.get("results", [])
    failures = run.payload.get("failures", [])
    if not rows:
        return ToolResult(
            status="failed",
            summary=f"{len(smiles)} 分子すべてで軌道計算に失敗しました",
            data={"failures": failures, "stdout": run.stdout[-2000:]},
            retryable=True, error_type="scf_failed",
        )

    csv_path = workspace / output_csv
    return ToolResult(
        status="success" if not failures else "partial",
        summary=(f"HOMO/LUMO を {len(rows)}/{len(smiles)} 分子で計算 "
                 f"({method}/{basis}) → {csv_path.name}"),
        data={"output_csv": str(csv_path), "results": rows, "failures": failures,
              "engine": "opt_tddft"},
        artifacts=[artifact(csv_path)] if csv_path.exists() else [],
        error_type="scf_failed" if failures else None,
    )


# ---------------------------------------------------------------------------
# 2. calculate_tddft_spectrum
# ---------------------------------------------------------------------------

def calculate_tddft_spectrum(
    workspace: Path,
    smiles: list[str],
    functional: str = "CAMB3LYP",
    basis: str = "6-31g(d)",
    nstates: int = 10,
    charge: int = 0,
    solvent_model: str | None = None,
    solvent_eps: float = 4.7113,
    use_geom_opt: bool = False,
    opt_max_steps: int = 50,
    max_cycle: int = 200,
    output_csv: str = "tddft_spectrum.csv",
    orbital_csv: str = "orbital_features.csv",
    plot: bool = True,
    plot_stdev: float = 50000.0,
    timeout_sec: int | None = None,
    threads: int = 4,
    *,
    sandbox,
) -> ToolResult:
    """OptTDDFT の TDDFTSolver で励起波長・振動子強度（UV-Vis スペクトル）を計算する。"""
    if not smiles:
        return _failed("smiles is empty", "invalid_input")
    if nstates < 1:
        return _failed("nstates は 1 以上にしてください（HOMO/LUMO だけなら "
                       "calculate_orbitals を使う）", "invalid_input")

    workspace = Path(workspace)
    spec = {
        "smiles": list(smiles),
        "charge": int(charge),
        "output_csv": output_csv,
        "orbital_csv": orbital_csv,
        "plot": bool(plot),
        "plot_prefix": "tddft_spectrum",
        "plot_stdev": plot_stdev,
        **_solver_spec(functional=functional, basis=basis, max_cycle=max_cycle,
                       nstates=nstates, solvent_model=solvent_model,
                       solvent_eps=solvent_eps, use_geom_opt=use_geom_opt,
                       opt_max_steps=opt_max_steps,
                       timeout_seconds=timeout_sec or sandbox.config.timeout_sec,
                       threads=threads),
    }
    run = run_env_script(
        sandbox, workspace, _script(_SPECTRUM_BODY, "spectrum"), spec,
        timeout_sec=timeout_sec, threads=threads,
        extra_failures=(_GEOMETRIC_FAILURE, *_PYSCF_FAILURES),
        timeout_hint=("分子数を分割し、nstates や基底関数を下げるか "
                      "timeout_sec を上げてください。"),
    )
    if run.error is not None:
        return run.error

    molecules = run.payload.get("molecules", [])
    failures = run.payload.get("failures", [])
    if not molecules:
        return ToolResult(
            status="failed",
            summary=f"{len(smiles)} 分子すべてで TDDFT 計算に失敗しました",
            data={"failures": failures, "stdout": run.stdout[-2000:]},
            retryable=True, error_type="tddft_failed",
        )

    outputs = [workspace / output_csv, workspace / orbital_csv]
    outputs += [workspace / name for name in run.payload.get("images", [])]
    lambda_max = ", ".join(f"{m['smiles']}: {m['max_wavelength_nm']:.0f}nm"
                           for m in molecules[:3])
    return ToolResult(
        status="success" if not failures else "partial",
        summary=(f"TDDFT ({functional}/{basis}, nstates={nstates}) で "
                 f"{len(molecules)}/{len(smiles)} 分子のスペクトルを計算 → "
                 f"{output_csv} / {orbital_csv}（λmax {lambda_max}）"),
        data={"output_csv": str(workspace / output_csv),
              "orbital_csv": str(workspace / orbital_csv),
              "molecules": molecules, "states": run.payload.get("states", []),
              "failures": failures, "images": run.payload.get("images", [])},
        artifacts=[artifact(p) for p in outputs if p.exists()],
        error_type="tddft_failed" if failures else None,
    )


# ---------------------------------------------------------------------------
# 3. optimize_absorption_wavelength（Optuna による MI 探索）
# ---------------------------------------------------------------------------

def optimize_absorption_wavelength(
    workspace: Path,
    scaffold: str,
    side_chains_pos1: list[str],
    side_chains_pos2: list[str],
    target_wavelength_nm: float = 500.0,
    n_trials: int = 10,
    study_name: str = "tddft_mi_optimization",
    functional: str = "CAMB3LYP",
    basis: str = "6-31g(d)",
    nstates: int = 10,
    solvent_model: str | None = None,
    solvent_eps: float = 4.7113,
    use_geom_opt: bool = False,
    opt_max_steps: int = 50,
    max_cycle: int = 200,
    search_timeout_sec: int = 900,
    generate_report: bool = False,
    output_csv: str = "optuna_trials.csv",
    output_json: str = "optimization_summary.json",
    plot_stdev: float = 50000.0,
    timeout_sec: int | None = None,
    threads: int = 4,
    *,
    sandbox,
) -> ToolResult:
    """骨格 + 置換基の探索空間から、目標吸収波長に近い分子を Optuna で探索する。"""
    if "[*:1]" not in scaffold or "[*:2]" not in scaffold:
        return _failed("scaffold にはダミー原子 [*:1] と [*:2] が必要です "
                       "(例: 'c1cc([*:1])ccc1[*:2]')", "invalid_input")
    if not side_chains_pos1 or not side_chains_pos2:
        return _failed("side_chains_pos1 / side_chains_pos2 を 1 つ以上指定してください",
                       "invalid_input")
    if n_trials < 1:
        return _failed("n_trials は 1 以上にしてください", "invalid_input")

    workspace = Path(workspace)
    # 1 trial あたりの計算上限は探索全体の打ち切り時間に合わせる（Prune で先へ進む）
    trial_timeout = max(60, int(search_timeout_sec))
    spec = {
        "scaffold": scaffold,
        "side_chains_pos1": list(side_chains_pos1),
        "side_chains_pos2": list(side_chains_pos2),
        "target_wavelength_nm": float(target_wavelength_nm),
        "n_trials": int(n_trials),
        "study_name": study_name,
        "search_timeout_sec": int(search_timeout_sec),
        "generate_report": bool(generate_report),
        "output_csv": output_csv,
        "output_json": output_json,
        "plot_stdev": plot_stdev,
        **_solver_spec(functional=functional, basis=basis, max_cycle=max_cycle,
                       nstates=nstates, solvent_model=solvent_model,
                       solvent_eps=solvent_eps, use_geom_opt=use_geom_opt,
                       opt_max_steps=opt_max_steps, timeout_seconds=trial_timeout,
                       threads=threads),
    }
    # 探索の打ち切り後にも CSV/レポートを書き切れるよう、sandbox 側に余裕を持たせる
    run = run_env_script(
        sandbox, workspace, _script(_OPTUNA_BODY, "optuna"), spec,
        timeout_sec=timeout_sec or search_timeout_sec + 600, threads=threads,
        extra_failures=(_GEOMETRIC_FAILURE, *_PYSCF_FAILURES),
        timeout_hint=("search_timeout_sec を下げる（打ち切り後も結果は保存されます）か、"
                      "基底関数・nstates を下げてください。"),
    )
    if run.error is not None:
        return run.error

    summary = run.payload.get("summary", {})
    outputs = [workspace / output_csv, workspace / output_json,
               workspace / f"{study_name}.db"]
    outputs += [workspace / name for name in run.payload.get("reports", [])]
    completed = summary.get("n_completed_trials", 0)
    if not completed:
        return ToolResult(
            status="failed",
            summary=(f"{summary.get('n_trials_total', 0)} trial すべてが Prune され、"
                     "有効な分子が得られませんでした（分子組み立て or SCF 失敗）"),
            data={"summary": summary, "trials": run.payload.get("trials", []),
                  "stdout": run.stdout[-2000:]},
            artifacts=[artifact(p) for p in outputs if p.exists()],
            retryable=True, error_type="no_valid_trial",
        )

    best_wavelength = summary.get("best_wavelength_nm")
    headline = (f"Optuna 探索完了: {completed}/{summary.get('n_trials_total')} trial 成功 "
                f"(pruned {summary.get('n_pruned_trials', 0)})、"
                f"best={summary.get('best_smiles')}")
    if best_wavelength is not None:
        headline += (f" λmax={best_wavelength:.1f}nm "
                     f"(target {target_wavelength_nm:.0f}nm, "
                     f"Δ={summary.get('difference_from_target_nm', 0.0):.1f}nm)")
    return ToolResult(
        status="success",
        summary=headline,
        data={"summary": summary, "trials": run.payload.get("trials", []),
              "output_json": str(workspace / output_json),
              "output_csv": str(workspace / output_csv),
              "reports": run.payload.get("reports", [])},
        artifacts=[artifact(p) for p in outputs if p.exists()],
    )


# ---------------------------------------------------------------------------
# 4. scan_esipt_pes
# ---------------------------------------------------------------------------

def scan_esipt_pes(
    workspace: Path,
    atom_idx_1: int,
    atom_idx_2: int,
    xyz: str | None = None,
    xyz_file: str | None = None,
    start_dist: float = 1.0,
    end_dist: float = 2.0,
    step_size: float = 0.1,
    functional: str = "CAMB3LYP",
    basis: str = "6-31g(d)",
    nstates: int = 5,
    solvent_model: str | None = None,
    solvent_eps: float = 4.7113,
    opt_max_steps: int = 50,
    max_cycle: int = 200,
    output_csv: str = "esipt_scan_results.csv",
    output_png: str = "esipt_pes_profile.png",
    output_json: str = "esipt_scan_summary.json",
    timeout_sec: int | None = None,
    threads: int = 4,
    *,
    sandbox,
) -> ToolResult:
    """ESIPT の Relaxed PES スキャン（距離拘束付き構造最適化 + TDDFT）を実行する。"""
    workspace = Path(workspace)
    if xyz_file:
        path = Path(xyz_file)
        if not path.is_absolute():
            path = workspace / path
        if not path.exists():
            return _failed(f"xyz_file が見つかりません: {xyz_file}", "input_not_found")
        xyz = path.read_text(encoding="utf-8")
    if not (xyz or "").strip():
        return _failed("xyz（インライン座標）または xyz_file を指定してください",
                       "invalid_input")

    lines = [ln for ln in (line.strip() for line in xyz.replace(";", "\n").splitlines())
             if ln]
    if lines and len(lines[0].split()) == 1:
        lines = lines[2:]                       # 標準 XYZ ヘッダ（原子数 + コメント）を落とす
    if max(atom_idx_1, atom_idx_2) >= len(lines):
        return _failed(f"原子インデックスが範囲外です（原子数 {len(lines)}、"
                       f"0 始まりで指定）", "invalid_input")
    if atom_idx_1 == atom_idx_2:
        return _failed("atom_idx_1 と atom_idx_2 は異なる原子を指定してください",
                       "invalid_input")
    if step_size <= 0 or end_dist <= start_dist:
        return _failed("start_dist < end_dist かつ step_size > 0 にしてください",
                       "invalid_input")

    n_points = int((end_dist - start_dist) / step_size) + 1
    spec = {
        "initial_xyz": "\n".join(lines),
        "atom_idx_1": int(atom_idx_1),
        "atom_idx_2": int(atom_idx_2),
        "start_dist": float(start_dist),
        "end_dist": float(end_dist),
        "step_size": float(step_size),
        "output_csv": output_csv,
        "output_png": output_png,
        "output_json": output_json,
        **_solver_spec(functional=functional, basis=basis, max_cycle=max_cycle,
                       nstates=nstates, solvent_model=solvent_model,
                       solvent_eps=solvent_eps, use_geom_opt=False,
                       opt_max_steps=opt_max_steps,
                       timeout_seconds=timeout_sec or sandbox.config.timeout_sec,
                       threads=threads),
    }
    run = run_env_script(
        sandbox, workspace, _script(_ESIPT_BODY, "esipt"), spec,
        timeout_sec=timeout_sec, threads=threads,
        extra_failures=(_GEOMETRIC_FAILURE, *_PYSCF_FAILURES),
        timeout_hint=("スキャン範囲を分割する（start_dist/end_dist）か step_size を "
                      "粗くしてください。"),
    )
    if run.error is not None:
        return run.error

    summary = run.payload.get("summary", {})
    failures = run.payload.get("failures", [])
    outputs = [workspace / output_csv, workspace / output_png, workspace / output_json]
    barrier = summary.get("s1_barrier_kcal")
    return ToolResult(
        status="success" if not failures else "partial",
        summary=(f"ESIPT PES スキャン: {summary.get('n_points')}/{n_points} 点を計算 "
                 f"(S0 barrier {summary.get('s0_barrier_kcal', 0):.1f} kcal/mol, "
                 f"S1 barrier {barrier if barrier is None else round(barrier, 1)} kcal/mol) "
                 f"→ {output_csv} / {output_png}"),
        data={"summary": summary, "points": run.payload.get("points", []),
              "failures": failures},
        artifacts=[artifact(p) for p in outputs if p.exists()],
        error_type="scan_incomplete" if failures else None,
    )
