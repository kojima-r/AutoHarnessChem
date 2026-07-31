#!/usr/bin/env python
"""OptTDDFT (PySCF + RDKit + Optuna) のライブラリを直接使うデモ。

conda 環境 `pyscf` で実行する（harness を経由しない、素の opt_tddft の使用例）。

  conda run -n pyscf python examples/pyscf_opt_tddft_demo.py
  conda run -n pyscf python examples/pyscf_opt_tddft_demo.py \
      --smiles C=O --smiles C=CC=O --functional b3lyp --basis sto-3g --nstates 5

やっていること:
  1. MoleculeBuilder    — 骨格 SMILES（[*:1] [*:2]）に置換基を molzip で結合
  2. TDDFTSolver        — SMILES → 3D 構造 → DFT → TDDFT（HOMO/LUMO・励起波長・振動子強度）
  3. SpectrumVisualizer — UV-Vis スペクトル画像（ガウシアン・スミアリング）
  4. 結果を CSV / JSON にまとめて出力

出力（--out、既定 examples/output/pyscf）:
  tddft_demo_states.csv     … 1 励起状態 1 行
  tddft_demo_molecules.csv  … 1 分子 1 行（HOMO/LUMO/gap/λmax）
  tddft_demo_summary.json   … 実行条件と結果のまとめ
  spectrum_<smiles>.png     … 分子ごとの吸収スペクトル
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent / "output" / "pyscf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smiles", action="append", default=[],
                        help="計算する分子（複数可、既定: C=O と C=CC=O）")
    parser.add_argument("--functional", default="b3lyp", help="交換相関汎関数")
    parser.add_argument("--basis", default="sto-3g", help="基底関数")
    parser.add_argument("--nstates", type=int, default=5, help="計算する励起状態数")
    parser.add_argument("--solvent", choices=["none", "pcm"], default="none",
                        help="PCM 溶媒モデルを使うか")
    parser.add_argument("--solvent-eps", type=float, default=4.7113,
                        help="溶媒の誘電率（--solvent pcm のとき）")
    parser.add_argument("--timeout", type=int, default=1800, help="1 分子あたりの上限秒")
    parser.add_argument("--threads", type=int, default=4, help="PySCF に許可するスレッド数")
    parser.add_argument("--scaffold", default="c1cc([*:1])ccc1[*:2]",
                        help="MoleculeBuilder のデモに使う骨格 SMILES")
    parser.add_argument("--side-chains", nargs=2, default=["C=O", "N"],
                        metavar=("POS1", "POS2"), help="骨格に付ける置換基")
    parser.add_argument("--with-scaffold", action="store_true",
                        help="組み立てた分子も TDDFT 計算する（重い）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="出力ディレクトリ")
    return parser.parse_args()


def limit_threads(threads: int) -> None:
    """PySCF のバックエンドが全コアを占有しないようにする（import 前に設定）。"""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)
    os.environ.setdefault("MPLBACKEND", "Agg")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    limit_threads(args.threads)

    # import は limit_threads の後（環境変数を効かせるため）
    from opt_tddft.core.builder import MoleculeBuilder
    from opt_tddft.core.quantum_solver import SolverConfig, TDDFTSolver
    from opt_tddft.postprocess.visualizer import SpectrumVisualizer

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 骨格 + 置換基から分子を組み立てる（Optuna 探索で使われるのと同じ経路）
    side_chains = [MoleculeBuilder.format_side_chain(args.side_chains[0], 1),
                   MoleculeBuilder.format_side_chain(args.side_chains[1], 2)]
    assembled = MoleculeBuilder.build_from_scaffold(args.scaffold, side_chains)
    print(f"[builder] {args.scaffold} + {side_chains} -> {assembled}")

    targets = list(args.smiles) or ["C=O", "C=CC=O"]
    if args.with_scaffold:
        targets.append(assembled)

    # 2. TDDFT 計算
    config = SolverConfig(
        basis=args.basis, functional=args.functional, nstates=args.nstates,
        solvent_model=None if args.solvent == "none" else "pcm",
        solvent_eps=args.solvent_eps, use_geom_opt=False,
        timeout_seconds=args.timeout,
    )
    solver = TDDFTSolver(config)
    print(f"[solver] {args.functional}/{args.basis} nstates={args.nstates} "
          f"solvent={args.solvent} threads={args.threads}")

    state_rows: list[dict] = []
    molecule_rows: list[dict] = []
    failures: list[dict] = []
    images: list[str] = []

    for smiles in targets:
        started = time.time()
        try:
            result = solver.run_calculation(smiles)
        except Exception as e:                       # 1 分子の失敗で全体を止めない
            failures.append({"smiles": smiles, "error": f"{type(e).__name__}: {e}"})
            print(f"[tddft] {smiles}: FAILED ({e})")
            continue
        elapsed = time.time() - started

        wavelengths = [float(w) for w in result["wavelengths_nm"]]
        strengths = [float(f) for f in result["oscillator_strengths"]]
        if not wavelengths:
            failures.append({"smiles": smiles, "error": "no converged excited states"})
            continue

        for index, (wavelength, strength) in enumerate(zip(wavelengths, strengths), 1):
            state_rows.append({
                "smiles": smiles, "state_index": index,
                "wavelength_nm": round(wavelength, 3),
                "excitation_energy_ev": round(1240.0 / wavelength, 4),
                "oscillator_strength": round(strength, 5),
                "matrix_type": result["matrix_type"],
            })

        strongest = max(zip(wavelengths, strengths), key=lambda pair: pair[1])
        molecule_rows.append({
            "smiles": smiles,
            "functional": args.functional, "basis": args.basis,
            "homo_ev": round(float(result["homo_ev"]), 4),
            "lumo_ev": round(float(result["lumo_ev"]), 4),
            "gap_ev": round(float(result["gap_ev"]), 4),
            "max_wavelength_nm": round(max(wavelengths), 3),
            "strongest_wavelength_nm": round(strongest[0], 3),
            "strongest_oscillator_strength": round(strongest[1], 5),
            "n_states": len(wavelengths),
            "matrix_type": result["matrix_type"],
            "elapsed_sec": round(elapsed, 1),
        })
        print(f"[tddft] {smiles}: HOMO={result['homo_ev']:.3f} eV "
              f"LUMO={result['lumo_ev']:.3f} eV  "
              f"lambda_max={max(wavelengths):.1f} nm  "
              f"f_max@{strongest[0]:.1f} nm  ({elapsed:.1f}s, {result['matrix_type']})")

        # 3. スペクトル画像
        safe = re.sub(r"[^A-Za-z0-9]+", "_", smiles).strip("_") or "molecule"
        image = out_dir / f"spectrum_{safe}.png"
        SpectrumVisualizer.plot_spectrum(wavelengths, strengths, str(image))
        images.append(image.name)

    # 4. 出力
    write_csv(out_dir / "tddft_demo_states.csv", state_rows)
    write_csv(out_dir / "tddft_demo_molecules.csv", molecule_rows)
    summary = {
        "engine": "opt_tddft (PySCF + RDKit)",
        "functional": args.functional, "basis": args.basis, "nstates": args.nstates,
        "solvent_model": args.solvent, "threads": args.threads,
        "scaffold_demo": {"scaffold": args.scaffold, "side_chains": args.side_chains,
                          "assembled_smiles": assembled},
        "molecules": molecule_rows, "failures": failures, "images": images,
    }
    (out_dir / "tddft_demo_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[done] {len(molecule_rows)}/{len(targets)} 分子を計算 -> {out_dir}")
    for name in sorted(p.name for p in out_dir.iterdir()):
        print(f"  - {name}")
    if failures:
        print(f"[warn] {len(failures)} 分子が失敗: {failures}")
    return 0 if molecule_rows else 1


if __name__ == "__main__":
    sys.exit(main())
