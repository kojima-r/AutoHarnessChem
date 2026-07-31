---
name: pyscf-orbitals
description: Calculate molecular orbital energies (HOMO/LUMO) and TDDFT absorption spectra with OptTDDFT (RDKit + PySCF).
version: 2.0.0
risk_level: medium
required_tools:
  - calculate_orbitals
  - calculate_tddft_spectrum
  - inspect_artifact
allowed_paths:
  - /workspace
expected_outputs:
  - orbital_features.csv
task_types:
  - orbital_calculation
---

# Procedure

計算エンジンは OptTDDFT（`tools/OptTDDFT` の `opt_tddft`）で、専用 conda 環境 `pyscf` の
subprocess として実行される。3D 構造生成（多コンフォマー探索 + UFF）は
ツール内部で行われるので、`generate_3d_structure` を別途呼ぶ必要はない。

1. **目的で使うツールを選ぶ**
   - HOMO/LUMO・gap・全エネルギーだけ → `calculate_orbitals`（SCF のみ・軽い）
   - 吸収波長・振動子強度・UV-Vis スペクトル → `calculate_tddft_spectrum`
     （`orbital_features.csv` も同時に出力される）
2. **軽い条件で1分子だけ試す** → 成功したら分子数・精度を上げる。
   - `calculate_orbitals` の既定は HF/STO-3G（高速確認用）。精度が必要なら
     `method="b3lyp"`, `basis="6-31g*"` 等へ引き上げる。
   - `calculate_tddft_spectrum` の既定は CAMB3LYP/6-31g(d)、`nstates=10`。
     TDDFT は SCF より桁違いに重いので、まず 1 分子・`nstates=5` で通し、
     必要なら `timeout_sec` を明示的に上げる（既定は sandbox の timeout）。
3. **電荷・スピンは分子に合わせて明示する**（例: SO4 は `charge=-2`, `spin=0`）。
   `RuntimeError: Electron number ... and spin ...` は charge/spin 不整合のサイン。
   `spin`（= 2S, 不対電子数）が 0 以外なら UHF/UKS で計算される。
   `calculate_tddft_spectrum` は閉殻（RKS）専用なので開殻分子には使わない。
4. **溶媒効果**が必要なら `solvent_model="pcm"` と `solvent_eps`（例: クロロホルム 4.7113）。
   気相計算が既定。
5. 結果は 1 分子 1 行の `orbital_features.csv`（smiles, homo_ev, lumo_ev, gap_ev,
   total_energy_hartree 列）、TDDFT は 1 状態 1 行の `tddft_spectrum.csv`
   （wavelength_nm, oscillator_strength, state_index 列）に保存される。
   `state_index=1` が最低励起状態 (S1) = 最長波長。
6. 報告では λmax を「最長波長」ではなく**振動子強度が最大の波長**
   （`strongest_wavelength_nm`）で述べると実測スペクトルと対応が付く。

# Completion criteria

- `orbital_features.csv` が存在し、全行で homo_ev < lumo_ev。
- HOMO/LUMO が -60〜+30 eV の物理的に妥当な範囲にある。
- TDDFT では `tddft_spectrum.csv` の wavelength_nm が 50〜2000 nm の範囲にある。
- SCF 未収束の分子は結果に含めず、失敗理由が報告されている（`failures` を確認）。

# Recovery procedure

- **SCF 未収束 (error_type=scf_failed)**: (a) 基底を STO-3G へ落とす、
  (b) `max_cycle` を 400 へ増やす、(c) `method="b3lyp"` など別の汎関数にする、の順で再試行。
- **タイムアウト (error_type=timeout)**: 分子数を分割する。`nstates` と基底を下げる。
  それでも足りなければ `timeout_sec` を明示的に上げる（sandbox 既定より長くできる）。
- **error_type=invalid_input で基底関数が無いと言われた**: Br/I などの重元素は
  `6-31g(d)` に含まれない。`basis="def2-svp"` に変更する。
- **error_type=missing_dependency で geometric が無い**: `use_geom_opt=false` にする
  （構造最適化なしでも計算できる）。環境へのインストールは agent 側では不可。
- **error_type=missing_environment**: conda 環境 `pyscf` が無い。修復不能として報告する。
- **HOMO > LUMO や範囲外の値**: 単位変換（Hartree/eV の二重変換）と占有電子数の算出を確認する。
- **TDDFT で励起状態が 0 件**: `nstates` を 15〜24 に増やす（極端に減らすと収束が悪化する）。
