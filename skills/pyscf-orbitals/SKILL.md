---
name: pyscf-orbitals
description: Calculate molecular orbital energies (HOMO/LUMO) with RDKit and PySCF.
version: 1.2.0
risk_level: medium
required_tools:
  - calculate_orbitals
  - run_python_sandbox
  - inspect_artifact
allowed_paths:
  - /workspace
expected_outputs:
  - orbital_features.csv
task_types:
  - orbital_calculation
---

# Procedure

1. rdkit-preparation Skill の手順で 3D 構造を得る。
2. `calculate_orbitals` で SCF 計算を行う。既定は HF/STO-3G（高速確認用）。
   精度が必要なら method="b3lyp", basis="6-31g*" へ引き上げる。
3. 電荷・スピンは分子に合わせて明示する（例: SO4 は charge=-2, spin=0）。
   `RuntimeError: Electron number ... and spin ...` は charge/spin 不整合のサイン。
4. HOMO/LUMO は Hartree から eV へ変換して記録する（1 Hartree = 27.2114 eV）。
5. 結果は 1 分子 1 行の `orbital_features.csv`（smiles, homo_ev, lumo_ev, gap_ev, total_energy_hartree 列）に保存する。

# Completion criteria

- `orbital_features.csv` が存在し、全行で homo_ev < lumo_ev。
- HOMO/LUMO が -60〜+30 eV の物理的に妥当な範囲にある。
- SCF 未収束の分子は結果に含めず、失敗理由が報告されている。

# Recovery procedure

- SCF 未収束: (a) 初期猜勢を変える (`mf.init_guess='atom'`)、(b) level shift を入れる、
  (c) 基底を STO-3G へ落とす、の順で再試行する。
- タイムアウト: 分子数を分割して複数回に分ける。基底を小さくする。timeout 自体は延長不可。
- HOMO > LUMO や範囲外の値: 単位変換（Hartree/eV の二重変換）と占有電子数の算出を確認する。
