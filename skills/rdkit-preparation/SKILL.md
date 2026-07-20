---
name: rdkit-preparation
description: Standardize SMILES and build 3D structures with RDKit (ETKDGv3 + MMFF) as input for quantum chemistry.
version: 1.0.0
risk_level: low
required_tools:
  - standardize_smiles
  - generate_3d_structure
  - calculate_rdkit_descriptors
allowed_paths:
  - /workspace
expected_outputs:
  - "*.xyz"
task_types:
  - orbital_calculation
  - molecular_regression
---

# Procedure

1. 入力 SMILES を `standardize_smiles` で正準化する（塩・溶媒和は FragmentParent で除去）。
2. `generate_3d_structure` で 3D 化する。手順は AddHs → ETKDGv3 (randomSeed=42) → MMFF 最適化。
3. 埋め込み失敗時は UFF へフォールバック、それでも失敗する分子は failure として記録し他分子の処理を続ける。
4. 記述子が必要なら `calculate_rdkit_descriptors` で MolWt / LogP / TPSA 等を CSV に保存する。

# Notes

- RDKit のモジュール配置は版により移動する。`rdMolStandardize` は
  `from rdkit.Chem.MolStandardize import rdMolStandardize` を試し、ImportError なら
  `from rdkit.Chem import rdMolStandardize` を使うこと。
- 電荷を持つ分子（例: SO4^2-）は中性化せず、後段の量子化学計算へ charge を明示的に渡す。

# Completion criteria

- すべての有効分子に対し 3D 構造（xyz）または明示的な失敗理由が存在する。
- 出力ファイルは workspace 直下に保存されている。

# Recovery procedure

- ETKDG 失敗: `useRandomCoords=True` で再試行、または maxAttempts を増やす。
- MMFF パラメータ欠落: UFF に切り替える。
- どうしても 3D 化できない分子は除外し、除外リストを報告へ含める。
