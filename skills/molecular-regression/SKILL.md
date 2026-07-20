---
name: molecular-regression
description: Build and cross-validate regression models on molecular features (orbital energies, RDKit descriptors).
version: 1.0.0
risk_level: low
required_tools:
  - cross_validate_model
  - calculate_rdkit_descriptors
  - inspect_dataset
allowed_paths:
  - /workspace
expected_outputs:
  - cv_metrics.json
  - true_vs_pred.png
task_types:
  - molecular_regression
---

# Procedure

1. 特徴量テーブルを構築する。典型構成: orbital_features.csv (HOMO/LUMO/gap) +
   rdkit_descriptors.csv + データセット固有の数値列（温度・時間等）を smiles でマージ。
2. マージ後のテーブルを `merged_features.csv` として保存する。
3. `cross_validate_model` で K-fold CV（既定 5-fold、行数が少なければ 3-fold）を実行する。
4. `cv_metrics.json`（r2 / rmse / mae）、`oof_predictions.csv`、`true_vs_pred.png` を確認する。
5. 特徴量重要度や残差の傾向があれば報告に加える。

# Notes

- target 列・ID 列・SMILES 文字列を特徴量に入れない（リーク防止）。
- 行数 < fold 数のときは fold 数を減らす。leave-one-out は n<20 のときのみ。
- r2 > 0.999 はリークを疑う。r2 < -1 はモデル不成立（特徴量を見直す）。

# Completion criteria

- `cv_metrics.json` に r2 / rmse / mae が記録されている。
- `true_vs_pred.png`（対角線付き散布図）が存在する。
- 使用した特徴量リストと除外行数が報告されている。

# Recovery procedure

- マージで行が消える: smiles の正準形不一致が原因。両側を standardize してからマージする。
- 欠損だらけの特徴量列: 列を落とすか、median 補完を行い、その旨を報告する。
- 極端に悪い r2: 特徴量を RDKit 記述子のみ / 軌道のみに切り替えて切り分ける。
