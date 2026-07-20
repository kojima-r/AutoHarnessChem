---
name: dataset-inspection
description: Inspect tabular molecular datasets (rows, dtypes, missing values, SMILES validity) before any modeling or calculation.
version: 1.0.0
risk_level: low
required_tools:
  - inspect_dataset
  - standardize_smiles
allowed_paths:
  - /workspace
expected_outputs:
  - features_qc.csv
task_types:
  - dataset_analysis
  - molecular_regression
---

# Procedure

1. `inspect_dataset` でデータセットの行数・列型・欠損・統計量を確認する。
2. SMILES 列が存在する場合、`standardize_smiles` で全行をパースし、無効な SMILES を特定する。
3. 無効行・欠損行の件数と割合を記録する。無効行が過半数なら列の取り違えを疑い、他の列を確認する。
4. QC 結果（行ごとの valid フラグ・正準 SMILES）を `features_qc.csv` として保存する。
5. target 列の分布（定数でないか、外れ値、クラス不均衡）を確認して報告に含める。

# Completion criteria

- 行数・列構成・欠損状況が報告されている。
- SMILES 列の有効/無効件数が確定している。
- 後続処理が使える QC 済みテーブルが workspace に保存されている。

# Recovery procedure

- CSV 読み込み失敗時: 区切り文字・エンコーディング (utf-8 / cp932) を変えて再試行する。
- SMILES 列名が不明な場合: 各文字列列の先頭 10 行を `standardize_smiles` に通し、最も有効率が高い列を採用する。
