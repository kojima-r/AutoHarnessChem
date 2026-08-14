---
name: esipt-pes-scan
description: Run a relaxed potential-energy-surface scan for excited-state intramolecular proton transfer (ESIPT) with OptTDDFT.
version: 1.1.0
risk_level: medium
required_tools:
  - scan_esipt_pes
  - inspect_artifact
allowed_paths:
  - /workspace
expected_outputs:
  - esipt_scan_results.csv
  - esipt_pes_profile.png
task_types:
  - pes_scan
---

# Procedure

`scan_esipt_pes` は OptTDDFT の `PESScanner` を使い、各距離で
「プロトン座標の調整 → 距離拘束付き構造最適化（geomeTRIC）→ TDDFT」を繰り返す
Relaxed Scan を実行する。geomeTRIC を持つ専用 conda 環境（既定 `pyscf_esipt`）で動く。

**実測の目安**: サリチルアルデヒド（15 原子）を b3lyp/sto-3g・`nstates=3`・
`opt_max_steps=5` でスキャンすると **1 点あたり約 90 秒**。点数 × この時間が
そのまま実行時間になるので、まず 3 点で全体像を見る。

1. **構造は XYZ 座標で与える**（`xyz` にインライン、または workspace 内の `xyz_file`）。
   SMILES から自動生成すると原子インデックスがずれるため、既知の XYZ を使う。
   SMILES しかない場合は `generate_3d_structure` で xyz を作り、
   `inspect_artifact` で中身（行の並び）を確認してからインデックスを決める。
2. **原子インデックスは 0 始まり**。`atom_idx_1` = 移動するプロトン (H)、
   `atom_idx_2` = アクセプター原子 (N/O 等)。XYZ の行順と一致していることを必ず確認する。
3. **スキャン範囲**は `start_dist`（既定 1.0 Å）→ `end_dist`（既定 2.0 Å）、
   `step_size`（既定 0.1 Å）。点数 = 各点で拘束付き構造最適化 + TDDFT なので、
   まず粗い刻み（0.2 Å）で全体像を掴み、必要な領域だけ細かくする。
4. 出力は `esipt_scan_results.csv`（distance_angstrom, s0/s1_energy_hartree,
   s0/s1_relative_kcal）、`esipt_pes_profile.png`（相対 kcal/mol の PES）、
   `esipt_scan_summary.json`（障壁高さ・極小点の距離）。
5. 報告では **S0/S1 の障壁高さ (kcal/mol) と極小点の距離**を述べる。
   ESIPT は「S1 で障壁が下がり、プロトン移動側の極小が安定化する」ことが特徴なので、
   S0 と S1 の障壁を比較して言及する。

# Completion criteria

- `esipt_scan_results.csv` に 3 点以上のスキャン点があり、距離が単調増加している。
- 全行で S1 エネルギー > S0 エネルギー（励起状態が基底状態より高い）。
- `esipt_pes_profile.png` が生成されている。

# Recovery procedure

- **error_type=missing_dependency（geometric）/ missing_environment**: geomeTRIC を持つ
  専用環境（既定 `pyscf_esipt`）が無い。agent 側では修復不能として報告し、代替として
  `calculate_tddft_spectrum` で個別構造の励起エネルギーを比較する方法を提案する。
- **error_type=scan_incomplete（途中で打ち切り）**: 収束しなくなった点で停止し、
  そこまでの結果は保存されている。`basis="sto-3g"` に落とす、`opt_max_steps` を増やす、
  `step_size` を細かくして構造の連続性を保つ、の順で再試行する。
- **error_type=timeout**: `status=partial` なら計算できた点までの CSV が残っている。
  残りの距離範囲を `start_dist`/`end_dist` で分けて呼び直す。`step_size` を粗くする、
  `timeout_sec` を上げる、のいずれでもよい。
- **error_type=invalid_input（原子インデックス範囲外）**: XYZ の行数を数え直す。
  標準 XYZ ヘッダ（1行目 原子数・2行目 コメント）はツール側で除去される。
- **S1 <= S0 になった**: 励起状態が取れていない（TDDFT が失敗している）。`nstates` を増やす。
