---
name: tddft-molecular-design
description: Search for molecules whose TDDFT absorption wavelength matches a target, using OptTDDFT's Optuna (TPE) materials-informatics loop.
version: 1.1.0
risk_level: medium
required_tools:
  - optimize_absorption_wavelength
  - calculate_tddft_spectrum
  - inspect_artifact
allowed_paths:
  - /workspace
expected_outputs:
  - optimization_summary.json
  - optuna_trials.csv
task_types:
  - molecular_design
---

# Procedure

OptTDDFT の MI パイプライン（骨格 + 置換基の組み合わせを Optuna/TPE で探索し、
「吸収波長と目標波長の差の絶対値」を最小化する）を `optimize_absorption_wavelength`
として呼ぶ。専用 conda 環境 `pyscf` で実行される。

1. **目標波長の意味を決める**（`objective`）。
   - 既定 `objective="strongest"` — **振動子強度が最大の吸収帯**を目標に合わせる。
     実測の UV-Vis で観測される帯はこれなので、通常こちらを使う。
   - `objective="longest"` — 最長波長（S1）に合わせる。クマリン類などでは S1 が
     振動子強度 ≈ 0 の n→π\* **暗状態**で、実測されない帯を追ってしまうので注意。
   - `min_oscillator_strength`（既定 0.01）未満の状態は吸収帯として扱わない。
   - 結果には `objective_wavelength_nm`（目的値）と `strongest_wavelength_nm` /
     `max_wavelength_nm` の両方が残るので、報告では**どの帯の話か**を明示する。
2. **探索空間を決める**
   - `scaffold`: ダミー原子 `[*:1]` `[*:2]` を含む骨格 SMILES（例 `c1cc([*:1])ccc1[*:2]`）。
   - `side_chains_pos1` / `side_chains_pos2`: 置換基 SMILES のリスト。
     水素は `"H"` と書く（`[*:1]H` は不正。ツール側で `[H]` に変換される）。
     ホルミル基は `C(=O)H` ではなく `C=O` と書く（暗黙の水素を RDKit に補完させる）。
   - `target_wavelength_nm`: 目標波長。
3. **計算予算を明示する**。1 trial = TDDFT 1 回分なので、既定 `n_trials=10`,
   `search_timeout_sec=900` から始める。打ち切られても完了済み trial の結果は保存される。
   時間が足りない場合は `basis="sto-3g"` / `nstates=5` で当たりを付け、
   有望な骨格に絞ってから精度を上げる。
4. **単一分子で足場を確認する**。探索前に `calculate_tddft_spectrum` で骨格そのもの
   （置換基 = H）を 1 回計算し、条件（汎関数・基底・nstates）でまともな波長が出ることを
   確かめてから探索に入る。全 trial が Prune される事故を防げる。
5. `study_name` を変えると新しい探索、同じにすると `<study_name>.db` から**再開**になる。
   置換基リストを変えたまま同じ study 名で再開すると
   `CategoricalDistribution does not support dynamic value space` になるので、
   探索空間を変えたら必ず `study_name` も変える。
6. 結果は `optimization_summary.json`（best_smiles / best_wavelength_nm =
   目的の帯 / best_strongest_oscillator_strength / top_trials）と
   `optuna_trials.csv`（全 trial の状態・目的波長・最強吸収・最長波長）。
   `generate_report=true` で Excel・PowerPoint・スペクトル画像も生成される。
7. 報告には **best 分子・その波長（どの帯か）・振動子強度・目標との差・
   成功 trial 数 / 全 trial 数**を含める。
   Prune された trial 数も書く（探索が機能したかの判断材料になる）。

# Completion criteria

- `optimization_summary.json` に `n_completed_trials >= 1` と非空の `best_smiles` がある。
- 報告した波長がどの帯（最強吸収帯 / 最長波長）かが明示されている。
- `best_wavelength_nm` が 50〜2000 nm の妥当範囲にある。
- `optuna_trials.csv` に全 trial（COMPLETE / PRUNED）が記録されている。

# Recovery procedure

- **error_type=no_valid_trial（全 trial が Prune）**: 分子組み立ての失敗か SCF 失敗。
  (a) 置換基 SMILES を `standardize_smiles` で検証する、(b) `basis="sto-3g"` に落とす、
  (c) 骨格のダミー原子の位置と数（`[*:1]` と `[*:2]` の両方が必要）を確認する。
- **error_type=timeout**: `search_timeout_sec` を下げる（打ち切り後も結果は保存される）か、
  `n_trials` を減らす。`timeout_sec` は `search_timeout_sec + 600` が既定。
- **同じ SMILES ばかり出てくる**: 置換基候補が少なすぎる。候補を増やすか
  `n_trials` を候補の組み合わせ数以下に抑える。
- **error_type=missing_environment / missing_dependency**: 環境側の問題。修復不能として報告する。
