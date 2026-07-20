---
name: reaction-prediction
description: Predict reaction yield, products (forward), or precursors (retrosynthesis) with pretrained ReactionT5v2 models.
version: 1.0.0
risk_level: medium
required_tools:
  - predict_reaction_t5
  - standardize_smiles
allowed_paths:
  - /workspace
expected_outputs:
  - reactiont5_predictions.csv
task_types:
  - reaction_prediction
---

# Procedure

1. 予測の種類を決める:
   - **yield**: 反応条件から収率（0–100%）を回帰予測
   - **forward**: 反応物・試薬から生成物 SMILES を予測
   - **retrosynthesis**: 生成物から前駆体 SMILES を予測
2. SMILES は事前に `standardize_smiles` で正準化する。
3. モデル入力文字列を task ごとの形式で組み立てる:
   - yield: `REACTANT:<smiles>.<smiles>REAGENT:<smiles> PRODUCT:<smiles>`
   - forward: `REACTANT:<smiles>.<smiles>REAGENT:<smiles>`
   - retrosynthesis: 生成物 SMILES のみ（接頭辞なし）
   複数成分は `.` で連結する。REAGENT が無い場合も `REAGENT:` 接頭辞自体は残す。
4. `predict_reaction_t5` を呼ぶ。複数候補が必要な forward/retrosynthesis では
   `num_beams` を 3〜5 に上げる（候補は candidates 列に `|` 区切りで入る）。
5. 結果 `reactiont5_predictions.csv` を確認し、予測 SMILES は
   `standardize_smiles` でパース可能かチェックしてから報告する。

# Notes

- このツールは専用 conda 環境（reactiont5: torch + transformers）で実行される。
  pyscf 環境とは分離されており、`run_python_sandbox` から torch を import しても動かない。
  torch が必要な処理は必ず `predict_reaction_t5` を使うこと。
- 予測は学習済みモデル（sagawa/ReactionT5v2-*）による推定であり、
  実験値ではないことを報告に明記する。
- yield が 0–100% の範囲外になった場合は入力形式の誤りを疑う。

# Completion criteria

- `reactiont5_predictions.csv` が存在し、全行に予測値（predicted_yield または prediction）がある。
- 生成された SMILES が RDKit でパース可能（無効な場合はその旨を報告）。

# Recovery procedure

- error_type=missing_environment: conda 環境 `reactiont5` が無い。修復不能として報告する。
- error_type=model_unavailable: HuggingFace キャッシュ/ネットワークの問題。修復不能として報告する。
- error_type=timeout: 入力を分割して複数回に分けて呼び出す。
- 範囲外の yield / 無効な SMILES 出力: 入力文字列の形式（接頭辞・`.` 連結・正準化）を見直して1回だけ再試行する。
