---
name: aizynth-retrosynthesis
description: Plan multi-step retrosynthetic routes down to purchasable starting materials with AiZynthFinder (MCTS / Retro*).
version: 1.0.0
risk_level: medium
required_tools:
  - plan_retrosynthesis
  - standardize_smiles
  - inspect_artifact
allowed_paths:
  - /workspace
expected_outputs:
  - retrosynthesis_routes.json
  - retrosynthesis_routes.csv
task_types:
  - retrosynthesis_planning
---

# Procedure

1. **ツールの選択を間違えないこと。**
   - `plan_retrosynthesis`（このSkill）= AiZynthFinder による**多段の経路探索**。
     stock（購入可能化合物）まで遡り、経路木・段数・前駆体を返す。
   - `predict_reaction_t5(task="retrosynthesis")` = **1 段階**の前駆体予測。
     「この分子の前駆体候補は？」だけなら後者の方が速い。
   両方を使い、ReactionT5 の 1 段階候補と AiZynthFinder の経路を突き合わせるのも有効。
2. 目標 SMILES は `standardize_smiles` で正準化してから渡す。
3. `plan_retrosynthesis(targets=[...])` を呼ぶ。探索予算の目安:
   - まず既定（`algorithm="mcts"`, `iteration_limit=100`, `time_limit_sec=120`）で 1 分子。
   - 解けなければ `iteration_limit=300`, `time_limit_sec=300`, `max_transforms=8` へ上げる。
   - 複数分子を渡すと全体の実行時間は `分子数 × time_limit_sec` まで伸びる。
     3〜5 分子ずつに分けて呼ぶ。
4. 結果を読む:
   - `retrosynthesis_routes.csv` = 1 経路 1 行（target_smiles, route_rank, solved,
     n_steps, n_precursors, score, precursors）。まずこれで俯瞰する。
   - `retrosynthesis_routes.json` = 経路木（`routes[].tree`）を含む生データ。
     特定経路の反応段を説明するときだけ参照する（大きいので全文を読み込まない）。
   - `solved=true` は「その経路の全ての葉が stock にある」＝出発物質まで到達した意味。
5. 報告には **solved 分子数 / 全分子数、最良経路の段数、出発物質（precursors）**を含め、
   `score`（state score）は経路の良さの内部指標であり収率の予測値ではないと明記する。
   予測は学習済み expansion policy（USPTO 由来テンプレート）に基づく提案であり、
   実験的検証が別途必要であることも書く。

# Notes

- 実行は専用 conda 環境 `aizynth` で行われ、harness 本体や pyscf 環境からは独立している。
  `run_python_sandbox` から `aizynthfinder` を import しても動かない。
- 学習済みモデル（expansion policy + stock）は `config.yml` で指定される。
  未配置なら `error_type=model_unavailable` が返る（agent 側では修復不能）。
- 収率を知りたい場合は、得られた 1 段階反応について
  `predict_reaction_t5(task="yield")` を組み合わせる。

# Completion criteria

- `retrosynthesis_routes.json` と `retrosynthesis_routes.csv` が存在する。
- 各 target に経路が 1 件以上あり、solved な経路では全前駆体が in_stock。
- 未解決の target がある場合は、その事実と試した探索予算が報告されている。

# Recovery procedure

- **error_type=model_unavailable**: モデル未配置。修復不能として報告し、
  `download_public_data` と `AIZYNTH_CONFIG` の設定が必要であることを伝える。
- **error_type=no_route_found**: 経路が見つからない/stock に到達しない。
  `iteration_limit` を 3 倍、`time_limit_sec` を 2 倍にして 1 回だけ再試行する。
  それでも解けなければ「未解決」として、途中までの経路と未到達の中間体を報告する。
- **error_type=timeout**: target を分割して呼び直す（`time_limit_sec` × 分子数が上限）。
- **error_type=invalid_input（stock/expansion 名が無い）**: `stock` / `expansion` の
  指定を外して既定（config.yml の全件）に任せる。
- **error_type=missing_environment / missing_dependency**: 環境側の問題。修復不能として報告する。
