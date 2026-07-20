---
name: execution-recovery
description: Classify execution failures and choose the right recovery — retry, lighten computation, or report as unfixable.
version: 1.0.0
risk_level: low
required_tools:
  - run_python_sandbox
  - inspect_artifact
allowed_paths:
  - /workspace
expected_outputs: []
task_types: []
---

# Procedure

失敗した ToolResult の `error_type` に応じて対応を分岐する。

| error_type | 対応 |
|---|---|
| missing_dependency | 修復不能。環境へのインストールが必要な旨を報告する（代替ライブラリで書き換え可能なら1回だけ試す）。 |
| timeout | 計算を軽くする（基底縮小・分子分割・反復回数減）。timeout の延長は不可。 |
| policy_violation | ブロックされた操作を使わない実装に書き換える。回避目的の難読化はしない。 |
| invalid_smiles / embedding_failed | 該当分子を除外して続行し、除外リストを報告する。 |
| scf_failed | pyscf-orbitals Skill の Recovery procedure に従う。 |
| runtime_error | stderr の traceback 末尾を読み、最大2回まで修正して再実行する。 |

# Rules

- 同一の失敗に対する再試行は最大 2 回。3 回目は方針を変えるか、修復不能として報告する。
- sandbox の conda 環境・timeout・作業ディレクトリは固定であり、変更を提案しない。
- 部分的に成功している場合は成功分を保存してから残りを再試行する。

# Completion criteria

- すべての失敗が「解決済み」「除外して続行」「修復不能（理由付き）」のいずれかに分類されている。

# Recovery procedure

- 本 Skill 自体が回復手順集である。判断に迷う場合は修復不能として正直に報告する。
