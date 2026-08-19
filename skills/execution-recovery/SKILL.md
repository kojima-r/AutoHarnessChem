---
name: execution-recovery
description: Classify execution failures and choose the right recovery — retry, lighten computation, or report as unfixable.
version: 1.3.0
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
| missing_dependency | 修復不能。環境へのインストールが必要な旨を報告する（代替ライブラリで書き換え可能なら1回だけ試す）。構造最適化の geomeTRIC 欠落なら `use_geom_opt=false` で回避できる。 |
| missing_environment | ツール専用の conda 環境 / docker image が使えない。**修復不能として報告する**（`conda run` や `docker run` を Bash で手動実行して回避しない）。別ツールでの代替は提案してよい。 |
| model_unavailable | 学習済みモデル/データが未配置。修復不能として報告し、必要な準備コマンドを伝える。 |
| timeout | 計算を軽くする（基底縮小・分子分割・反復回数減・状態数減）。専用環境ツールは `timeout_sec` を明示的に上げられるので、軽量化で足りない場合はそれを使う。 |
| timeout + status=partial | **完了分は成果物として保存済み**（`interrupted=true`）。`data.pending` に残りの入力が入っているので、それだけを（分割して）呼び直す。すべてやり直さない。 |
| out_of_memory | メモリ上限（SIGKILL / SIGSEGV / OpenBLAS の確保エラー）。専用環境で動く重いツール（量子化学・ReactionT5・逆合成・RDKit 系）はいずれも `memory_limit_mb` を持つので上げてよい（既定は ReactionT5 65536MB、量子化学 32768MB、その他 16384〜24576MB）。GPU を使うツールは VRAM ではなくアドレス空間の不足で `CUDA error: out of memory` になることがあり、これも上限を上げて直す。あわせて入力を分割し、基底関数・状態数・分子数・スレッド数を減らす。C 拡張は確保失敗を検査せず SIGSEGV になるため、原因不明のクラッシュもまずメモリ上限を疑う。 |
| policy_violation | ブロックされた操作を使わない実装に書き換える。回避目的の難読化はしない。 |
| invalid_smiles / embedding_failed | 該当分子を除外して続行し、除外リストを報告する。 |
| invalid_input | 引数の形（基底関数・charge/spin・原子インデックス・骨格のダミー原子）を直して1回だけ再試行する。 |
| scf_failed / tddft_failed | pyscf-orbitals Skill の Recovery procedure に従う。 |
| no_valid_trial | tddft-molecular-design Skill に従う（探索空間と計算条件を軽くして再試行）。 |
| scan_incomplete | esipt-pes-scan Skill に従う。部分結果は保存されているので、まずそれを報告する。 |
| no_route_found | aizynth-retrosynthesis Skill に従い、探索予算を上げて1回だけ再試行する。 |
| runtime_error | stderr の traceback 末尾を読み、最大2回まで修正して再実行する。 |

# Rules

- **前の試行で何が失敗したかは `tool_errors.jsonl`（workspace 直下）で確認できる。**
  失敗した呼び出しの引数・error_type・stderr/traceback が 1 行 1 件で残っているので、
  再計画の最初に `inspect_artifact` で読み、同じ失敗を繰り返さないこと。
- 同一の失敗に対する再試行は最大 2 回。3 回目は方針を変えるか、修復不能として報告する。
- sandbox の conda 環境・作業ディレクトリは固定であり、変更を提案しない。
  実行上限は「ツールが引数として持っている場合のみ」変更してよい（`run_python_sandbox` の timeout は不可）。
- 部分的に成功している場合は成功分を保存してから残りを再試行する。
- 専用環境（`pyscf` / `reactiont5` / `aizynth`）が必要なツールの代わりに
  `run_python_sandbox` で同じライブラリを import しても動かない。書き換えで回避しようとしない。

# Completion criteria

- すべての失敗が「解決済み」「除外して続行」「修復不能（理由付き）」のいずれかに分類されている。

# Recovery procedure

- 本 Skill 自体が回復手順集である。判断に迷う場合は修復不能として正直に報告する。
