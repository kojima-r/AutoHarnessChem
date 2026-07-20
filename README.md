# AutoHarnessChem

同一の量子化学・機械学習タスクを、実行時設定で **Deep Agents / Claude Agent SDK / OpenAI Agents SDK** の3基盤に切り替えて実行できる Agentic Harness。設計は [`../specification.md`](../specification.md) に基づく。

```text
User / API / CLI
        ↓
Task Interpreter          harness/controller.py (interpret_task)
        ↓
Common Harness Core       harness/
 ├─ Task Ledger           task_ledger.py
 ├─ Skill Registry        skill_registry.py (+ SkillCompiler)
 ├─ Tool Registry         tools/registry.py
 ├─ Policy Gate           policy.py
 ├─ Artifact Manager      artifacts.py
 ├─ Scientific Verifier   verifier.py（Evolver変更禁止）
 └─ Trace Normalizer      traces.py
        ↓
AgentRuntimeAdapter       adapters/{deepagents,claude,openai}.py
        ↓
Sandbox / Python Executor tools/sandbox.py (Docker / local conda)
```

## セットアップ

```bash
cd AutoHernessChem
pip install -e ".[dev]"                 # コア（pydantic + PyYAML）+ pytest

# 使いたい SDK を選んで追加（1つ以上）
pip install -e ".[claude]"              # Claude Agent SDK
pip install -e ".[openai]"              # OpenAI Agents SDK
pip install -e ".[deepagents]"          # Deep Agents (LangGraph)

# sandbox イメージ（docker を使う場合）
docker build -t autoharnesschem/sandbox:latest docker/
docker build -t autoharnesschem/reactiont5:latest -f docker/Dockerfile.reactiont5 docker/
# docker が無い環境では conda env を使う local sandbox に自動フォールバック
```

API キーは各SDKの流儀に従い環境変数で渡す（`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`、Web検索を使う場合は `TAVILY_API_KEY`）。

## 使い方

```bash
# タスク実行（provider は config/default.yaml の routing で自動選択）
ahc run "ベンゼンの HOMO/LUMO を計算して orbital_features.csv に保存してください"

# provider を明示切替
ahc run "..." --provider claude
ahc run "..." --provider openai --input ../data_smi.csv

# 実行中は動作過程（正規化イベント）が逐次表示される。--quiet で抑制
#   [14:11:11] reasoning_summary controller  開始 provider=deepagents task_type=orbital_calculation ...
#   [14:11:11] tool_call         claude      → calculate_orbitals {"smiles": ["O"], "basis": "sto-3g"}
#   [14:11:11] tool_result       claude      ✔ calculate_orbitals [success] 1/1 molecules → orbital_features.csv
#   [14:11:11] reasoning_summary verifier    検証合格: missing=0 warnings=0
ahc run "..." --quiet

# Skill の一覧・各SDK向け配置・バージョン固定
ahc skills list
ahc skills compile --provider all
ahc skills lock

# ベンチマーク（SDK横断比較 → benchmarks/results/*.json + routing 更新案）
ahc benchmark --provider deepagents --provider claude --label baseline
ahc benchmark --tag smoke

# 既存 workspace の再検証
ahc verify --workspace workspaces/run-xxxx --task-type orbital_calculation

# 自己改善ループ（開発モードのみ。改善候補は git diff として出力、自動適用しない）
ahc evolve analyze
ahc evolve propose
ahc evolve evaluate --proposal prop-xxxx
ahc evolve promote  --proposal prop-xxxx

# Web UI + API サーバ
ahc api --port 8000        # → http://127.0.0.1:8000/ をブラウザで開く
```

`pip install -e .` をしない場合は `python -m app.cli ...` でも同じ。

## Web インターフェース

`ahc api` で起動する Web コンソール（`pip install -e ".[api]"` が必要）:

- **タスク投稿** — リクエスト文・provider・task_type・期待出力を指定して実行。入力ファイル（CSV等）はブラウザからアップロードすると workspace へコピーされる
- **ラン監視** — 実行中は正規化イベント（`AgentEvent`）を自動ポーリングでライブ表示。検証結果（✅/❌/⚠️）も表示
- **成果物閲覧** — 画像はインライン表示、`report_user.html` は埋め込み表示、その他はダウンロードリンク
- REST API は `/api/*`（OpenAPI ドキュメントは `/docs`）。タスクはバックグラウンド実行され、`POST /api/tasks` は即座に `run_id` を返す

## 実行ループ

1. ユーザ要求 → `TaskSpec`（task_type 判定・期待出力・達成条件の設定）
2. `SkillRegistry.select()` が Skill を選択（task_type 対応 + 常時ロードの検証/回復/報告 Skill）
3. routing 設定（またはフォールバック）で Adapter を起動
4. Agent が共通ツール（`calculate_orbitals`, `cross_validate_model`, `run_python_sandbox` 等）で実行
5. `ScientificVerifier` が構造化判定（期待出力の存在 + HOMO<LUMO・値域・リーク検査など）
6. 不合格なら `required_repairs` を注入して再実行（`max_replans` まで）
7. 合格なら `workspaces/<run_id>/report.{json,md}` と manifest を保存

終了条件は文字列マッチではなく `VerificationResult` による構造化判定。全SDKのイベントは `AgentEvent` に正規化され `traces/<run_id>.jsonl` へ記録される。

## ツール実行環境の分離

依存が競合するツールは `sandbox.named_envs` でツール専用環境に分離できる。
`predict_reaction_t5`（ReactionT5v2 による収率/生成物/逆合成予測）は torch/transformers を
必要とするため、既定の `pyscf` 環境とは別の環境で実行される:

| 実行系 | 既定ツール群 | predict_reaction_t5 |
|---|---|---|
| local sandbox | conda env `pyscf` | conda env `reactiont5` |
| docker sandbox | `autoharnesschem/sandbox` | `autoharnesschem/reactiont5`（モデル焼き込み済み） |

harness 本体のプロセスに torch は不要 — ツールは自己完結スクリプトを生成して
専用環境の subprocess として実行し、結果を JSON/CSV で受け取る。

## Skill

正本は `skills/`（Agent Skills 標準の SKILL.md + YAML frontmatter）。`ahc skills compile` で各SDK向けに配置する:

| provider | 配置先 |
|---|---|
| deepagents | `build/skills/deepagents/`（backend へ mount） |
| claude | `.claude/skills/` への symlink |
| openai | `build/skills/openai/<name>-<version>.zip`（Skill Bundle） |

本番モード（`--config config/production.yaml`）では `skills.lock` との一致を強制し、自己改善は無効化される。

## 自己改善（Harness Evolver）

`traces/` の横断解析 → 失敗分類（analyzer） → SKILL.md への改善候補を unified diff で生成（proposer） → 一時コピー上でベンチマーク再実行（evaluator） → promotion gate 判定（promoter）。

- 変更可能: `skills/*/SKILL.md`・Skill付属スクリプト・instructions（`evolver/guard.py` が強制）
- 変更禁止: Benchmark 正解・Scientific Verifier・Security policy・評価指標・Evolver 自身
- 採用には gate 通過 **かつ** 人手レビュー（`git apply proposals/<id>/patch.diff` → `ahc skills lock`）が必要

## テスト

```bash
pytest            # SDK・rdkit 不要（スキーマ/Skill/Policy/Verifier/Evolver の単体テスト）
```
