# AutoHarnessChem

**同一の量子化学・機械学習タスクを、実行時設定だけで Deep Agents / Claude Agent SDK / OpenAI Agents SDK の3基盤に切り替えて実行できる Agentic Harness。**

SDK ごとに違うのは「エージェントの動かし方」だけで、Skill・ツール・サンドボックス・検証・ベンチマーク・自己改善といった資産は共通化されています。エージェントは自然言語のタスクを受け取り、計画 → ツール実行 → 科学的検証 → 不合格なら自動修復、というループを、構造化された合否判定に到達するまで自律的に回します。設計方針は [`../specification.md`](../specification.md) に基づきます。

## 目次

- [コンセプト](#コンセプト)
- [アーキテクチャ](#アーキテクチャ)
- [動作の仕組み（自律実行ループ）](#動作の仕組み自律実行ループ)
- [セットアップ](#セットアップ)
- [使い方（CLI）](#使い方cli)
- [Web インターフェース](#web-インターフェース)
- [共通ツール](#共通ツール)
- [Skill](#skill)
- [ツール実行環境の分離](#ツール実行環境の分離)
- [科学的検証（Scientific Verifier）](#科学的検証scientific-verifier)
- [動作過程の可視化（Trace）](#動作過程の可視化trace)
- [自己改善ループ（Harness Evolver）](#自己改善ループharness-evolver)
- [設定](#設定)
- [ディレクトリ構成](#ディレクトリ構成)
- [テスト](#テスト)

---

## コンセプト

3つのエージェント SDK は得意分野が異なります。この harness は、どの SDK を使うかを **設定（またはタスク種別による自動ルーティング）** で切り替えられるようにし、SDK 固有の差分を `adapters/` に閉じ込めます。

| 分離される「共通資産」 | 実体 |
|---|---|
| Skill（手順書） | `skills/*/SKILL.md`（Agent Skills 標準） |
| ツール | `tools/`（型付き共通ツール群） |
| サンドボックス | `tools/sandbox.py`（Docker / local conda） |
| 状態・成果物管理 | `harness/task_ledger.py`, `harness/artifacts.py` |
| 科学的検証 | `harness/verifier.py` |
| ベンチマーク | `benchmarks/` |
| 自己改善 | `evolver/` |

SDK 固有の実装（`adapters/deepagents.py`, `adapters/claude.py`, `adapters/openai.py`）は、すべて同じ `AgentRuntimeAdapter` インターフェースを満たし、各 SDK のイベントを共通の `AgentEvent` 形式へ正規化します。

---

## アーキテクチャ

```text
User / API / CLI
        │
        ▼
Task Interpreter          harness/controller.py (interpret_task)
        │   自然言語 → TaskSpec（task_type 判定・達成条件・期待出力）
        ▼
Common Harness Core       harness/
 ├─ Task Ledger           task_ledger.py     … 試行と状態遷移を ledger.json に永続化
 ├─ Skill Registry        skill_registry.py  … SKILL.md の読込 + 各SDKへの配置(Compiler)
 ├─ Tool Registry         tools/registry.py  … 共通ツールを name→spec で保持
 ├─ Policy Gate           policy.py          … コード/書込パス/危険操作の検査
 ├─ Artifact Manager      artifacts.py       … 成果物を manifest.json に記録
 ├─ Scientific Verifier   verifier.py        … 構造化された合否判定（Evolver変更禁止）
 └─ Trace Normalizer      traces.py          … 全SDKのイベントを AgentEvent に正規化
        │
        ▼
AgentRuntimeAdapter       adapters/{deepagents,claude,openai}.py
        │   provider を切替（未導入なら runtime_fallback で他providerへ）
        ▼
Sandbox / Python Executor tools/sandbox.py（Docker / local conda、ツール専用環境も可）
```

---

## 動作の仕組み（自律実行ループ）

`ahc run "..."` を実行すると、`HarnessController` が以下を順に行います（`harness/controller.py`）。

1. **タスク解釈** — 自然言語のリクエストを `TaskSpec` に変換。キーワードから `task_type`（例: `orbital_calculation`, `molecular_regression`, `reaction_prediction`）を判定し、期待出力ファイルと達成条件を補完する。
2. **Skill 選択** — `task_type` に対応する Skill に加え、常時ロードの検証・回復・報告 Skill を選ぶ。
3. **Adapter 起動** — routing 設定（またはフォールバック）で provider を決めて起動。
4. **実行** — エージェントが共通ツール（`calculate_orbitals` 等）や `run_python_sandbox` を使ってタスクを遂行。
5. **検証** — `ScientificVerifier` が構造化判定（期待出力の存在 + ドメイン検査: HOMO<LUMO、値域、リーク疑い等）。
6. **修復ループ** — 不合格なら `required_repairs`（不足・警告の具体的な直し方）を次の試行プロンプトに注入して再実行。`max_replans` 回まで。
7. **保存** — 合格したら `workspaces/<run_id>/` に `report.json` / `report.md` / `manifest.json` を保存。

> 終了条件は「特定の文字列が出たら終わり」ではなく、`VerificationResult`（passed / missing / warnings / repairs）による**構造化判定**です。これにより SDK に依存しない一貫した合否基準が得られます。

---

## セットアップ

```bash
cd AutoHernessChem
pip install -e ".[dev]"          # コア（pydantic + PyYAML）+ pytest

# 使いたい SDK を1つ以上追加
pip install -e ".[claude]"       # Claude Agent SDK
pip install -e ".[openai]"       # OpenAI Agents SDK
pip install -e ".[deepagents]"   # Deep Agents (LangGraph)

pip install -e ".[api]"          # Web UI / REST API（fastapi, uvicorn, python-multipart）

# sandbox イメージ（Docker を使う場合。無ければ local conda に自動フォールバック）
docker build -t autoharnesschem/sandbox:latest docker/
docker build -t autoharnesschem/reactiont5:latest -f docker/Dockerfile.reactiont5 docker/
```

**API キー**は各 SDK の流儀に従い環境変数で渡します（`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`、Web検索を使う場合は `TAVILY_API_KEY`）。

**科学計算の依存**（rdkit / pyscf / scikit-learn 等）は、local sandbox が使う conda 環境（既定 `pyscf`）側に入っている必要があります。harness 本体のプロセスには不要です。

---

## 使い方（CLI）

インストール後は `ahc`、未インストールでも `python -m app.cli` で同じことができます。

### タスク実行

```bash
# provider は config の routing で自動選択
ahc run "ベンゼンの HOMO/LUMO を計算して orbital_features.csv に保存してください"

# provider を明示、入力ファイルを workspace へコピー
ahc run "data_smi.csv から回帰モデルを交差検証して" --provider claude --input ../data_smi.csv

# 期待出力を明示（Verifier がその存在を必須要件にする）
ahc run "..." --expect orbital_features.csv --expect true_vs_pred.png
```

実行中は**動作過程が逐次表示**されます（`--quiet` で抑制）:

```text
[14:11:11] reasoning_summary controller  開始 provider=deepagents task_type=orbital_calculation skills=pyscf-orbitals,...
[14:11:11] reasoning_summary controller  試行 1/4
[14:11:11] tool_call         claude      → calculate_orbitals {"smiles": ["O"], "basis": "sto-3g"}
[14:11:11] tool_result       claude      ✔ calculate_orbitals [success] 1/1 molecules → orbital_features.csv
[14:11:11] reasoning_summary verifier    検証合格: missing=0 warnings=0
[14:11:11] reasoning_summary controller  終了 status=succeeded attempts=1
```

### Skill 管理

```bash
ahc skills list                      # 一覧（名前・バージョン・リスク・説明）
ahc skills compile --provider all    # 各SDK向けに配置（deepagents/claude/openai）
ahc skills lock                      # skills.lock を生成（本番用のバージョン固定）
ahc skills check                     # skills.lock との差分検査
```

### ベンチマーク

```bash
# SDK横断で比較 → benchmarks/results/*.json（成功率・実行時間 + routing 更新案）
ahc benchmark --provider deepagents --provider claude --label baseline
ahc benchmark --tag smoke            # タグで絞り込み
```

### 検証・自己改善・サーバ

```bash
ahc verify --workspace workspaces/run-xxxx --task-type orbital_calculation
ahc evolve analyze                   # trace の失敗分類 + リカバリ検出
ahc evolve propose                   # SKILL.md への改善候補を diff で出力（自動適用なし）
ahc api --port 8000                  # → http://127.0.0.1:8000/
```

---

## Web インターフェース

`ahc api`（要 `pip install -e ".[api]"`）で起動する Web コンソール。CLI と同じ機能をブラウザから使えます。

- **タスク投稿** — リクエスト文・provider・task_type・期待出力を指定して実行。入力ファイル（CSV 等）はブラウザからアップロードすると workspace へコピーされる。
- **ラン監視** — 実行中は正規化イベント（`AgentEvent`）を自動ポーリングでライブ表示。CLI と同じ粒度でフェーズ（試行 N/M・検証合否）やツール呼び出しが見える。
- **成果物閲覧** — 画像はインライン表示、`report_user.html` は埋め込み表示、その他はダウンロードリンク。
- **REST API** — `/api/*`（OpenAPI ドキュメントは `/docs`）。タスクはバックグラウンド実行され、`POST /api/tasks` は即座に `run_id` を返す。

---

## 共通ツール

エージェントに任意の shell を書かせるだけでなく、代表的な処理を型付きツール（入出力が `ToolResult` スキーマ）として提供します（`tools/`）。

| ツール | 役割 |
|---|---|
| `inspect_dataset` | CSV の行数・列型・欠損・統計量の確認 |
| `standardize_smiles` | RDKit による SMILES 正準化 |
| `generate_3d_structure` | ETKDGv3 + MMFF で 3D 構造生成（xyz 出力） |
| `calculate_orbitals` | RDKit+PySCF で HOMO/LUMO/gap を計算 |
| `calculate_rdkit_descriptors` | RDKit 記述子（MolWt, LogP, TPSA 等） |
| `cross_validate_model` | 特徴量から target を予測する回帰モデルの K-fold CV + 散布図 |
| `predict_reaction_t5` | ReactionT5v2 による収率／生成物／逆合成予測（専用環境で実行） |
| `inspect_artifact` | 生成物ファイルのサイズ・テキストプレビュー |
| `run_python_sandbox` | 任意 Python を sandbox（制限付き）で実行 |
| `verify_scientific_result` | Scientific Verifier をツールとして呼ぶ |
| `search_official_documentation` | 公式ドキュメント検索（要 `TAVILY_API_KEY`） |

各ツールは失敗時に `error_type`（`timeout` / `missing_dependency` / `scf_failed` / `invalid_smiles` 等）と `retryable` を返し、これがエージェントの回復判断と Evolver の分析に使われます。

---

## Skill

Skill は SDK 非依存の「手順書」です。正本は `skills/`（Agent Skills 標準の `SKILL.md` + YAML frontmatter）に置き、`SkillCompiler` が各 SDK 向けに配置します。

| provider | 配置先 |
|---|---|
| deepagents | `build/skills/deepagents/`（backend へ mount） |
| claude | `.claude/skills/` への symlink |
| openai | `build/skills/openai/<name>-<version>.zip`（Skill Bundle） |

各 `SKILL.md` は **Procedure（手順）/ Completion criteria（完了条件）/ Recovery procedure（回復手順）** を持ちます。同梱の Skill:

`dataset-inspection` · `rdkit-preparation` · `pyscf-orbitals` · `molecular-regression` · `reaction-prediction` · `scientific-verification`（常時）· `execution-recovery`（常時）· `result-reporting`（常時）

---

## ツール実行環境の分離

依存が競合するツールは `sandbox.named_envs` で**ツール専用環境**に分離できます。`predict_reaction_t5`（ReactionT5v2）は torch/transformers を必要とするため、既定の `pyscf` 環境とは別の環境で実行されます。

| 実行系 | 既定ツール群 | `predict_reaction_t5` |
|---|---|---|
| local sandbox | conda env `pyscf` | conda env `reactiont5` |
| docker sandbox | `autoharnesschem/sandbox` | `autoharnesschem/reactiont5`（モデル焼き込み済み） |

harness 本体のプロセスに torch は不要です。ツールは自己完結スクリプトを生成し、専用環境の subprocess として実行して、結果を JSON/CSV で受け取ります。分離したい他ツールも `config/default.yaml` の `sandbox.named_envs` にエントリを追加するだけで同様に扱えます。

---

## 科学的検証（Scientific Verifier）

`harness/verifier.py` が、報告の前に結果の妥当性を構造化判定します（`VerificationResult`）。

- **共通** — `TaskSpec.expected_outputs` のファイルが workspace に存在するか。
- **軌道計算** — HOMO < LUMO か、値が物理的に妥当な範囲（-60〜+30 eV）か、単位変換（Hartree/eV）の取り違えがないか。
- **回帰** — `r2 ≤ 1`、`r2 > 0.999` はリーク疑い、`r2 < -1` はモデル不成立の警告。
- **反応予測** — 収率が 0〜100% の範囲内か、予測 SMILES が空でないか。

不合格時の `required_repairs` はそのまま次の試行のプロンプトに注入されます。**Verifier は Evolver の変更禁止対象**（自己改善で評価基準を甘くさせない）です。

---

## 動作過程の可視化（Trace）

全 SDK のイベントは `AgentEvent`（`run_id` / `event_type` / `actor` / `payload` / `timestamp`）に正規化され、`traces/<run_id>.jsonl` に追記されます。`event_type` は `reasoning_summary` / `tool_call` / `tool_result` / `delegation` / `artifact` / `error` / `final`。

このイベント列が、CLI の逐次表示・Web のライブ表示・Evolver の失敗分析という**3つの用途で共有**されます。`TraceWriter.subscribe()` にリスナーを登録すると、記録と同時に受け取れます。

---

## 自己改善ループ（Harness Evolver）

`traces/` の横断解析から Skill の改善候補を作る開発用ループです（本番モードでは無効）。

```text
traces/ 解析（失敗分類 + リカバリ検出） →  SKILL.md への改善候補を diff 生成
   analyzer                                proposer
        →  一時コピー上でベンチマーク再実行  →  promotion gate 判定
           evaluator                            promoter
```

proposer は2段階で提案します:

1. **リカバリベース（優先）** — トレース内で「あるツールが失敗し、後続で同じツールが成功」した箇所を検出し、**どの引数をどう変えたら通ったか**（例: `basis: 6-31g* → sto-3g`）を抽出。そのツールを `required_tools` に持つ Skill の Recovery procedure へ、再現可能な具体手順として恒久化します。→ 次回以降そのリカバリを再発見せずに済む。
2. **汎用テンプレート（フォールバック）** — 具体的なリカバリが観測されなかった頻出失敗にのみ、一般的な予防ガイダンスを追記。同一 (Skill, カテゴリ) が 1 でカバー済みなら汎用提案は抑制します。

安全策:

- **変更可能** — `skills/*/SKILL.md`・付属スクリプト・instructions（`evolver/guard.py` が強制）。
- **変更禁止** — Benchmark の正解・Scientific Verifier・Security policy・評価指標・Evolver 自身。
- **採用条件** — promotion gate（成功率改善 ≥ 0.05、レイテンシ増 ≤ 20% 等）を通過し、**かつ人手レビュー**が必須。改善候補は自動適用せず `evolver/proposals/<id>/patch.diff` として出力されます。適用は `git apply` → `ahc skills lock`。

---

## 設定

`config/default.yaml` をベースに、`--config` で指定したファイルを深いマージで上書きします。

| キー | 意味 |
|---|---|
| `runtime.provider` | 既定 provider（`deepagents` / `claude` / `openai`） |
| `runtime.max_replans` | 検証不合格時の再実行回数の上限 |
| `runtime.sandbox` | `type`(docker/local)・`conda_env`・timeout・リソース制限・`named_envs` |
| `routing` | `task_type` → provider の対応（ベンチマーク実測から更新可能）+ `fallback` |
| `mode` | `development` / `production`（本番は自己改善を強制無効化） |
| `self_improvement` | Evolver の有効/無効 |
| `approval` | 高リスク操作の承認方式（`interactive` / `deny` / `allow`） |
| `runtime_fallback` | provider 未導入時に他 provider へ切り替えるか |

**本番実行**（`--config config/production.yaml`）では `mode: production` になり、自己改善が無効化され、`skills.lock` との一致が強制されます。

---

## ディレクトリ構成

```text
AutoHernessChem/
├── app/            CLI (cli.py) / REST API + Web UI (api.py, web/)
├── schemas/        TaskSpec / ToolResult / VerificationResult / AgentEvent など
├── harness/        Common Harness Core（controller, verifier, policy, ...）
├── tools/          共通ツール（chem, reactiont5）+ registry + sandbox
├── adapters/       AgentRuntimeAdapter 実装（base, deepagents, claude, openai）
├── skills/         Skill の正本（*/SKILL.md）
├── benchmarks/     ベンチマーク定義 (tasks.yaml) と runner
├── evolver/        自己改善（analyzer, proposer, evaluator, promoter, guard）
├── config/         default.yaml / production.yaml
├── docker/         sandbox イメージ（Dockerfile, Dockerfile.reactiont5）
├── tests/          単体テスト
├── traces/         実行イベントの記録（<run_id>.jsonl）
└── workspaces/     ラン毎の作業ディレクトリと成果物・report
```

---

## テスト

```bash
pytest            # SDK・rdkit・torch 不要（スタブで動作）
```

スキーマ / Skill / Policy / Verifier / Trace / Controller ループ / Evolver / Web API を対象とした単体テストが含まれます。SDK や重い依存が無い環境でも全て実行できます。
