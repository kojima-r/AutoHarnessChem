# AutoHarnessChem

**同一の量子化学・機械学習タスクを、実行時設定だけで Deep Agents / Claude Agent SDK / OpenAI Agents SDK の3基盤に切り替えて実行できる Agentic Harness。**

SDK ごとに違うのは「エージェントの動かし方」だけで、Skill・ツール・サンドボックス・検証・ベンチマーク・自己改善といった資産は共通化されています。エージェントは自然言語のタスクを受け取り、計画 → ツール実行 → 科学的検証 → 不合格なら自動修復、というループを、構造化された合否判定に到達するまで自律的に回します。

## 目次

- [コンセプト](#コンセプト)
- [アーキテクチャ](#アーキテクチャ)
- [動作の仕組み（自律実行ループ）](#動作の仕組み自律実行ループ)
- [セットアップ](#セットアップ)
- [使い方（CLI）](#使い方cli)
- [Web インターフェース](#web-インターフェース)
- [構造式の可視化（SmilesDrawer）](#構造式の可視化smilesdrawer)
- [共通ツール](#共通ツール)
- [デモ（examples/）](#デモ各-conda-環境のライブラリを直接使うサンプル)
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
| ツール | `tools/`（型付き共通ツール群 + `tools/OptTDDFT` の量子化学エンジン） |
| サンドボックス | `tools/sandbox.py`（Docker / local conda） |
| 状態・成果物管理 | `harness/task_ledger.py`, `harness/artifacts.py` |
| 科学的検証 | `harness/verifier.py` |
| ベンチマーク | `benchmarks/` |
| 自己改善 | `evolver/` |

SDK 固有の実装（`adapters/deepagents.py`, `adapters/claude.py`, `adapters/openai.py`）は、すべて同じ `AgentRuntimeAdapter` インターフェースを満たし、各 SDK のイベントを共通の `AgentEvent` 形式へ正規化します。

### 設計上の原則

この harness が一貫して守っている方針です。

- **SDK 非依存を徹底する** — Skill・ツール・検証・ベンチマークは特定 SDK に依存させず、SDK 固有処理は `adapters/` にのみ置く。
- **終了条件を構造化する** — 「特定の文字列が出たら完了」ではなく、`VerificationResult` による合否判定で終える。
- **科学的妥当性を検証する** — 実行が成功しただけでは合格とせず、単位・値域・リークなどドメイン的な妥当性まで検査する。
- **自己改善を安全に制御する** — 改善は Skill と instructions に限り、評価基準（Verifier・ベンチマーク正解・評価指標）と安全設定は変更禁止。改善候補は自動適用せず diff として出力し、ゲート通過と人手承認を経て採用する。
- **本番では固定する** — 本番モードでは自己改善を無効化し、検証済み Skill バージョンを `skills.lock` で固定する。

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

1. **タスク解釈** — 自然言語のリクエストを `TaskSpec` に変換。キーワードから `task_type`（`orbital_calculation` / `molecular_design` / `pes_scan` / `molecular_regression` / `reaction_prediction` / `retrosynthesis_planning` / `dataset_analysis` ほか）を判定し、期待出力ファイルと達成条件を補完する。
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
docker build -t autoharnesschem/opttddft:latest -f docker/Dockerfile.opttddft .
docker build -t autoharnesschem/reactiont5:latest -f docker/Dockerfile.reactiont5 docker/
docker build -t autoharnesschem/aizynth:latest -f docker/Dockerfile.aizynth docker/
```

**API キー**は各 SDK の流儀に従い環境変数で渡します（`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`、Web検索を使う場合は `TAVILY_API_KEY`）。

### ツール専用 conda 環境（local sandbox を使う場合）

harness 本体のプロセスに重い依存は不要です。ツールごとに以下の環境が使われます。

```bash
# 1. 量子化学（calculate_orbitals / calculate_tddft_spectrum /
#              optimize_absorption_wavelength / scan_esipt_pes）
conda create -n pyscf python=3.12 && conda activate pyscf
pip install -e tools/OptTDDFT          # opt_tddft（pyscf 2.11 / numpy 1.26.4 / rdkit / optuna）
pip install geometric                  # 構造最適化・ESIPT スキャンを使う場合（任意）

# 2. 反応予測（predict_reaction_t5）
conda create -n reactiont5 python=3.11 && conda activate reactiont5
pip install torch transformers sentencepiece

# 3. 逆合成経路探索（plan_retrosynthesis）
conda create -n aizynth python=3.11 && conda activate aizynth
pip install "aizynthfinder[all]"
download_public_data /path/to/aizynth_data          # 学習済み policy + stock（~1GB）
export AIZYNTH_CONFIG=/path/to/aizynth_data/config.yml
```

`AIZYNTH_CONFIG` を設定しない場合は `$AIZYNTH_DATA/config.yml` →
`data/aizynth/config.yml`（DL 先への symlink でよい: `ln -s /path/to/aizynth_data data/aizynth`）→
`~/aizynth_data/config.yml` の順に探索し、見つからなければ
`error_type=model_unavailable` として DL コマンドを案内します。

環境名は `config/default.yaml` の `sandbox.named_envs`（`opttddft` / `reactiont5` / `aizynth`）で変更できます。

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
[14:11:11] tool_result       claude      ✔ calculate_orbitals [success] HOMO/LUMO を 1/1 分子で計算 (HF/sto-3g) → orbital_features.csv
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

### デモ（各 conda 環境のライブラリを直接使うサンプル）

```bash
ahc demo                             # 3環境すべて（pyscf → aizynth → reactiont5）
ahc demo --env pyscf                 # 1つだけ
ahc demo --env pyscf --arg=--nstates --arg=10   # デモスクリプトへ引数を渡す
bash examples/run_examples.sh pyscf   # CLI を使わない同等の実行
```

`examples/` の 3 つのスクリプトは harness を経由せず、それぞれの環境のライブラリ
（`opt_tddft` / `transformers` / `aizynthfinder`）を直接使う実行可能な成果物です。
環境構築の動作確認にも使えます。詳細は [examples/README.md](examples/README.md)。

| スクリプト | env | 内容 |
|---|---|---|
| `pyscf_opt_tddft_demo.py` | `pyscf` | 骨格組み立て → DFT/TDDFT → スペクトル画像 + CSV/JSON |
| `reactiont5_demo.py` | `reactiont5` | forward / retrosynthesis / yield の 3 モデル推論 |
| `aizynth_demo.py` | `aizynth` | 逆合成経路探索（経路木・段数・出発物質を出力） |

---

## Web インターフェース

`ahc api`（要 `pip install -e ".[api]"`）で起動する Web コンソール。CLI と同じ機能をブラウザから使えます。

- **タスク投稿** — リクエスト文・provider・task_type・期待出力を指定して実行。入力ファイル（CSV 等）はブラウザからアップロードすると workspace へコピーされる。
- **構造アシスト（プロンプト入力の補助）** — SMILES を入れるとその場で構造を描画して妥当性を確認でき、プリセット分子とタスクテンプレートからプロンプト・task_type・期待出力を一括で作れる。詳細は下記。
- **ラン監視** — 実行中は正規化イベント（`AgentEvent`）を自動ポーリングでライブ表示。CLI と同じ粒度でフェーズ（試行 N/M・検証合否）やツール呼び出しが見える。
- **成果物閲覧** — 画像はインライン表示、`report_user.html` は埋め込み表示、その他はダウンロードリンク。成果物 CSV の SMILES は構造式として描画される。
- **REST API** — `/api/*`（OpenAPI ドキュメントは `/docs`）。タスクはバックグラウンド実行され、`POST /api/tasks` は即座に `run_id` を返す。

---

## 構造式の可視化（SmilesDrawer）

化合物・反応を SMILES 文字列のままにせず、**Web UI と HTML レポートの両方で構造式として描画**します。描画には [SmilesDrawer 2.x](https://github.com/reymond-group/smilesDrawer)（MIT）を使い、`app/web/vendor/smiles-drawer.min.js` として同梱しています（CDN 非依存 = オフラインでも動く）。

### Web UI

| 場所 | 内容 |
|---|---|
| 構造アシスト | SMILES / 反応 SMILES を入力すると即座に描画。解釈できない場合は赤字で通知するので、**投げる前に構造を確認**できる |
| プリセット | ベンゼン・アスピリン・パラセタモール・カフェイン・硫酸イオン・アセチル化反応などをクリックで入力 |
| タスクテンプレート | 「HOMO/LUMO 計算」「TDDFT 吸収スペクトル」「逆合成経路」「生成物予測」「収率予測」「分子設計」「ESIPT PES スキャン」「回帰」。クリックすると**プロンプト文・task_type・期待出力**が入力欄に揃う（入力中の SMILES が埋め込まれる） |
| プロンプト内の構造 | リクエスト欄に書かれた SMILES を自動検出してサムネイル表示（`CC(=O)Nc1ccc(O)cc1` のような括弧を含む SMILES も分断しない。解釈できたものだけ表示） |
| ラン詳細の「構造」 | 成果物 CSV（`orbital_features.csv` / `tddft_spectrum.csv` / `reactiont5_predictions.csv` / `retrosynthesis_routes.csv` 等）と trace のツール引数から構造を描画。ReactionT5 の入力は `反応物>試薬>生成物`、逆合成経路は `前駆体>>目標分子` として**反応式**で表示 |

### HTML レポート

`render_report_html` ツールが `report_user.md` を変換し、SmilesDrawer を HTML に埋め込みます（オフライン可）。Markdown 側の記法:

````markdown
```smiles
CC(=O)Nc1ccc(O)cc1                               パラセタモール
CC(=O)OC(C)=O.Nc1ccc(O)cc1>>CC(=O)Nc1ccc(O)cc1   アセチル化（`>` を含めば反応式）
```

文中では `smiles:CCO` と書くと小さな構造チップになります。
````

- 1 行 = `SMILES<空白>ラベル`（ラベル省略可）。SMILES 自体に空白は入れない。
- `auto_structures`（既定 true）で、workspace の CSV の SMILES 列から「構造一覧」を自動追加。
- 描画できない文字列は消さずにテキストとして残すので、JS を切っても内容は読めます。

---

## 共通ツール

エージェントに任意の shell を書かせるだけでなく、代表的な処理を型付きツール（入出力が `ToolResult` スキーマ）として提供します（`tools/`）。

| ツール | 役割 |
|---|---|
| `inspect_dataset` | CSV の行数・列型・欠損・統計量の確認 |
| `standardize_smiles` | RDKit による SMILES 正準化 |
| `generate_3d_structure` | ETKDGv3 + MMFF で 3D 構造生成（xyz 出力） |
| `calculate_orbitals` | **OptTDDFT(PySCF)** で HOMO/LUMO/gap・全エネルギーを計算（SCF のみ） |
| `calculate_tddft_spectrum` | **OptTDDFT** の TDDFT で励起波長・振動子強度・UV-Vis スペクトル画像 |
| `optimize_absorption_wavelength` | **OptTDDFT + Optuna** で目標吸収波長に近い分子を探索（MI） |
| `scan_esipt_pes` | **OptTDDFT** の ESIPT Relaxed PES スキャン（S0/S1 曲面 + 障壁） |
| `calculate_rdkit_descriptors` | RDKit 記述子（MolWt, LogP, TPSA 等） |
| `cross_validate_model` | 特徴量から target を予測する回帰モデルの K-fold CV + 散布図 |
| `predict_reaction_t5` | ReactionT5v2 による収率／生成物／**1 段階**逆合成予測（専用環境で実行） |
| `plan_retrosynthesis` | **AiZynthFinder** による**多段**の逆合成経路探索（専用環境で実行） |
| `inspect_artifact` | 生成物ファイルのサイズ・テキストプレビュー |
| `render_report_html` | `report_user.md` → **構造式付き** `report_user.html`（SmilesDrawer 埋め込み） |
| `run_python_sandbox` | 任意 Python を sandbox（制限付き）で実行 |
| `verify_scientific_result` | Scientific Verifier をツールとして呼ぶ |
| `search_official_documentation` | 公式ドキュメント検索（要 `TAVILY_API_KEY`） |

各ツールは失敗時に `error_type`（`timeout` / `missing_dependency` / `missing_environment` / `out_of_memory` / `scf_failed` / `no_valid_trial` / `model_unavailable` / `no_route_found` 等）と `retryable` を返し、これがエージェントの回復判断と Evolver の分析に使われます。

### 量子化学エンジン（OptTDDFT）

量子化学系のツールは `tools/OptTDDFT`（`opt_tddft` パッケージ）を実体とし、harness 本体の
プロセスではなく**専用の conda 環境 `pyscf` の subprocess** として実行されます
（`tools/opttddft.py` が自己完結スクリプトを生成 → JSON で結果を受け取る）。
以前 harness 内で直接 pyscf を叩いていた `calculate_orbitals` はこの実装に置き換わり、
OptTDDFT の構造生成（多コンフォマー探索 + UFF）・`SolverConfig`・TDDFT/TDA フォールバック・
PCM 溶媒・Optuna 探索・レポート生成をそのまま再利用します。ツール側で追加しているのは
**電荷/スピンの明示指定**と**実行上限（`timeout_sec` / `threads` / メモリ）の制御**だけです。

- 各ツールは `timeout_sec` を個別に指定できます（既定は sandbox の `timeout_sec`）。
  TDDFT や Optuna 探索は既定の 600s では終わらないことが多いため、ここを上げて使います。
- `threads`（既定 4）で PySCF のスレッド数を制限します（全コア占有の防止）。
- 構造最適化（`use_geom_opt=true` / `scan_esipt_pes`）には **geomeTRIC** が必要です。
  `pyscf` 環境に無い場合は `error_type=missing_dependency` として報告されます
  （`conda run -n pyscf pip install geometric`。numpy は 1.26.4 に固定したまま入れること）。

### 逆合成: 1 段階予測と経路探索の使い分け

| | ツール | 出力 | 用途 |
|---|---|---|---|
| 1 段階 | `predict_reaction_t5(task="retrosynthesis")` | 前駆体 SMILES 候補 | 速い。単一反応の候補列挙 |
| 多段 | `plan_retrosynthesis` | 経路木・段数・出発物質・solved | 購入可能物質まで遡った合成計画 |

`plan_retrosynthesis` は AiZynthFinder の expansion policy（USPTO テンプレート）と stock を使い、
MCTS または Retro* で経路を探索します。学習済みモデルは別途ダウンロードが必要です（下記セットアップ）。

---

## Skill

Skill は SDK 非依存の「手順書」です。正本は `skills/`（Agent Skills 標準の `SKILL.md` + YAML frontmatter）に置き、`SkillCompiler` が各 SDK 向けに配置します。

| provider | 配置先 |
|---|---|
| deepagents | `build/skills/deepagents/`（backend へ mount） |
| claude | `.claude/skills/` への symlink |
| openai | `build/skills/openai/<name>-<version>.zip`（Skill Bundle） |

各 `SKILL.md` は **Procedure（手順）/ Completion criteria（完了条件）/ Recovery procedure（回復手順）** を持ちます。同梱の Skill:

`dataset-inspection` · `rdkit-preparation` · `pyscf-orbitals`（HOMO/LUMO + TDDFT）· `tddft-molecular-design`（Optuna 探索）· `esipt-pes-scan` · `molecular-regression` · `reaction-prediction` · `aizynth-retrosynthesis` · `scientific-verification`（常時）· `execution-recovery`（常時）· `result-reporting`（常時）

---

## ツール実行環境の分離

依存が競合するツールは `sandbox.named_envs` で**ツール専用環境**に分離します。量子化学（pyscf + numpy 1.26.4 固定）・ReactionT5（torch/transformers）・AiZynthFinder（onnxruntime + 学習済みモデル）は互いに依存が衝突するため、それぞれ別環境で実行されます。

| ツール | local sandbox | docker sandbox |
|---|---|---|
| `run_python_sandbox` ほか既定 | conda env `pyscf` | `autoharnesschem/sandbox` |
| `calculate_orbitals` / `calculate_tddft_spectrum` / `optimize_absorption_wavelength` / `scan_esipt_pes` | conda env `pyscf`（`opt_tddft`） | `autoharnesschem/opttddft`（OptTDDFT 同梱） |
| `predict_reaction_t5` | conda env `reactiont5` | `autoharnesschem/reactiont5`（モデル焼き込み済み） |
| `plan_retrosynthesis` | conda env `aizynth` | `autoharnesschem/aizynth`（policy/stock 焼き込み済み） |

harness 本体のプロセスに pyscf も torch も aizynthfinder も不要です。ツールは自己完結スクリプトを生成し（`tools/envrun.py` が共通化）、専用環境の subprocess として実行して、結果を JSON/CSV で受け取ります。実行上限（timeout / CPU / メモリ）は呼び出しごとに差し替えられ、`SIGKILL` は `out_of_memory` として分類されます。分離したい他ツールも `sandbox.named_envs` にエントリを追加するだけで同様に扱えます。

> docker sandbox は `network=none` で動くため、モデルは image に焼き込みます（実行時ダウンロード不可）。AiZynthFinder の config は image 内の `AIZYNTH_CONFIG` が使われます。

---

## 科学的検証（Scientific Verifier）

`harness/verifier.py` が、報告の前に結果の妥当性を構造化判定します（`VerificationResult`）。

- **共通** — `TaskSpec.expected_outputs` のファイルが workspace に存在するか。
- **軌道計算** — HOMO < LUMO か、値が物理的に妥当な範囲（-60〜+30 eV）か、単位変換（Hartree/eV）の取り違えがないか。TDDFT では励起波長が 50〜2000 nm か（`1240/E` の変換ミス検出）、振動子強度が非負か。
- **分子設計** — 有効な trial が 1 件以上あるか、`best_smiles` が空でないか、目的関数値が `|波長 - 目標|` と整合しているか。
- **PES スキャン** — 3 点以上あるか、距離が単調増加か、**全点で S1 > S0**（励起状態が基底状態より高い）か。
- **回帰** — `r2 ≤ 1`、`r2 > 0.999` はリーク疑い、`r2 < -1` はモデル不成立の警告。
- **反応予測** — 収率が 0〜100% の範囲内か、予測 SMILES が空でないか。
- **逆合成経路** — 経路が抽出できているか、`solved` な経路の全前駆体が stock にあるか（未解決は警告）。

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
│   └── web/vendor/   同梱フロントエンドライブラリ（SmilesDrawer 2.x, MIT）
├── schemas/        TaskSpec / ToolResult / VerificationResult / AgentEvent など
├── harness/        Common Harness Core（controller, verifier, policy, ...）
├── tools/          共通ツール + registry + sandbox
│   ├── chem.py       RDKit / pandas / sklearn 系ツール
│   ├── report.py     Markdown → 構造式付き HTML レポート
│   ├── opttddft.py   OptTDDFT を pyscf 環境で実行する量子化学ツール
│   ├── reactiont5.py ReactionT5v2（reactiont5 環境）
│   ├── aizynth.py    AiZynthFinder 逆合成経路探索（aizynth 環境）
│   ├── envrun.py     ツール専用環境でのスクリプト実行の共通層
│   └── OptTDDFT/     TDDFT MI パイプライン本体（opt_tddft パッケージ）
├── examples/       各 conda 環境のライブラリを直接使う実行可能サンプル
├── adapters/       AgentRuntimeAdapter 実装（base, deepagents, claude, openai）
├── skills/         Skill の正本（*/SKILL.md）
├── benchmarks/     ベンチマーク定義 (tasks.yaml) と runner
├── evolver/        自己改善（analyzer, proposer, evaluator, promoter, guard）
├── config/         default.yaml / production.yaml
├── data/           学習済みモデル置き場（git 管理外。例: data/aizynth → DL 先の symlink）
├── docker/         sandbox イメージ（Dockerfile{,.opttddft,.reactiont5,.aizynth}）
├── tests/          単体テスト
├── traces/         実行イベントの記録（<run_id>.jsonl）
└── workspaces/     ラン毎の作業ディレクトリと成果物・report
```

---

## テスト

```bash
pytest                              # SDK・rdkit・pyscf・torch・aizynthfinder 不要（スタブで動作）

npm install --no-save jsdom         # 任意: ブラウザ側の構造描画テストを有効化
pytest tests/test_web_structures.py # jsdom が無ければ skip される
```

スキーマ / Skill / Policy / Verifier / Trace / Controller ループ / Evolver / Web API / 専用環境ツール（OptTDDFT・ReactionT5・AiZynthFinder）/ HTML レポート / examples を対象とした単体テストが含まれます。構造式描画については、node + jsdom があれば `tests/js/check_structures.js` が実際に DOM を動かし、**レポートの全構造が描画されること・反応式が描けること・プロンプト内 SMILES を壊さず検出すること**まで確認します。専用環境ツールのテストは sandbox をスタブ化し、**生成されるスクリプトの構文・渡される設定・失敗分類・成果物の登録**を検証するので、SDK や重い依存・学習済みモデルが無い環境でも全て実行できます。
