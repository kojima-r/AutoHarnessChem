# ChemEval による ahc の評価

[ChemEval](https://openreview.net/forum?id=JrqjSkEPrX)（ICLR 2026, USTC / iFLYTEK）は
化学能力を **4 レベル / 13 能力次元 / 62 タスク**に分けて測るベンチマーク。ここではその
問題を **ahc（AutoHarnessChem）に解かせて**採点する。つまり素の LLM ではなく
「ツールを使えるエージェント + Verifier + 再計画ループ」としての実力を測る。

- 公式リポジトリ: `benchmarks_chemeval/ChemEval/`（別 git のクローン。**参照専用・改変しない**）
- データ: HuggingFace [`Ooo1/ChemEval`](https://huggingface.co/datasets/Ooo1/ChemEval)
  （text 3,880 問 = 53 タスク × 0-shot/3-shot、multimodal 1,240 問 = 54 タスク。
  公式リポジトリにはデータは含まれない）
- 実装: この README と同じディレクトリの python モジュール群（`__init__.py` に一覧）

`benchmarks/`（SDK 横断のタスク成功率ベンチマーク）とは別物で、あちらは
「成果物を作れるか」、こちらは「答えが化学的に正しいか」を測る。

## 使い方

```bash
# 0) 依存（harness 本体だけでよい。rdkit などは専用環境へ委譲する）
pip install -e ".[dev,claude]"

# 1) データ取得（HuggingFace の parquet → JSONL。初回のみ、text で約 2MB）
python -m benchmarks_chemeval.evaluate prepare
python -m benchmarks_chemeval.evaluate prepare --split multimodal   # 画像も展開する

# 2) タスク一覧（id / 採点方式 / ahc に渡す task_type）
python -m benchmarks_chemeval.evaluate tasks

# 3) 小さく試す（各タスク 1 問・0-shot・claude で実行 → 採点 → レポート）
python -m benchmarks_chemeval.evaluate run --label smoke \
    --provider claude --limit-per-task 1 --concurrency 4

# 4) レベルを絞って本番評価（自由記述は LLM-as-judge で採点）
python -m benchmarks_chemeval.evaluate run --label l4 \
    --level scientific_knowledge_deduction --limit-per-task 20 \
    --provider claude --concurrency 4 --judge auto

# 5) run を回さず採点だけやり直す（judge を後から足す / 指標を直したとき）
python -m benchmarks_chemeval.evaluate score --label l4 --judge auto
```

`ahc chemeval <サブコマンド> ...` でも同じものが動く（`ahc --config` は引き継がれる）。
まず `--dry-run` で対象問題数を確認するとよい。

### 主なオプション

| オプション | 意味 |
| --- | --- |
| `--split text\|multimodal` | text（既定）か画像つき multimodal か |
| `--task ID` / `--level` / `--dimension` / `--metric` | 出題範囲の絞り込み（複数指定可） |
| `--shot 0\|3` / `--all-shots` | 0-shot（既定）か 3-shot か |
| `--limit-per-task N` | タスクごとの出題数（既定 5、`0` で全件。seed 固定サンプリング） |
| `--provider` | 複数指定で SDK 横断比較（`benchmarks/` と同じ発想） |
| `--concurrency N` | 同時実行数（1 問 = 1 run） |
| `--item-timeout SEC` | 1 問の実時間上限。到達したら延長せず打ち切って次へ進む |
| `--cleanup-workspaces` | 採点後に workspace を消す（数千問を回すとき） |
| `--judge auto\|claude\|anthropic\|none` | 自由記述タスクの LLM 採点（既定 none） |
| `--no-chem` | rdkit による分子系採点をやめる（文字列一致に退避） |
| `--overwrite` | 記録済みの問題も再実行する（既定は未実行分だけ追加 = 再開） |

## 出力

`benchmarks_chemeval/results/<label>/`

| ファイル | 内容 |
| --- | --- |
| `records.jsonl` | 1 行 1 問の実行ログ（query / target / answer / run_id / 所要時間）。**再開に使う** |
| `scored.jsonl` | 採点済みレコード（score / metrics / 判定の詳細） |
| `metrics.json` | provider → レベル → 次元 → タスクの集計 |
| `report.md` | 上記の Markdown レポート |
| `judge_cache.json` | LLM 採点のキャッシュ（再採点で API を呼び直さない） |

個々の実行の詳細（ツール呼び出し・失敗理由）は通常の run と同じく
`workspaces/chemeval-<label>-<provider>-<item_id>/` と `traces/` に残る。

## 動作の流れ

1. **出題** — ChemEval の `query` を**原文のまま**渡し、答えの保存先だけを指示する
   （workspace 直下の `chemeval_answer.json` に `{"answer": ...}`）。
   問題文自体が出力形式を指定しているので、それに従わせる。
2. **実行** — `HarnessController.run(task_type=<カタログの ahc_task_type>,
   expected_outputs=["chemeval_answer.json"])`。task_type で Skill とツールが決まるため、
   HOMO/LUMO は量子化学計算（pyscf / OptTDDFT）、生成物・前駆体予測は ReactionT5、
   多段合成経路は AiZynthFinder を**実際に使って**答えられる。
   答えファイルが無ければ Verifier が不合格にし、`--max-replans` 回まで再試行する。
3. **抽出** — `chemeval_answer.json` → 最終メッセージ中の JSON → 最終メッセージ全文の
   順に答えを探す（どこから取れたかは `answer_source` に残る）。
4. **採点** — タスクごとの指標（下表）。分子構造の正準化と Tanimoto は
   `tools/envrun.py` 経由で `rdkit` 専用環境の subprocess として計算する
   （harness 本体に rdkit を入れないため）。自由記述は LLM-as-judge。
5. **集計** — レベル / 次元 / タスク別に平均。ahc 固有の観測値（Verifier 合格率・
   試行回数・所要時間・答えの取得元）も併記する。

## 採点方式

`tasks.yaml` の `metric` 列。公式 `Textual/code evaluate/*.py` の定義に合わせて
`metrics.py` で純 python 再実装している（対応関係は `metrics.py` の docstring）。

| metric | 内容 | 代表タスク |
| --- | --- | --- |
| `choice` / `true_false` / `yes_no` | 選択肢・正誤・Yes/No の一致 | 選択問題、判断問題、BBBP/ClinTox/HIV |
| `contains` | 単一ラベルの包含一致 | 文献トピック分類、反応種別 |
| `entity_f1` / `relation_f1` / `reagent_f1` | 集合 F1（実体・関係・試薬） | NER、情報抽出、試薬/溶媒/配位子推薦 |
| `sider` | 20 ラベルの平均一致率 + 完全一致率 | SIDER |
| `regression` | RMSE / MAE（範囲回答は中央値） | ESOL、Lipo、HOMO/LUMO、融点、沸点、温度・時間推薦 |
| `range_overlap` | 範囲の重なり / 和 | 活性化エネルギーの範囲 |
| `smiles` | 正準 SMILES 完全一致 + Tanimoto + 妥当性 | 生成物予測、前駆体推薦、IUPAC→SMILES |
| `formula` | 原子組成の一致 + cos / L1 / L2 類似度 | SMILES→分子式、IUPAC→分子式 |
| `iupac` | 小文字一致 + BLEU-4 + 編集距離 | SMILES→IUPAC |
| `selfies` | 正準化後の一致 | SMILES ⇄ SELFIES |
| `text_exact` | 記法を揃えた完全一致 | 分子式・反応式画像 → LaTeX（multimodal） |
| `reaction_smiles` | 反応 SMILES を役割ごとに正準化して集合一致 | 反応スキーム画像 → 反応 SMILES（multimodal） |
| `judge` | LLM-as-judge（0..1） | 空所補充、短答、計算、要約・提綱生成、合成経路、反応中間体、スペクトル解釈 |

レポートの `score` は「0..1 で高いほど良い代表値」。**回帰タスクは尺度が違うので
`score` には混ぜず**、`rmse` / `mae` を主指標として別に出す。`judge` 系は
`--judge` を付けない限り採点されず（`score=None`）、集計から外れる。

## multimodal split

`--split multimodal` で画像つきの 54 タスク（1,240 問）を評価できる。

- `prepare --split multimodal` が parquet（約 64MB）から画像を `data/images/` に展開する。
- 各問題の画像は run のときに workspace へコピーされ、プロンプトでファイル名を示す
  （エージェントは Read で画像を読む。画像を読めない provider では成績が落ちる）。
- multimodal 側はデータの `filename` が空で、タスク名が `file_path` 列に入っているため、
  カタログは `file_path` を key として引く（`mm_` 接頭の id）。
- 3-shot は multimodal には無い（すべて 0-shot 扱い）。

## 注意

- **正解と採点方式は Evolver の変更禁止対象**。`benchmarks/tasks.yaml` と同じ扱いで、
  `tasks.yaml`・`metrics.py`・`score.py` を自己改善で緩めないこと。
- ChemEval のライセンスは CC BY-NC-SA 4.0（非商用）。`benchmarks_chemeval/ChemEval/` と
  `data/` は親リポジトリにコミットしない（`.gitignore` 済み）。
- 1 問 = 1 エージェント実行なので、全 3,880 問を回すと SDK の API コストと時間が
  大きい。`--limit-per-task` で規模を決め、`--concurrency` で並列度を上げ、
  途中で止めても同じ `--label` で再開する運用を前提にしている。
- parquet の読み出しには pyarrow が必要。harness 本体の環境に無い場合は
  pyarrow を持つ python（conda 環境など）を自動で探す。見つからないときは
  `CHEMEVAL_PYTHON=/path/to/python` で指定するか、`data/text.jsonl` を自分で用意する
  （`query` / `target` / `filename` を持つ JSON Lines であればよい）。
- `selfies` パッケージが専用環境に無い場合、SMILES⇄SELFIES は文字列一致で採点する
  （SELFIES を SMILES として読むと別分子になるため、意図的に退避している）。
