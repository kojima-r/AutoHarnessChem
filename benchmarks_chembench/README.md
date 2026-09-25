# ChemBench による ahc の評価

[ChemBench](https://github.com/lamalab-org/chembench)（*Are large language models
superhuman chemists?*, arXiv:2404.01475, Jablonka ら / LamaLab）は化学・材料の能力を
**9 トピック / 2,788 問**で測る人手作成ベンチマーク。ここではその問題を
**ahc（AutoHarnessChem）に解かせて**採点し、**harness の有無**と**文献値**の 3 者で比べる。

- 公式リポジトリ: `benchmarks_chembench/chembench/`（別 git のクローン。**参照専用・改変しない**）
- データ: 公式クローンに同梱の公開 report から読む（**ダウンロード不要**）。
  HuggingFace [`jablonkagroup/ChemBench`](https://huggingface.co/datasets/jablonkagroup/ChemBench)
  は照合にだけ使う
- 実装: この README と同じディレクトリの python モジュール群（`__init__.py` に一覧）

`benchmarks_chemeval/`（ChemEval 版）と対になる評価。**違いは文献値の質**で、
ChemBench は問題ごとの正誤を 37 手法ぶん公開しているため、
ahc が解いた問題だけに母集団を絞った「同一問題での比較」ができる。

## 使い方

```bash
# 0) 依存（harness 本体だけでよい。rdkit などは専用環境へ委譲する）
pip install -e ".[dev,claude]"

# 1) トピックと問題数を確認（データ取得は不要）
python -m benchmarks_chembench.evaluate topics

# 2) 採点が公式 ChemBench と一致することを確認（公開 report で照合）
python -m benchmarks_chembench.evaluate validate

# 3) harness あり（ahc）で解かせる → 採点 → レポート
python -m benchmarks_chembench.evaluate run --label smoke \
    --provider claude --limit-per-topic 4 --concurrency 4

# 4) harness なし（素の LLM・ツールなし 1 往復）で同じ問題を解かせる
python -m benchmarks_chembench.evaluate run --label smoke-bare --bare \
    --limit-per-topic 4 --concurrency 4

# 5) 3 者（ahc / 素の LLM / 文献 37 手法）を並べたレポートを作る
python -m benchmarks_chembench.evaluate score --label smoke --compare-label smoke-bare
```

`ahc chembench <サブコマンド> ...` でも同じものが動く（`ahc --config` は引き継がれる）。
まず `--dry-run` で対象問題数を確認するとよい。

**同じ `--limit-per-topic` と `--seed` を使えば harness あり / なしで同じ問題が選ばれる**
（トピックごとに seed 固定でサンプリングするため）。比較の前提なので変えないこと。

### 文献値と同じ設定で回す（`--limit-per-topic 0`）

文献値は**全 2,788 問**での成績として公表されている。`--limit-per-topic N` は
9 トピックから**均等に**取るので、N を指定した run は実際のベンチマークの構成比
（`chemical_preference` 1,001 問 / `toxicity_and_safety` 675 問で全体の 6 割）とは
重みが違う。公表値と横並びにするには **`--limit-per-topic 0`（全件）**で回す。

```bash
# 素の LLM（安いので先に完走させる。これが揃うと harness 側の途中経過も比較できる）
python -m benchmarks_chembench.evaluate run --label full-bare --bare \
    --model claude-opus-5 --limit-per-topic 0 --concurrency 6 --no-score

# harness あり（枠をまたぐので retry-failed 付きで何度も呼ぶ）
python -m benchmarks_chembench.evaluate run --label full-harness \
    --provider claude --limit-per-topic 0 --concurrency 6 \
    --retry-failed --no-score

# 途中でも採点してレポートを出せる（同一問題に絞った 3 者比較になる）
python -m benchmarks_chembench.evaluate score --label full-harness \
    --compare-label full-bare
```

全件は **2,788 問 × 2 = 5,576 実行**で、ChemEval 側の実測（セッション枠約 160 問 /
週次枠約 1,470 問）からすると**週次枠 数回ぶん**。1 セッションでは終わらない前提で、
同じ `--label` で何度も呼んで積み増す運用になる。

**枠切れをまたぐときは `--retry-failed` を必ず付ける。** SDK は利用枠切れを例外では
なく `You've hit your session limit ...` という**本文**で返すことがあり、そのまま
記録すると「答えの無い実行済みレコード」が残って `load_done` が永久に飛ばす
（ChemEval の運用で実際に踏んだ罠）。`--retry-failed` は再開前に
**「答えが取れていない record」を落として再実行対象に戻す**（判定はエラー文言では
なく answer の有無で行う）。同じ問題を無限に回さないよう、落とした回数を
`results/<label>/purge_counts.json` に持ち越し、`--max-retries`（既定 3）を超えたものは
確定失敗として残して前へ進む。単体で呼ぶなら `evaluate purge --label <label>`。

### 主なオプション

| オプション | 意味 |
| --- | --- |
| `--topic ID` | トピックで絞る（複数指定可） |
| `--metric-kind mcq\|numeric` | 選択肢問題だけ / 数値問題だけ |
| `--requires Calculation` | 要求能力で絞る（harness が効く所を狙い撃ちできる） |
| `--difficulty difficulty-basic` | 難易度で絞る |
| `--human-subset` | 人間が解いた部分集合（236 問）だけを対象にする |
| `--limit-per-topic N` | トピックごとの出題数（既定 5）。**`0` で全件 = 文献値と同じ設定** |
| `--retry-failed` | 再開前に「答えが取れていない record」を再実行対象に戻す（長期実行では必須） |
| `--max-retries N` | `--retry-failed` で同じ問題を再実行する上限（既定 3） |
| `--provider` | 複数指定で SDK 横断比較 |
| `--concurrency N` | 同時実行数（1 問 = 1 run） |
| `--item-timeout SEC` | 1 問の実時間上限。到達したら延長せず打ち切って次へ |
| `--bare` | harness を通さず素の LLM に 1 往復で解かせる |
| `--model NAME` | `--bare` で使うモデル名 |
| `--cleanup-workspaces` | 採点後に workspace を消す（大量に回すとき） |
| `--overwrite` | 記録済みの問題も再実行する（既定は未実行分だけ追加 = 再開） |
| `--reference-model NAME` | 出題を読み出す公開 report（既定 `gpt-4o`） |

## 出力

`benchmarks_chembench/results/<label>/`

| ファイル | 内容 |
| --- | --- |
| `records.jsonl` | 1 行 1 問の実行ログ（答え / run_id / 所要時間 + **採点に必要な正解情報**）。**再開に使う** |
| `scored.jsonl` | 採点済みレコード（score / metrics / 判定の詳細） |
| `metrics.json` | トピック → 採点方式 → 要求能力の集計と文献値の突き合わせ |
| `report.md` | 上記の Markdown レポート |
| `purge_counts.json` | `--retry-failed` で再実行に戻した回数（問題ごと。無限ループ防止） |

個々の実行の詳細（ツール呼び出し・失敗理由）は通常の run と同じく
`workspaces/chembench-<label>-<provider>-<question_name>/` と `traces/` に残る。

## 動作の流れ

1. **出題** — 公開 report にある**プロンプト原文**（選択肢の並び順まで同一）を渡し、
   答えの保存先だけを足す（workspace 直下の `chembench_answer.json` に `{"answer": ...}`）。
   `--bare` では保存先の指示も足さない（文献と完全に同条件）。
2. **実行** — `HarnessController.run(task_type=<トピックの ahc_task_type>,
   expected_outputs=["chembench_answer.json"])`。答えファイルが無ければ Verifier が
   不合格にし、`--max-replans` 回まで再試行する。
3. **抽出** — `chembench_answer.json` → 最終メッセージ中の `[ANSWER]...[/ANSWER]` →
   最終メッセージ全文の順に答えを探す（どこから取れたかは `answer_source` に残る）。
4. **採点** — 公式の `all_correct` 規則（下記）。ahc / 素の LLM で**同一の関数**を通す。
5. **集計** — トピック / 採点方式 / 要求能力別に平均し、同じ問題に絞った文献値と並べる。

## なぜ HuggingFace ではなく公開 report から出題するのか

ChemBench の問題は HuggingFace にもあるが、**出題の正本は公式クローン同梱の
`chembench/reports/<model>/reports/*/*.json`** にしている。理由は 3 つ。

1. **文献値と 1 対 1 で突き合わせられる。** report の `name`（question_name）が
   `chembench/reports/<model>/<model>.json` の問題ごとの `all_correct` と同じ鍵なので、
   ahc が解いた問題だけに文献値を絞れる。HuggingFace 側の `name` は重複が多く
   （例: `organic_chemistry_okuyama_maskill` が 152 問中に何度も現れる）、
   問題文で突き合わせても 2,788 問中 2,361 問（85%）しか一致しない。
2. **文献の各モデルが実際に見たプロンプトそのもの**が入っている。ahc にも同じ文字列を
   渡せるので、出題条件の差が消える。
3. クローンだけで完結する（ネットワークも pyarrow も要らない）。

正解（`targets_`）は生の LaTeX 表記（`\ce{NaCl}`）で、プロンプト側の選択肢は
chembench の後処理で剥がされた表記になっている。`dataset.post_process()` は公式
`utils.py` の 5 つの正規表現をそのまま移したもので、これを掛けると
**全モデルの report 横断 12,720 件すべてで選択肢テキストが一致する**。
したがって「選択肢の文字 → 得点」の対応は推測なしに復元できる。対応が取れない
問題は黙って 0 点にせず**落とす**（`resolve_score_map` が `None` を返す）。

出題が HuggingFace 版と食い違っていないかは
`python -m benchmarks_chembench.evaluate verify-hf` で照合できる（正解の集合を指紋にする）。
実測（2026-09-22）は **2,788 問中 2,782 問が一致**し、一致した全問で**トピックの判定も
HuggingFace の config と一致**した（= `classified_questions_leaderboard.csv` 由来の
トピック付けが正しいことの独立な確認）。残る 6 問は上流のデータ改訂によるもので、
HuggingFace 版は 2,785 行しかない（report 時点から 3 問以上減っている）。

## 採点方式

ChemBench の採点方式は 2 つだけで、代表値は `all_correct`（正答率）。

| metric_kind | 規則 | 問題数 |
| --- | --- | --- |
| `mcq` | 選んだ選択肢の集合が正解の集合と**完全一致**（`hamming == 0`）。部分点なし | 2,544 |
| `numeric` | `|答え − 正解| < 0.01 × 正解` | 244 |

`metrics.py` が公式実装（`prompter.py` / `metrics.py` / `utils.py`）の規則を純 python で
再実装している。対応関係は `metrics.py` の docstring に表で書いてある。

踏襲している「癖」（**文献値も同じ規則で採点されているので緩めない**）:

- 選択肢問題は**部分点なし**。複数正解の問題で 1 つ取りこぼすと 0 点。
  出来の度合いは `f1` / `multiple_choice_grade` に出るのでレポートに併記する。
- 得点 0.5 の選択肢は「正解」に数えない（公式が `v == 1` で判定しているため）。
- 数値の許容差は**相対 1% ではなく「正解の 1% を絶対量として」** mae と比較する。
  そのため**正解が 0 以下の問題は原理的に不正解**になる。公式の挙動なのでそのまま。
- 答えが取れなかった問題は 0 点（`answered=False`）。集計から外さない
  ——「形式どおりに答えられない」ことも能力の一部として測るのが ChemBench の立場。

### 採点が公式と一致することの確認

```bash
python -m benchmarks_chembench.evaluate validate
```

公開 report には各モデルの**回答本文**と、公式実装がそのとき計算した指標
（`metrics.hamming` / `metrics.mae`）が両方入っている。そこで「回答本文 → 本実装で採点 →
公式が記録した正誤と比較」を全問について行う。実測（2026-09-22）:

| モデル | 比較数 | 一致率 | MCQ | numeric |
| --- | --: | --: | --- | --- |
| random_baseline | 2,788 | **100.00%** | 2,544 / 2,544 | 244 / 244 |
| gpt-4o | 2,757 | **99.93%** | 2,540 / 2,542 | 215 / 215 |
| claude3.5 | 2,646 | **100.00%** | 2,435 / 2,435 | 211 / 211 |

不一致は gpt-4o の 2 問だけで、いずれも**選択肢の本文で答えた**ケース
（`[ANSWER]D3d[/ANSWER]`）。公式の正規表現はこれを拾えず、LLM に答えを抽出させる
フォールバックで救済している。本実装はそのフォールバックを持たないので
**わずかに厳しい**——差は ahc 側に不利に働くので、比較としては安全側。
refusal（回答拒否）は公式が本文を見ずに 0 点としているため母集団から外し、件数だけ報告する。

## harness の有無の比較

| 列 | 中身 | ツール | Verifier | 再計画 |
| --- | --- | --- | --- | --- |
| **AHC** | `runner.py`（ahc） | あり | あり | あり |
| **素のLLM** | `bare.py`（`--bare`） | **なし**（`tools=[]`） | なし | なし |
| **文献** | 公開 report 37 手法 | 大半なし。`*-react` と `paper-qa` のみあり | — | — |

素の LLM 側で気をつけていること（ChemEval 版と同じ罠を踏まないため）:

- `allowed_tools=[]` は「許可リスト未指定」の意味で**ツールは無効にならない**。
  ツールセット自体を空にする `tools=[]` が必要。念のため `ToolUseBlock` を検知したら
  `ToolUseDetected` で失敗扱いにする（黙って「素の LLM」を名乗る結果が混ざるのを防ぐ）。
- 利用枠切れのとき SDK は例外ではなく `You've hit your session limit ...` という**本文**を
  返す。これを答えとして記録すると「回答済みだが不正解」になり再開もできないので、
  `UsageLimitReached` で失敗として記録する。

**文献値の大半は素の LLM なので、AHC と並べた差には harness の寄与が入る。**
同条件で比べたいときは

- レポート 3.1 の「ツールを使うエージェント同士」の行（`gpt-4o-react` /
  `claude3.5-react` / `paper-qa` が相手）
- 自分で測った「素のLLM」列（同じモデル・同じ問題・同じ採点）

の 2 つを見る。前者は「エージェント構成としてどうか」、後者は「harness が何点足したか」。

## 文献値との比較

`baselines.yaml` は**数値を持たない**（メタ情報だけ）。値は実行時に
`chembench/reports/<model>/<model>.json` の問題ごとの `all_correct` から読む。

比較で守っていること:

- **母集団を揃える。** 文献値は ahc が解いた問題だけに絞って再集計する。
- **n を必ず併記する。** モデルごとに解けた問題数が違う（refusal・欠測）ため、
  「文献最高」は**その問題集合を全問解いている手法**の中から選ぶ。
  全問解いた手法が無い場合はその旨をレポートに出す。
- **ツールの有無を明示する**（`kind: agent` / 種別列）。ahc はツールつきエージェントなので
  素の LLM と並べるだけでは不公平。
- **参考として全 2,788 問での値も併記する**（この run の部分集合が代表的かを判断するため）。
- 人間（化学者 20 名 × ツール有無）は約 120 問の部分集合しか解いていないので、
  レポートでも human subset の節でだけ比べる。
- `*-T-one`（temperature=1）や `log_prob*`（対数確率で選択）は同じモデルの別条件なので
  順位表では参考値として扱う。

文献値は**出典データであって採点には一切使わない**（`score` の計算経路から独立）。

## トピックと task_type

全トピックの `ahc_task_type` は `generic`（`topics.yaml` の `default_ahc_task_type`）。
ChemBench は**知識・推論のベンチマーク**であって計算タスク集ではなく、実測で

| 種別 | 該当問題数 / 2,788 |
| --- | --: |
| 量子化学（HOMO/LUMO など）に触れる | 1 |
| 逆合成 | 1 |
| 反応生成物の予測 | 5 |
| SMILES を含む（うち NMR ピーク数え 56） | 91 |

しかないため、専用ツールへ振り分ける意味のあるトピックが無い。ツールは
`build_default_registry` が task_type に関係なく全部登録するので、`generic` でも
エージェントは Bash・RDKit・量子化学計算を必要に応じて使える。
振り分けを変えたくなったら `topics.yaml` の `task_type_overrides` を書くだけでよい
（コード変更は不要）。

つまりこの評価で見える harness の寄与は「重い化学計算」ではなく
**Bash での数値計算・RDKit での確認・Verifier による回答形式の担保・再計画**である、
という想定で読む。

## 注意

- **正解と採点方式は Evolver の変更禁止対象**。`benchmarks/tasks.yaml` と同じ扱いで、
  `topics.yaml`・`metrics.py`・`score.py`・`validate.py` を自己改善で緩めないこと。
  `baselines.yaml`（文献値のメタ情報）も出典データなので書き換えない。
- ChemBench のデータセットは MIT ライセンスだが**評価専用**で、
  学習・fine-tuning に使わないこと（公式の明示的な要請）。問題本文には
  `canary` 文字列が埋め込まれている。
- `benchmarks_chembench/chembench/` は入れ子クローンなので**親リポジトリへコミットしない**
  （`.gitignore` 済み）。`data/` と `results/` も同様。
- **まれにエージェントが `chembench_answer.json` を workspace ではなくプロセスの
  カレントディレクトリ（= リポジトリ直下）に書く**（約 200 問で 1〜2 件）。採点には
  影響しない（`extract.read_answer_file` は workspace 以下しか探さない）が、
  全件 run は数日かかって `git status` を汚し続けるので `/chembench_answer.json` を
  `.gitignore` に入れてある。ただし**そのとき workspace 側に答えファイルが無ければ
  Verifier が不合格にして 1 回やり直す**ので、頻発するようなら
  `runner.PROMPT_TEMPLATE` の保存先の指示を見直すこと。
- 1 問 = 1 エージェント実行なので、全 2,788 問を回すと SDK の API コストと時間が大きい
  （ChemEval 側の実測では**セッション枠で約 160 問・週次枠で約 1,470 問**が上限）。
  文献値と同条件で比べるには全件が必要なので、同じ `--label` で何度も呼んで
  積み増す運用が前提（上の「文献値と同じ設定で回す」を参照）。
- `chemical_preference`（1,001 問）と `toxicity_and_safety`（675 問）で全体の 6 割を占める。
  **`--limit-per-topic N` はトピックごとに均等に取るので、この構成比を再現しない。**
  N 指定の run は「同一問題に絞った文献値」との比較には使えるが、公表されている
  全 2,788 問の値と横並びにはできない（レポートは両方の列を出して区別できるようにしている）。
- トピックごとの問題数が偏っているため、**全件 run では `chemical_preference` が最も遅い**
  （エージェントが RDKit で記述子を計算しに行く。実測 約 85 秒/問）。
