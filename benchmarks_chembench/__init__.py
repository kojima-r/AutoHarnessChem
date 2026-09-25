"""ChemBench で AutoHarnessChem を評価するためのパッケージ。

構成:
  chembench/     公式リポジトリのクローン（別 git。参照専用で改変しない）
  topics.yaml    9 トピックの定義と ahc に渡す task_type
  baselines.yaml 公開 report（文献値）のメタ情報。数値は持たず report から読む
  catalog.py     トピック定義と「問題 → トピック / 要求能力 / 難易度」の対応表
  dataset.py     出題の読み込み（公式の公開 report から。プロンプトも正解も原文のまま）
  runner.py      ahc（HarnessController）に 1 問ずつ解かせる実行ループ
  bare.py        harness を通さない素の LLM（文献と同条件のベースライン）
  extract.py     エージェント出力から答えを取り出す
  metrics.py     採点（公式の all_correct 規則を純 python で再実装）
  score.py       records.jsonl の採点
  validate.py    本実装の採点が公式と一致するかの検証（公開 report で照合）
  baselines.py   文献値の読み込みと ahc の結果との突き合わせ（採点には関与しない）
  report.py      トピック / 採点方式 / 要求能力別の集計と Markdown レポート
  evaluate.py    CLI（`python -m benchmarks_chembench.evaluate` / `ahc chembench`）

`benchmarks_chemeval/` と対になる評価で、違いは文献値の質。ChemBench は
**問題ごとの正誤が 37 手法ぶん公開されている**ため、ahc が解いた問題だけに
母集団を絞った同一問題での比較ができる（論文の表を転記する必要がない）。
"""
