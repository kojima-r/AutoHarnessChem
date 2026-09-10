"""ChemEval（ICLR 2026）で AutoHarnessChem を評価するためのパッケージ。

構成:
  ChemEval/      公式リポジトリのクローン（別 git。参照専用で改変しない）
  tasks.yaml     ChemEval の各タスク → 採点方式 / ahc の task_type の対応表
  dataset.py     HuggingFace からのデータ取得と読み込み
  runner.py      ahc（HarnessController）に 1 問ずつ解かせる実行ループ
  extract.py     エージェント出力から答えを取り出す
  metrics.py     採点（公式の指標を純 python で再実装）
  chem_metrics.py 正準 SMILES / Tanimoto（rdkit 環境へ envrun で委譲）
  judge.py       自由記述タスクの LLM-as-judge
  score.py       records.jsonl の採点
  baselines.yaml 論文 Table 1（0-shot text）の文献値。他手法と並べるための出典データ
  baselines.py   文献値の読み込みと ahc の結果との突き合わせ（採点には関与しない）
  report.py      レベル / 次元 / タスク別の集計と Markdown レポート
  evaluate.py    CLI（`python -m benchmarks_chemeval.evaluate` / `ahc chemeval`）
"""
