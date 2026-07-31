---
name: result-reporting
description: Produce the final user-facing report in Japanese as both Markdown and HTML, with molecular structures drawn from SMILES.
version: 2.0.0
risk_level: low
required_tools:
  - render_report_html
  - inspect_artifact
allowed_paths:
  - /workspace
expected_outputs:
  - report_user.md
  - report_user.html
task_types: []
---

# Procedure

1. 検証（scientific-verification）に合格してから報告を書く。
2. 報告は日本語 Markdown で `report_user.md` として workspace 直下に保存し、以下の構成にする:
   - **結論**（1〜3行。数値は具体的に: 例「HOMO = -6.72 eV」「CV R2 = 0.41」）
   - **手順の要約**（使用した手法・基底・モデル・fold 数）
   - **構造**（扱った分子・反応を SMILES で示す。下記の記法で構造式として描画される）
   - **成果物**（各ファイルの相対パスと内容の1行説明。画像は `![alt](path)` で参照）
   - **制限事項・警告**（除外した分子、科学的警告、修復不能だった点）
3. **構造式の記法**（SmilesDrawer で描画される。分子や反応を扱った報告では必ず入れる）:

   ````markdown
   ```smiles
   CCO                                              エタノール
   CC(=O)Nc1ccc(O)cc1                               パラセタモール
   CC(=O)OC(C)=O.Nc1ccc(O)cc1>>CC(=O)Nc1ccc(O)cc1   アセチル化（反応式）
   ```
   ````

   - 1 行 = `SMILES<空白>ラベル`。ラベルは省略可。**SMILES 自体に空白を入れない**。
   - `>` を含む行（`反応物>>生成物`、`反応物>試薬>生成物`）は反応式として描画される。
   - 文中に小さく出したいときは `` `smiles:CCO` `` と書く（構造チップになる）。
4. HTML 版は `render_report_html` ツールで生成する（引数は既定のままで良い）。
   - SmilesDrawer が HTML に埋め込まれるので、オフラインでも構造が表示される。
   - `auto_structures`（既定 true）により、workspace の CSV（`orbital_features.csv`、
     `tddft_spectrum.csv`、`reactiont5_predictions.csv`、`retrosynthesis_routes.csv` 等）の
     SMILES 列から「構造一覧」が自動で追加される。本文の ```smiles ブロックはそれとは別に、
     結論に関係する分子・反応だけを選んで書くこと。
5. HTML 内の画像参照は workspace からの相対パスにする（md と同じディレクトリに置くため、
   `![...](true_vs_pred.png)` → `<img src="true_vs_pred.png">` がそのまま表示できる）。
6. 誇張しない。失敗・除外・警告は隠さず明記する。md と html の内容は一致させること。
   予測モデル（ReactionT5・AiZynthFinder）の出力は推定値であり実験値ではないと明記する。

# Completion criteria

- 結論に定量値が含まれ、すべての成果物ファイルが報告から参照されている。
- `report_user.md` と `report_user.html` の両方が workspace 直下に存在し、内容が一致している。
- 分子・反応を扱ったタスクでは、報告に SMILES（```smiles ブロックまたは `smiles:` チップ）が
  含まれ、HTML で構造として描画されている（`render_report_html` の summary で件数を確認できる）。

# Recovery procedure

- 成果物パスが不明な場合は `inspect_artifact` で存在を確認してから記載する。
- `render_report_html` が `error_type=input_not_found` を返す: `report_user.md` を
  workspace 直下（サブディレクトリではない）に保存してから呼び直す。
- 構造が 0 件だと報告された: ```smiles ブロックの記法（フェンスの言語指定が `smiles`、
  SMILES に空白を含めない）を見直す。
- ツールが使えない環境では `run_python_sandbox` で最小限の HTML を書き出してもよいが、
  その場合は構造が描画されないことを報告に明記する。
