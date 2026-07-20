---
name: result-reporting
description: Produce the final user-facing report in Japanese as both Markdown and HTML with artifacts referenced by path.
version: 1.1.0
risk_level: low
required_tools:
  - inspect_artifact
  - run_python_sandbox
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
   - **成果物**（各ファイルの相対パスと内容の1行説明。画像は `![alt](path)` で参照）
   - **制限事項・警告**（除外した分子、科学的警告、修復不能だった点）
3. 同じ内容を HTML 版 `report_user.html` としても保存する。生成には本 Skill 付属の
   `scripts/render_html.py` を `run_python_sandbox` で実行するのが確実:
   スクリプト本文を読み込み、`INPUT_MD = "report_user.md"`, `OUTPUT_HTML = "report_user.html"`
   のまま実行すればよい（外部パッケージ不要）。
4. HTML 内の画像参照は workspace からの相対パスにする（md と同じディレクトリに置くため、
   `![...](true_vs_pred.png)` → `<img src="true_vs_pred.png">` がそのまま表示できる）。
5. 誇張しない。失敗・除外・警告は隠さず明記する。md と html の内容は一致させること。

# Completion criteria

- 結論に定量値が含まれ、すべての成果物ファイルが報告から参照されている。
- `report_user.md` と `report_user.html` の両方が workspace 直下に存在し、内容が一致している。

# Recovery procedure

- 成果物パスが不明な場合は `inspect_artifact` で存在を確認してから記載する。
- HTML 変換に失敗した場合は、`markdown` パッケージに頼らず付属スクリプト
  `scripts/render_html.py`（正規表現ベースの簡易変換、依存なし）を使う。
