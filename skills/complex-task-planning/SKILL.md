---
name: complex-task-planning
description: Plan and execute composite requests (design + calculation + retrosynthesis + report) within the harness time budget, always leaving usable artifacts.
version: 1.1.0
risk_level: low
required_tools:
  - verify_scientific_result
  - inspect_artifact
  - render_report_html
allowed_paths:
  - /workspace
expected_outputs: []
task_types: []
---

# Procedure

複数の側面を含む要求（例「骨格から吸収波長の条件を満たす分子を探し、合成経路も出す」）は、
**実時間上限（既定 3600s）ごとに「必ず報告まで到達できる状態」を保ちながら**進める。
上限に達すると harness がユーザへ「さらに待つか」を確認し、延長された場合は
**作業が中断されずそのまま続く**（計算はやり直されない）。打ち切られた場合はその時点の
成果物だけで検証・報告される。したがって「いま打ち切られても意味のある成果が残るか」を
常に満たしておくことが重要で、長く待てるかどうかに賭けてはいけない。

1. **要求を副タスクへ分解し、最初に書き出す**。各副タスクに (a) 使うツール、
   (b) 生成する成果物ファイル名、(c) 概算コストを付ける。
   達成条件に「要求に含まれる他の側面」が列挙されている場合、それが分解の指針になる。
2. **安い順に実行する**。高コストなものを先に回すと時間切れで何も残らない。
   目安（軽 → 重）: `standardize_smiles` / `calculate_rdkit_descriptors` →
   `plan_retrosynthesis`（数十秒/分子）→ `calculate_orbitals`（数十秒/分子）→
   `calculate_tddft_spectrum`（b3lyp/sto-3g なら ~10 秒/分子、
   CAMB3LYP/6-31g(d) なら **~9 分・4GB 超/分子**）→ `optimize_absorption_wavelength`
   （TDDFT × trial 数）→ `scan_esipt_pes`（拘束最適化 × 点数）。
3. **重い計算はスクリーニング → 精密化の 2 段にする**。
   まず軽い条件（`basis="sto-3g"`, `nstates=5`, 1 分子ずつ）で候補を絞り、
   残り時間に余裕がある場合のみ上位候補を高精度で再計算する。
   候補は 3〜4 件までに絞る（全候補を高精度で回す計画は立てない）。
4. **1 ステップごとに成果物を残す**。各副タスクが終わるたびに CSV / JSON が
   workspace 直下にあることを `inspect_artifact` で確認する。
   ツールが `status=partial`（`interrupted=true` / `pending` 付き）を返したら、
   **完了分は成果物として確定している**。残りは `pending` の入力だけで呼び直す。
5. **中間報告を早めに作る**。重い工程に入る前に `report_user.md` の骨格
   （結論・手順・成果物・制限事項）を書き、結果が増えるたびに更新して
   `render_report_html` を再実行する。時間切れでも報告が残る。
6. **最後に `verify_scientific_result`** で不足を確認し、埋められないものは
   制限事項として明記する。

# Rules

- **待機ループを作らない。** `sleep` や `run_python_sandbox` でのポーリングで
  時間を使わない。ツール呼び出しはそれ自体が完了まで待つ（timeout も内部で処理される）。
- **ツールを Bash で置き換えない。** 専用環境ツールが失敗した場合でも、
  `conda run` 等で手動実行して回避しない（成果物の記録・検証・再現性が失われる）。
  環境側の問題は `error_type`（`missing_environment` / `missing_dependency` /
  `model_unavailable`）とともに修復不能として報告する。
- **予算配分**: 1 試行の上限のうち、重い計算は 6 割程度までに抑え、
  残りを検証と報告に使う。時間が読めない工程は入力を分割して複数回呼ぶ。
- **上限は延長されうるが、それを前提に計画しない。** 数時間〜数日かかる計算を
  投げる場合も、区切りごとに成果物と中間報告を残す（延長が承認されない場合に
  何も残らないのを避ける）。
- 部分的な結果しか得られなかった場合も、**何ができて何ができなかったか**を
  数値付きで報告する（黙って省略しない）。

# Completion criteria

- 要求に含まれる各側面について、成果物または「できなかった理由」が揃っている。
- `report_user.md` / `report_user.html` が存在し、部分結果でも結論が書かれている。

# Recovery procedure

- **時間切れで打ち切られた（次の試行で修復指示が来る）**: 分解し直し、
  もっとも情報量の多い副タスク 1 つに絞って完了させ、報告する。
- **ツールが partial を返し続ける**: 入力を 1 件に減らし、計算条件を 1 段落とす
  （`basis="sto-3g"`, `nstates=3`）。それでも通らなければ、その分子は除外して報告する。
- **必須の期待出力が作れない**: 代替の成果物（軽い条件での結果、既知値との比較）を作り、
  期待出力が満たせない理由を報告に明記する。
