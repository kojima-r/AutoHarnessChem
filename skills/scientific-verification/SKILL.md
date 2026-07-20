---
name: scientific-verification
description: Verify scientific plausibility of results before reporting (units, ranges, leakage, convergence).
version: 1.0.0
risk_level: low
required_tools:
  - verify_scientific_result
  - inspect_artifact
allowed_paths:
  - /workspace
expected_outputs: []
task_types: []
---

# Procedure

1. 最終報告の前に必ず `verify_scientific_result` を呼び、構造化判定を確認する。
2. requirements_missing が空でない場合、報告せずに不足出力を生成する。
3. scientific_warnings は無視せず、原因を特定して修正するか、修正不能な理由を報告に明記する。

# Checklist

- 軌道エネルギー: HOMO < LUMO、値は -60〜+30 eV、単位変換は一度だけ。
- SCF: converged フラグを確認。未収束の値を結果に混ぜない。
- 回帰: r2 ≤ 1、r2 > 0.999 はリーク疑い、fold 数と行数の整合。
- ファイル: 期待される出力がすべて workspace 直下にある。

# Completion criteria

- `verify_scientific_result` の判定が passed である。

# Recovery procedure

- 判定不合格時は required_repairs の各項目を順に実施し、再度検証する。
