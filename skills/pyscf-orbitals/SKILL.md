---
name: pyscf-orbitals
description: Calculate molecular orbital energies (HOMO/LUMO) and TDDFT absorption spectra with OptTDDFT (RDKit + PySCF).
version: 2.2.0
risk_level: medium
required_tools:
  - calculate_orbitals
  - calculate_tddft_spectrum
  - inspect_artifact
allowed_paths:
  - /workspace
expected_outputs:
  - orbital_features.csv
task_types:
  - orbital_calculation
---

# Procedure

計算エンジンは OptTDDFT（`tools/OptTDDFT` の `opt_tddft`）で、専用 conda 環境 `pyscf` の
subprocess として実行される。3D 構造生成（多コンフォマー探索 + UFF）は
ツール内部で行われるので、`generate_3d_structure` を別途呼ぶ必要はない。

1. **目的で使うツールを選ぶ**
   - HOMO/LUMO・gap・全エネルギーだけ → `calculate_orbitals`（SCF のみ・軽い）
   - 吸収波長・振動子強度・UV-Vis スペクトル → `calculate_tddft_spectrum`
     （`orbital_features.csv` も同時に出力される）
2. **軽い条件で1分子だけ試す** → 成功したら分子数・精度を上げる。
   - `calculate_orbitals` の既定は HF/STO-3G（高速確認用）。精度が必要なら
     `method="b3lyp"`, `basis="6-31g*"` 等へ引き上げる。
   - `calculate_tddft_spectrum` の既定は CAMB3LYP/6-31g(d)、`nstates=10`。
     TDDFT は SCF より桁違いに重い（芳香環を含む分子で 1 件あたり数分〜十数分）。
     **複数分子を一度に渡さない**（1 回の呼び出しで 1〜2 分子）。まず
     `functional="b3lyp"`, `basis="sto-3g"`, `nstates=5` で通し、
     必要なら `timeout_sec` を明示的に上げる（既定は sandbox の timeout）。
   - 打ち切られても分子ごとに CSV が更新されるため、`status=partial` の結果は
     そのまま使える。`data.pending` の分子だけを次の呼び出しに回す。
   - **同じ `output_csv` を指定して何回呼んでも結果は累積される**（同じ
     (smiles, method, basis) の行だけ置き換わる）。1 分子ずつ呼んでも前の分子は
     消えないので、`output_csv` を毎回変える必要はない。summary の「累計 N 行」で
     テーブル全体の行数を確認できる。
   - **実測の目安**（4 スレッド）: クマリン（C9H6O2）の TDDFT は
     b3lyp/sto-3g で約 10 秒、CAMB3LYP/6-31g(d) で **約 9 分・メモリ 4GB 超**。
     高精度条件を使うときは `timeout_sec` を 900 以上、`memory_limit_mb` を
     既定（16384）以上にし、分子は 1 件ずつ投げる。
3. **電荷・スピンは分子に合わせて明示する**（例: SO4 は `charge=-2`, `spin=0`）。
   `RuntimeError: Electron number ... and spin ...` は charge/spin 不整合のサイン。
   `spin`（= 2S, 不対電子数）が 0 以外なら UHF/UKS で計算される。
   `calculate_tddft_spectrum` は閉殻（RKS）専用なので開殻分子には使わない。
4. **溶媒効果**が必要なら `solvent_model="pcm"` と `solvent_eps`（例: クロロホルム 4.7113）。
   気相計算が既定。
5. 結果は 1 分子 1 行の `orbital_features.csv`（smiles, homo_ev, lumo_ev, gap_ev,
   total_energy_hartree 列）、TDDFT は 1 状態 1 行の `tddft_spectrum.csv`
   （wavelength_nm, oscillator_strength, state_index 列）に保存される。
   `state_index=1` が最低励起状態 (S1) = 最長波長。
6. 報告では λmax を「最長波長」ではなく**振動子強度が最大の波長**
   （`strongest_wavelength_nm`）で述べると実測スペクトルと対応が付く。

# Completion criteria

- `orbital_features.csv` が存在し、全行で homo_ev < lumo_ev。
- HOMO/LUMO が -60〜+30 eV の物理的に妥当な範囲にある。
- TDDFT では `tddft_spectrum.csv` の wavelength_nm が 50〜2000 nm の範囲にある。
- SCF 未収束の分子は結果に含めず、失敗理由が報告されている（`failures` を確認）。

# Recovery procedure

- **SCF 未収束 (error_type=scf_failed)**: (a) 基底を STO-3G へ落とす、
  (b) `max_cycle` を 400 へ増やす、(c) `method="b3lyp"` など別の汎関数にする、の順で再試行。
- **タイムアウト (error_type=timeout)**: `status=partial` なら完了分の CSV は残っている
  ので、`data.pending` の分子だけを 1 件ずつ呼び直す。`status=failed`（1 件も終わって
  いない）なら `nstates` と基底を下げるか `timeout_sec` を上げる。
- **error_type=invalid_input で基底関数が無いと言われた**: Br/I などの重元素は
  `6-31g(d)` に含まれない。`basis="def2-svp"` に変更する。
- **error_type=out_of_memory（SIGSEGV / 強制終了）**: メモリ上限が主因。
  `memory_limit_mb` を上げる（例 16384 → 32768）、基底関数を下げる（`6-31g(d)` →
  `sto-3g`）、`nstates` を減らす、`threads` を減らす、の順に試す。
  4096MB のような小さい上限では CAMB3LYP/6-31g(d) の TDDFT は SIGSEGV で落ちる。
- **error_type=missing_dependency で geometric が無い**: `use_geom_opt=false` にする
  （構造最適化なしでも計算できる）。環境へのインストールは agent 側では不可。
- **error_type=missing_environment**: conda 環境 `pyscf` / docker image が使えない。
  修復不能として報告する（Bash から `conda run` で手動実行して回避しないこと）。
- **HOMO > LUMO や範囲外の値**: 単位変換（Hartree/eV の二重変換）と占有電子数の算出を確認する。
- **TDDFT で励起状態が 0 件**: `nstates` を 15〜24 に増やす（極端に減らすと収束が悪化する）。
