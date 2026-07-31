# examples — 各 conda 環境のライブラリを直接使うサンプル

harness（エージェント・ツール層）を経由せず、**それぞれの専用 conda 環境のライブラリを
そのまま使う**実行可能なコード例です。環境が正しく構築できているかの動作確認にも使えます。

| スクリプト | conda env | 使うライブラリ | 主な出力 |
|---|---|---|---|
| `pyscf_opt_tddft_demo.py` | `pyscf` | `opt_tddft`（PySCF + RDKit） | `tddft_demo_states.csv` / `tddft_demo_molecules.csv` / `spectrum_*.png` |
| `reactiont5_demo.py` | `reactiont5` | `torch` + `transformers` | `reactiont5_demo_<task>.csv` |
| `aizynth_demo.py` | `aizynth` | `aizynthfinder` | `aizynth_demo_routes.json` / `aizynth_demo_routes.csv` |

## 実行

```bash
# まとめて（pyscf → aizynth → reactiont5）
bash examples/run_examples.sh
ahc demo --env all                  # harness CLI からでも同じ

# 個別に
conda run -n pyscf      python examples/pyscf_opt_tddft_demo.py
conda run -n aizynth    python examples/aizynth_demo.py
conda run -n reactiont5 python examples/reactiont5_demo.py --task forward

# 出力先を変える
bash examples/run_examples.sh pyscf            # → examples/output/pyscf/
DEMO_OUT=/tmp/demo bash examples/run_examples.sh
```

各スクリプトは `--help` を持ち、条件（汎関数・基底・励起状態数・探索予算など）を
コマンドラインから変えられます。出力は既定で `examples/output/<env>/`（git 管理外）。

## 各デモの内容

### 1. `pyscf_opt_tddft_demo.py`（env: `pyscf`）

`opt_tddft` の 3 つの層を順に使います。

1. `MoleculeBuilder` — 骨格 SMILES（`c1cc([*:1])ccc1[*:2]`）に置換基を `Chem.molzip` で結合
2. `TDDFTSolver` — SMILES → 3D 構造（多コンフォマー + UFF）→ DFT → TDDFT
   （HOMO/LUMO、励起波長、振動子強度。TDDFT が失敗したら TDA へ自動フォールバック）
3. `SpectrumVisualizer` — ガウシアン・スミアリングで UV-Vis スペクトル画像

既定は b3lyp/STO-3G・`nstates=5` で小分子 2 つ（ホルムアルデヒド、アクロレイン）なので
数秒〜数十秒で終わります。組み立てた分子も計算したい場合は `--with-scaffold`。

```bash
conda run -n pyscf python examples/pyscf_opt_tddft_demo.py \
    --smiles C=O --functional camb3lyp --basis 6-31g\(d\) --nstates 10 --solvent pcm
```

### 2. `reactiont5_demo.py`（env: `reactiont5`）

ReactionT5v2 の 3 つの学習済みモデルを `transformers` から直接使います。

- `forward` — 反応物・試薬 → 生成物（既定）
- `retrosynthesis` — 生成物 → 前駆体（1 段階）
- `yield` — 反応 → 収率 (0–100%)。回帰ヘッド付きの独自クラスを定義して読み込む

初回実行時に HuggingFace から 1 モデル ~250MB をダウンロードします（要ネットワーク）。

```bash
conda run -n reactiont5 python examples/reactiont5_demo.py --task all --beams 5
```

### 3. `aizynth_demo.py`（env: `aizynth`）

AiZynthFinder で目標分子の逆合成経路を探索し、経路木・段数・出発物質を出力します。
学習済みモデルが必要です。

```bash
conda run -n aizynth download_public_data /path/to/aizynth_data   # 初回のみ（~1GB）
export AIZYNTH_CONFIG=/path/to/aizynth_data/config.yml

conda run -n aizynth python examples/aizynth_demo.py \
    --target "CC(=O)Nc1ccc(O)cc1" --iteration-limit 200 --time-limit 180
```

`config.yml` は `--config` → `$AIZYNTH_CONFIG` → `$AIZYNTH_DATA/config.yml` →
既定ディレクトリ（`<repo>/data/aizynth/`, `~/aizynth_data/`）の順に探索します。
DL 先が別の場所なら `ln -s /path/to/aizynth_data data/aizynth` でも認識されます。

## 注意

- ReactionT5 / AiZynthFinder の出力は**学習済みモデルによる推定**であり、実験値ではありません。
  AiZynthFinder の `score` は経路探索の内部指標で、収率の予測値ではありません。
- 量子化学計算はスレッドを食うため、各デモは既定で 4 スレッドに制限しています
  （`--threads` で変更可）。
