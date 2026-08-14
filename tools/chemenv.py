"""RDKit / pandas / scikit-learn を使うツールを専用環境で実行する。

harness 本体のプロセス（`ahc` 環境）には rdkit も pandas も入れない方針なので、
これらのツールも量子化学ツールと同じ仕組み（自己完結スクリプト + JSON 入出力）で
`SandboxConfig.named_envs["rdkit"]`（既定: conda env `pyscf`）に委譲する。

これにより「harness 環境に rdkit が無いので standardize_smiles が使えない」という
状態がなくなる（複合タスクでは SMILES 正準化や記述子計算が前処理として必要になる）。

提供するツール:
  - standardize_smiles          … RDKit の Cleanup + FragmentParent で正準化
  - generate_3d_structure       … ETKDGv3 + MMFF で 3D 構造（xyz）
  - calculate_rdkit_descriptors … MolWt / LogP / TPSA 等の記述子 CSV
  - inspect_dataset             … CSV の行数・列型・欠損・統計量（pandas）
  - cross_validate_model        … K-fold 交差検証 + 散布図（scikit-learn）
"""
from __future__ import annotations

import json
from pathlib import Path

from schemas import ToolResult
from tools.envrun import EnvScript, artifact, run_env_script

ENV_NAME = "rdkit"

# 既定メモリ上限（MB）。sandbox 既定の 4096 では rdkit / pandas / scikit-learn /
# matplotlib の import だけでアドレス空間が足りず SIGKILL される
DEFAULT_MEMORY_LIMIT_MB = 8192

_MISSING_HINTS = (
    (r"No module named 'rdkit'", "missing_dependency", False,
     "専用環境 `{env}` に rdkit がありません: `conda run -n {env} pip install rdkit`"
     "（agent 側では修復不能）。"),
    (r"No module named 'sklearn'|No module named 'scikit", "missing_dependency", False,
     "専用環境 `{env}` に scikit-learn がありません: "
     "`conda run -n {env} pip install scikit-learn`（agent 側では修復不能）。"),
    (r"No module named 'pandas'", "missing_dependency", False,
     "専用環境 `{env}` に pandas がありません: `conda run -n {env} pip install pandas`"
     "（agent 側では修復不能）。"),
)

_SCRIPT_BODY = r'''
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

spec = json.loads(Path("__INPUT_JSON__").read_text(encoding="utf-8"))
mode = spec["mode"]
result = {}


def write_csv(path, rows):
    import csv
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if mode == "standardize":
    from rdkit import Chem
    from rdkit import RDLogger
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError:                       # 版によって配置が変わる
        from rdkit.Chem import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    rows, failures = [], []
    for smiles in spec["smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failures.append(smiles)
            rows.append({"input": smiles, "canonical": None, "error": "parse_failed"})
            continue
        mol = rdMolStandardize.FragmentParent(rdMolStandardize.Cleanup(mol))
        rows.append({"input": smiles, "canonical": Chem.MolToSmiles(mol), "error": None})
    result = {"results": rows, "failures": failures}

elif mode == "structure3d":
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smiles = spec["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result = {"error": "invalid_smiles"}
    else:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        embedded = AllChem.EmbedMolecule(mol, params)
        if embedded != 0:                     # 失敗したらランダム座標で再試行
            params.useRandomCoords = True
            embedded = AllChem.EmbedMolecule(mol, params)
        if embedded != 0:
            result = {"error": "embedding_failed"}
        else:
            try:
                AllChem.MMFFOptimizeMolecule(mol)
                force_field = "MMFF"
            except Exception:
                AllChem.UFFOptimizeMolecule(mol)
                force_field = "UFF"
            conformer = mol.GetConformer()
            atoms, lines = [], [str(mol.GetNumAtoms()), smiles]
            for atom in mol.GetAtoms():
                position = conformer.GetAtomPosition(atom.GetIdx())
                atoms.append({"symbol": atom.GetSymbol(), "x": position.x,
                              "y": position.y, "z": position.z})
                lines.append("%s %.6f %.6f %.6f"
                             % (atom.GetSymbol(), position.x, position.y, position.z))
            xyz_path = "%s.xyz" % spec["name"]
            Path(xyz_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
            result = {"xyz_path": xyz_path, "atoms": atoms,
                      "n_atoms": mol.GetNumAtoms(), "force_field": force_field}

elif mode == "descriptors":
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    rows, failures = [], []
    for smiles in spec["smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failures.append(smiles)
            continue
        rows.append({
            "smiles": smiles,
            "mol_weight": Descriptors.MolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "tpsa": rdMolDescriptors.CalcTPSA(mol),
            "n_heavy_atoms": mol.GetNumHeavyAtoms(),
            "n_rings": rdMolDescriptors.CalcNumRings(mol),
            "n_hbd": rdMolDescriptors.CalcNumHBD(mol),
            "n_hba": rdMolDescriptors.CalcNumHBA(mol),
            "n_rotatable": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        })
    write_csv(spec["output_csv"], rows)
    result = {"rows": rows, "failures": failures,
              "output_csv": spec["output_csv"] if rows else None}

elif mode == "dataset":
    import pandas as pd

    frame = pd.read_csv(spec["path"])
    info = {
        "path": spec["path"],
        "n_rows": int(len(frame)),
        "columns": {c: str(t) for c, t in frame.dtypes.items()},
        "n_missing": {c: int(n) for c, n in frame.isna().sum().items() if n > 0},
        "head": json.loads(frame.head(5).to_json(orient="records")),
    }
    numeric = frame.select_dtypes("number")
    if len(numeric.columns):
        info["describe"] = json.loads(numeric.describe().to_json())
    result = info

elif mode == "cv":
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold

    frame = pd.read_csv(spec["features_csv"])
    target = spec["target_column"]
    if target not in frame.columns:
        result = {"error": "target_not_found", "columns": list(frame.columns)}
    else:
        drop = set(spec.get("drop_columns") or []) | {target}
        features = frame.drop(columns=[c for c in drop if c in frame.columns])
        features = features.select_dtypes("number")
        y = frame[target].to_numpy()
        if features.shape[1] == 0:
            result = {"error": "no_numeric_features"}
        else:
            mask = ~(features.isna().any(axis=1) | pd.isna(y))
            X, y = features[mask].to_numpy(), y[mask]
            n_folds = int(spec["n_folds"])
            if len(y) < n_folds:
                result = {"error": "insufficient_data", "n_rows": int(len(y))}
            else:
                estimator = (Ridge() if spec["model"] == "ridge"
                             else RandomForestRegressor(n_estimators=300, random_state=42))
                oof = np.zeros_like(y, dtype=float)
                splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                for train_idx, test_idx in splitter.split(X):
                    estimator.fit(X[train_idx], y[train_idx])
                    oof[test_idx] = estimator.predict(X[test_idx])
                metrics = {
                    "model": spec["model"], "n_folds": n_folds,
                    "n_samples": int(len(y)), "n_features": int(X.shape[1]),
                    "features": list(features.columns),
                    "r2": float(r2_score(y, oof)),
                    "rmse": float(np.sqrt(mean_squared_error(y, oof))),
                    "mae": float(mean_absolute_error(y, oof)),
                }
                Path("cv_metrics.json").write_text(
                    json.dumps(metrics, indent=2), encoding="utf-8")
                pd.DataFrame({"y_true": y, "y_pred": oof}).to_csv(
                    "oof_predictions.csv", index=False)
                outputs = ["cv_metrics.json", "oof_predictions.csv"]
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    figure, axes = plt.subplots(figsize=(5, 5))
                    axes.scatter(y, oof, alpha=0.7)
                    limits = [min(y.min(), oof.min()), max(y.max(), oof.max())]
                    axes.plot(limits, limits, "k--", linewidth=1)
                    axes.set_xlabel("true")
                    axes.set_ylabel("predicted")
                    axes.set_title("%s CV: R2=%.3f, RMSE=%.3g"
                                   % (spec["model"], metrics["r2"], metrics["rmse"]))
                    figure.tight_layout()
                    figure.savefig("true_vs_pred.png", dpi=150)
                    plt.close(figure)
                    outputs.append("true_vs_pred.png")
                except ImportError:
                    pass
                result = {"metrics": metrics, "outputs": outputs}

else:
    raise SystemExit("unknown mode: %s" % mode)

Path("__OUTPUT_JSON__").write_text(json.dumps(result, ensure_ascii=False, default=str),
                                  encoding="utf-8")
print("[chemenv] mode=%s done" % mode)
'''


def _script(mode: str) -> EnvScript:
    return EnvScript(body=_SCRIPT_BODY, script_name=f"_chemenv_{mode}.py",
                     input_json=f"_chemenv_{mode}_input.json",
                     output_json=f"_chemenv_{mode}_output.json")


def _run(sandbox, workspace: Path, mode: str, spec: dict,
         timeout_sec: int | None = None,
         memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
         ) -> tuple[dict | None, ToolResult | None]:
    run = run_env_script(sandbox, workspace, _script(mode), {"mode": mode, **spec},
                         timeout_sec=timeout_sec, memory_limit_mb=memory_limit_mb,
                         extra_failures=_MISSING_HINTS,
                         timeout_hint="入力を分割してください。")
    return run.payload, run.error


# ---------------------------------------------------------------------------
# standardize_smiles
# ---------------------------------------------------------------------------

def standardize_smiles(workspace: Path, smiles: list[str], *, sandbox) -> ToolResult:
    """SMILES を RDKit で正準化する（Cleanup + FragmentParent で塩・溶媒和を除去）。"""
    if not smiles:
        return ToolResult(status="failed", summary="smiles is empty",
                          error_type="invalid_input")
    payload, error = _run(sandbox, workspace, "standardize", {"smiles": list(smiles)})
    if error is not None:
        return error
    results, failures = payload["results"], payload["failures"]
    status = ("success" if not failures
              else ("partial" if len(failures) < len(smiles) else "failed"))
    return ToolResult(
        status=status,
        summary=f"standardized {len(smiles) - len(failures)}/{len(smiles)} SMILES",
        data={"results": results, "failures": failures},
        error_type="invalid_smiles" if failures else None,
    )


# ---------------------------------------------------------------------------
# generate_3d_structure
# ---------------------------------------------------------------------------

def generate_3d_structure(workspace: Path, smiles: str, name: str = "molecule",
                          *, sandbox) -> ToolResult:
    """SMILES から 3D 構造（ETKDGv3 + MMFF）を作り xyz として保存する。"""
    workspace = Path(workspace)
    payload, error = _run(sandbox, workspace, "structure3d",
                          {"smiles": smiles, "name": Path(name).name})
    if error is not None:
        return error
    if payload.get("error") == "invalid_smiles":
        return ToolResult(status="failed", summary=f"invalid SMILES: {smiles}",
                          retryable=False, error_type="invalid_smiles")
    if payload.get("error") == "embedding_failed":
        return ToolResult(status="failed",
                          summary=f"3D embedding failed for {smiles}"
                                  "（ランダム座標でも失敗）",
                          retryable=True, error_type="embedding_failed")
    xyz_path = workspace / payload["xyz_path"]
    return ToolResult(
        status="success",
        summary=(f"3D structure written to {xyz_path.name} "
                 f"({payload['n_atoms']} atoms, {payload['force_field']})"),
        data={"xyz_path": str(xyz_path), "atoms": payload["atoms"],
              "n_atoms": payload["n_atoms"]},
        artifacts=[artifact(xyz_path)] if xyz_path.exists() else [],
    )


# ---------------------------------------------------------------------------
# calculate_rdkit_descriptors
# ---------------------------------------------------------------------------

def calculate_rdkit_descriptors(workspace: Path, smiles: list[str],
                                output_csv: str = "rdkit_descriptors.csv",
                                *, sandbox) -> ToolResult:
    """RDKit 記述子（MolWt / LogP / TPSA 等）を計算して CSV に保存する。"""
    if not smiles:
        return ToolResult(status="failed", summary="smiles is empty",
                          error_type="invalid_input")
    workspace = Path(workspace)
    payload, error = _run(sandbox, workspace, "descriptors",
                          {"smiles": list(smiles), "output_csv": output_csv})
    if error is not None:
        return error
    rows, failures = payload["rows"], payload["failures"]
    if not rows:
        return ToolResult(status="failed", summary="no valid SMILES",
                          data={"failures": failures}, error_type="invalid_smiles")
    csv_path = workspace / output_csv
    return ToolResult(
        status="success" if not failures else "partial",
        summary=f"descriptors for {len(rows)}/{len(smiles)} molecules → {csv_path.name}",
        data={"output_csv": str(csv_path), "results": rows, "failures": failures},
        artifacts=[artifact(csv_path)] if csv_path.exists() else [],
        error_type="invalid_smiles" if failures else None,
    )


# ---------------------------------------------------------------------------
# inspect_dataset
# ---------------------------------------------------------------------------

def inspect_dataset(workspace: Path, path: str, *, sandbox) -> ToolResult:
    """CSV の行数・列型・欠損・統計量・先頭行を調べる。"""
    workspace = Path(workspace)
    target = Path(path)
    if not target.is_absolute():
        for base in (workspace, Path.cwd()):
            if (base / target).exists():
                target = (base / target).resolve()
                break
    if not target.exists():
        return ToolResult(status="failed", summary=f"dataset not found: {path}",
                          retryable=False, error_type="input_not_found")
    payload, error = _run(sandbox, workspace, "dataset", {"path": str(target)})
    if error is not None:
        return error
    return ToolResult(
        status="success",
        summary=f"{target.name}: {payload['n_rows']} rows, "
                f"columns={list(payload['columns'])}",
        data=payload,
    )


# ---------------------------------------------------------------------------
# cross_validate_model
# ---------------------------------------------------------------------------

def cross_validate_model(workspace: Path, features_csv: str, target_column: str,
                         model: str = "random_forest", n_folds: int = 5,
                         drop_columns: list[str] | None = None,
                         memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
                         *, sandbox) -> ToolResult:
    """特徴量 CSV から target 列を予測する回帰モデルを K-fold 交差検証する。"""
    workspace = Path(workspace)
    features = Path(features_csv)
    if not features.is_absolute():
        for base in (workspace, Path.cwd()):
            if (base / features).exists():
                features = (base / features).resolve()
                break
    if not features.exists():
        return ToolResult(status="failed",
                          summary=f"features file not found: {features_csv}",
                          error_type="input_not_found")

    payload, error = _run(sandbox, workspace, "cv", {
        "features_csv": str(features), "target_column": target_column,
        "model": model, "n_folds": int(n_folds),
        "drop_columns": list(drop_columns or []),
    }, memory_limit_mb=memory_limit_mb)
    if error is not None:
        return error

    problem = payload.get("error")
    if problem == "target_not_found":
        return ToolResult(status="failed",
                          summary=f"target column `{target_column}` not in "
                                  f"{payload.get('columns')}",
                          error_type="invalid_input")
    if problem == "no_numeric_features":
        return ToolResult(status="failed", summary="no numeric feature columns remain",
                          error_type="invalid_input")
    if problem == "insufficient_data":
        return ToolResult(status="failed",
                          summary=f"only {payload.get('n_rows')} usable rows "
                                  f"(< {n_folds} folds)",
                          error_type="insufficient_data")

    metrics = payload["metrics"]
    return ToolResult(
        status="success",
        summary=(f"{n_folds}-fold CV: R2={metrics['r2']:.3f}, "
                 f"RMSE={metrics['rmse']:.3g}"),
        data=metrics,
        artifacts=[artifact(workspace / name) for name in payload["outputs"]
                   if (workspace / name).exists()],
    )
