"""共通ツール実装。

すべて ToolResult を返す。重い依存 (rdkit / sklearn) は遅延 import し、
無ければ status="failed", error_type="missing_dependency" を返す。
ファイル出力はすべて workspace 配下に限定する。

量子化学計算（HOMO/LUMO・TDDFT・PES スキャン・MI 探索）は tools/opttddft.py
（OptTDDFT を専用の pyscf 環境で実行）が担当する。逆合成経路探索は
tools/aizynth.py。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from schemas import Artifact, ToolResult


def _missing(module: str) -> ToolResult:
    return ToolResult(
        status="failed",
        summary=f"required package `{module}` is not installed in the harness environment",
        retryable=False,
        error_type="missing_dependency",
    )


def _artifact(path: Path) -> Artifact:
    import mimetypes
    return Artifact(
        path=str(path),
        mime=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        bytes=path.stat().st_size,
        kind="figure" if path.suffix in (".png", ".svg", ".pdf") else "data",
    )


def _resolve_input(workspace: Path, path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        for candidate in (workspace / p, Path.cwd() / p):
            if candidate.exists():
                return candidate
    return p


# ---------------------------------------------------------------------------
# 1. inspect_dataset
# ---------------------------------------------------------------------------

def inspect_dataset(workspace: Path, path: str) -> ToolResult:
    try:
        import pandas as pd
    except ImportError:
        return _missing("pandas")
    f = _resolve_input(workspace, path)
    if not f.exists():
        return ToolResult(status="failed", summary=f"dataset not found: {path}",
                          retryable=False, error_type="input_not_found")
    df = pd.read_csv(f)
    info = {
        "path": str(f),
        "n_rows": int(len(df)),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "n_missing": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "head": df.head(5).to_dict(orient="records"),
    }
    numeric = df.select_dtypes("number")
    if len(numeric.columns):
        info["describe"] = json.loads(numeric.describe().to_json())
    return ToolResult(status="success",
                      summary=f"{f.name}: {len(df)} rows, columns={list(df.columns)}",
                      data=info)


# ---------------------------------------------------------------------------
# 2. standardize_smiles
# ---------------------------------------------------------------------------

def standardize_smiles(workspace: Path, smiles: list[str]) -> ToolResult:
    try:
        from rdkit import Chem
        try:
            from rdkit.Chem.MolStandardize import rdMolStandardize
        except ImportError:
            from rdkit.Chem import rdMolStandardize  # 移設に備えた保険
    except ImportError:
        return _missing("rdkit")

    results, failures = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failures.append(smi)
            results.append({"input": smi, "canonical": None, "error": "parse_failed"})
            continue
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        results.append({"input": smi, "canonical": Chem.MolToSmiles(mol), "error": None})

    status = "success" if not failures else ("partial" if len(failures) < len(smiles) else "failed")
    return ToolResult(
        status=status,
        summary=f"standardized {len(smiles) - len(failures)}/{len(smiles)} SMILES",
        data={"results": results, "failures": failures},
        error_type="invalid_smiles" if failures else None,
    )


# ---------------------------------------------------------------------------
# 3. generate_3d_structure
# ---------------------------------------------------------------------------

def generate_3d_structure(workspace: Path, smiles: str, name: str = "molecule") -> ToolResult:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return _missing("rdkit")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ToolResult(status="failed", summary=f"invalid SMILES: {smiles}",
                          retryable=False, error_type="invalid_smiles")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        return ToolResult(status="failed", summary=f"3D embedding failed for {smiles}",
                          retryable=True, error_type="embedding_failed")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), smiles]
    atoms = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append((atom.GetSymbol(), pos.x, pos.y, pos.z))
        lines.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    xyz_path = workspace / f"{name}.xyz"
    xyz_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return ToolResult(
        status="success",
        summary=f"3D structure written to {xyz_path.name} ({mol.GetNumAtoms()} atoms)",
        data={"xyz_path": str(xyz_path),
              "atoms": [{"symbol": s, "x": x, "y": y, "z": z} for s, x, y, z in atoms]},
        artifacts=[_artifact(xyz_path)],
    )


# ---------------------------------------------------------------------------
# 4. calculate_rdkit_descriptors
# ---------------------------------------------------------------------------

def calculate_rdkit_descriptors(
    workspace: Path, smiles: list[str], output_csv: str = "rdkit_descriptors.csv"
) -> ToolResult:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    except ImportError:
        return _missing("rdkit")

    rows, failures = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failures.append(smi)
            continue
        rows.append({
            "smiles": smi,
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

    if not rows:
        return ToolResult(status="failed", summary="no valid SMILES",
                          data={"failures": failures}, error_type="invalid_smiles")

    import csv as _csv
    out = workspace / output_csv
    with out.open("w", newline="", encoding="utf-8") as fp:
        writer = _csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return ToolResult(
        status="success" if not failures else "partial",
        summary=f"descriptors for {len(rows)}/{len(smiles)} molecules → {out.name}",
        data={"output_csv": str(out), "failures": failures},
        artifacts=[_artifact(out)],
    )


# ---------------------------------------------------------------------------
# 5. cross_validate_model
# ---------------------------------------------------------------------------

def cross_validate_model(
    workspace: Path,
    features_csv: str,
    target_column: str,
    model: str = "random_forest",
    n_folds: int = 5,
    drop_columns: list[str] | None = None,
) -> ToolResult:
    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import KFold
    except ImportError:
        return _missing("scikit-learn")

    f = _resolve_input(workspace, features_csv)
    if not f.exists():
        return ToolResult(status="failed", summary=f"features file not found: {features_csv}",
                          error_type="input_not_found")
    df = pd.read_csv(f)
    if target_column not in df.columns:
        return ToolResult(status="failed",
                          summary=f"target column `{target_column}` not in {list(df.columns)}",
                          error_type="invalid_input")

    drop = set(drop_columns or []) | {target_column}
    X = df.drop(columns=[c for c in drop if c in df.columns]).select_dtypes("number")
    y = df[target_column].to_numpy()
    if X.shape[1] == 0:
        return ToolResult(status="failed", summary="no numeric feature columns remain",
                          error_type="invalid_input")
    mask = ~(X.isna().any(axis=1) | pd.isna(y))
    X, y = X[mask].to_numpy(), y[mask]
    if len(y) < n_folds:
        return ToolResult(status="failed",
                          summary=f"only {len(y)} usable rows (< {n_folds} folds)",
                          error_type="insufficient_data")

    est = (Ridge() if model == "ridge"
           else RandomForestRegressor(n_estimators=300, random_state=42))
    oof = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in KFold(n_splits=n_folds, shuffle=True, random_state=42).split(X):
        est.fit(X[train_idx], y[train_idx])
        oof[test_idx] = est.predict(X[test_idx])

    metrics = {
        "model": model,
        "n_folds": n_folds,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "r2": float(r2_score(y, oof)),
        "rmse": float(np.sqrt(mean_squared_error(y, oof))),
        "mae": float(mean_absolute_error(y, oof)),
    }
    metrics_path = workspace / "cv_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred_path = workspace / "oof_predictions.csv"
    pd.DataFrame({"y_true": y, "y_pred": oof}).to_csv(pred_path, index=False)

    artifacts = [_artifact(metrics_path), _artifact(pred_path)]
    plot_path = workspace / "true_vs_pred.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y, oof, alpha=0.7)
        lims = [min(y.min(), oof.min()), max(y.max(), oof.max())]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
        ax.set_title(f"{model} CV: R2={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3g}")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        artifacts.append(_artifact(plot_path))
    except ImportError:
        pass

    return ToolResult(
        status="success",
        summary=f"{n_folds}-fold CV: R2={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3g}",
        data=metrics,
        artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# 6. inspect_artifact
# ---------------------------------------------------------------------------

def inspect_artifact(workspace: Path, path: str, max_bytes: int = 4000) -> ToolResult:
    f = _resolve_input(workspace, path)
    if not f.exists():
        return ToolResult(status="failed", summary=f"artifact not found: {path}",
                          error_type="input_not_found")
    stat = f.stat()
    data: dict = {"path": str(f), "bytes": stat.st_size, "suffix": f.suffix}
    if f.suffix.lower() in (".csv", ".json", ".txt", ".md", ".xyz", ".log"):
        text = f.read_text(encoding="utf-8", errors="replace")
        data["preview"] = text[:max_bytes]
        data["truncated"] = len(text) > max_bytes
    return ToolResult(status="success", summary=f"{f.name}: {stat.st_size} bytes", data=data)


# ---------------------------------------------------------------------------
# 7. run_python_sandbox  (registry.py で policy + sandbox を束縛して構築)
# ---------------------------------------------------------------------------

def run_python_sandbox(workspace: Path, code: str, *, sandbox, policy) -> ToolResult:
    blocked = policy.check_code(code)
    if blocked is not None:
        return blocked
    result = sandbox.run(code)
    artifacts = [_artifact(Path(p)) for p in result.new_files if Path(p).exists()]
    if result.timed_out:
        return ToolResult(
            status="failed",
            summary=f"execution timed out after {sandbox.config.timeout_sec}s。計算を軽くしてください（基底縮小・分子数削減など）。",
            data={"stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]},
            artifacts=artifacts, retryable=True, error_type="timeout",
        )
    if result.returncode != 0:
        error_type = "runtime_error"
        if "ModuleNotFoundError" in result.stderr:
            error_type = "missing_dependency"
        return ToolResult(
            status="failed",
            summary=f"script exited with code {result.returncode}",
            data={"stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]},
            artifacts=artifacts, retryable=error_type != "missing_dependency",
            error_type=error_type,
        )
    return ToolResult(
        status="success",
        summary="script executed successfully",
        data={"stdout": result.stdout[-8000:], "stderr": result.stderr[-2000:],
              "new_files": result.new_files},
        artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# 8. verify_scientific_result
# ---------------------------------------------------------------------------

def verify_scientific_result(workspace: Path, task_type: str = "generic",
                             expected_outputs: list[str] | None = None) -> ToolResult:
    from harness.verifier import ScientificVerifier
    from schemas import TaskSpec

    task = TaskSpec(description="inline verification", task_type=task_type,  # type: ignore[arg-type]
                    expected_outputs=expected_outputs or [])
    verification = ScientificVerifier().verify(task, workspace)
    return ToolResult(
        status="success" if verification.passed else "partial",
        summary="verification passed" if verification.passed
        else f"verification failed: {len(verification.required_repairs)} repairs required",
        data=verification.model_dump(),
    )


# ---------------------------------------------------------------------------
# 9. search_official_documentation
# ---------------------------------------------------------------------------

def search_official_documentation(workspace: Path, query: str, max_results: int = 5) -> ToolResult:
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return ToolResult(
            status="blocked",
            summary="TAVILY_API_KEY is not set — web search is unavailable in this environment",
            retryable=False, error_type="missing_credentials",
        )
    try:
        import httpx
    except ImportError:
        return _missing("httpx")
    response = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": max_results,
              "include_domains": [], "search_depth": "basic"},
        timeout=30,
    )
    response.raise_for_status()
    results = [
        {"title": r.get("title"), "url": r.get("url"), "content": (r.get("content") or "")[:500]}
        for r in response.json().get("results", [])
    ]
    return ToolResult(status="success", summary=f"{len(results)} results for: {query}",
                      data={"results": results})
