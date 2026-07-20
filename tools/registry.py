"""Tool Registry。

共通ツールを name → ToolSpec で保持し、各 Adapter が SDK 固有のツール形式へ変換する。
呼び出しは必ず ToolRegistry.call() を通し、例外も ToolResult(failed) に正規化する。
"""
from __future__ import annotations

import functools
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from schemas import SandboxConfig, ToolResult
from tools import chem
from tools.sandbox import create_sandbox


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (properties/required)
    func: Callable[..., ToolResult]
    risk_level: str = "low"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[n] for n in self.names()]

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        if name not in self._tools:
            return ToolResult(status="failed", summary=f"unknown tool: {name}",
                              retryable=False, error_type="unknown_tool")
        try:
            return self._tools[name].func(**kwargs)
        except Exception as e:
            return ToolResult(
                status="failed",
                summary=f"{type(e).__name__}: {e}",
                data={"traceback": traceback.format_exc()[-3000:]},
                retryable=True,
                error_type="tool_exception",
            )


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


def build_default_registry(workspace: Path, sandbox_config: SandboxConfig, policy) -> ToolRegistry:
    """workspace / sandbox / policy を束縛した既定のツール群を構築する。"""
    workspace = Path(workspace)
    sandbox = create_sandbox(sandbox_config, workspace)
    registry = ToolRegistry()
    bind = functools.partial

    smiles_list = {"type": "array", "items": {"type": "string"},
                   "description": "SMILES strings"}

    registry.register(ToolSpec(
        name="inspect_dataset",
        description="CSVデータセットの行数・列型・欠損・統計量・先頭行を調べる。",
        parameters=_schema({"path": {"type": "string", "description": "CSV path"}}, ["path"]),
        func=bind(chem.inspect_dataset, workspace),
    ))
    registry.register(ToolSpec(
        name="standardize_smiles",
        description="SMILESをRDKitで標準化（Cleanup + FragmentParent）し正準SMILESを返す。",
        parameters=_schema({"smiles": smiles_list}, ["smiles"]),
        func=bind(chem.standardize_smiles, workspace),
    ))
    registry.register(ToolSpec(
        name="generate_3d_structure",
        description="SMILESから3D構造を生成（ETKDGv3 + MMFF最適化）し、xyzファイルを保存する。",
        parameters=_schema({
            "smiles": {"type": "string"},
            "name": {"type": "string", "description": "output basename (default: molecule)"},
        }, ["smiles"]),
        func=bind(chem.generate_3d_structure, workspace),
    ))
    registry.register(ToolSpec(
        name="calculate_orbitals",
        description=(
            "RDKit+PySCFで各分子のHOMO/LUMO/gap (eV) と全エネルギーを計算し "
            "orbital_features.csv に保存する。method は 'HF' または DFT 汎関数名 (例 'b3lyp')。"
        ),
        parameters=_schema({
            "smiles": smiles_list,
            "method": {"type": "string", "default": "HF"},
            "basis": {"type": "string", "default": "sto-3g"},
            "charge": {"type": "integer", "default": 0},
            "spin": {"type": "integer", "default": 0},
            "output_csv": {"type": "string", "default": "orbital_features.csv"},
        }, ["smiles"]),
        func=bind(chem.calculate_orbitals, workspace),
        risk_level="medium",
    ))
    registry.register(ToolSpec(
        name="calculate_rdkit_descriptors",
        description="RDKit記述子（MolWt, LogP, TPSA等）を計算しCSVに保存する。",
        parameters=_schema({
            "smiles": smiles_list,
            "output_csv": {"type": "string", "default": "rdkit_descriptors.csv"},
        }, ["smiles"]),
        func=bind(chem.calculate_rdkit_descriptors, workspace),
    ))
    registry.register(ToolSpec(
        name="cross_validate_model",
        description=(
            "特徴量CSVからtarget列を予測する回帰モデルをK-fold交差検証し、"
            "cv_metrics.json / oof_predictions.csv / true_vs_pred.png を生成する。"
        ),
        parameters=_schema({
            "features_csv": {"type": "string"},
            "target_column": {"type": "string"},
            "model": {"type": "string", "enum": ["random_forest", "ridge"], "default": "random_forest"},
            "n_folds": {"type": "integer", "default": 5},
            "drop_columns": {"type": "array", "items": {"type": "string"}},
        }, ["features_csv", "target_column"]),
        func=bind(chem.cross_validate_model, workspace),
    ))
    registry.register(ToolSpec(
        name="inspect_artifact",
        description="workspace内の生成物（CSV/JSON/画像等）のサイズとテキストプレビューを取得する。",
        parameters=_schema({"path": {"type": "string"}}, ["path"]),
        func=bind(chem.inspect_artifact, workspace),
    ))
    registry.register(ToolSpec(
        name="run_python_sandbox",
        description=(
            "任意のPythonコードをsandbox（docker/local, timeout・メモリ制限付き）で実行する。"
            "ファイル出力はカレントディレクトリ（workspace）へ。画像はAggバックエンドでsavefigすること。"
        ),
        parameters=_schema({"code": {"type": "string", "description": "self-contained Python script"}},
                           ["code"]),
        func=lambda code: chem.run_python_sandbox(workspace, code, sandbox=sandbox, policy=policy),
        risk_level="medium",
    ))
    registry.register(ToolSpec(
        name="verify_scientific_result",
        description="workspaceの結果をScientific Verifierで検査し、不足要件と科学的警告を返す。",
        parameters=_schema({
            "task_type": {"type": "string", "default": "generic"},
            "expected_outputs": {"type": "array", "items": {"type": "string"}},
        }, []),
        func=bind(chem.verify_scientific_result, workspace),
    ))
    registry.register(ToolSpec(
        name="search_official_documentation",
        description="公式ドキュメント・リリースノートをWeb検索する（TAVILY_API_KEY が必要）。",
        parameters=_schema({
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 5},
        }, ["query"]),
        func=bind(chem.search_official_documentation, workspace),
    ))
    return registry
