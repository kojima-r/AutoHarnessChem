"""Tool Registry。

共通ツールを name → ToolSpec で保持し、各 Adapter が SDK 固有のツール形式へ変換する。
呼び出しは必ず ToolRegistry.call() を通し、例外も ToolResult(failed) に正規化する。
"""
from __future__ import annotations

import functools
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from schemas import SandboxConfig, ToolResult, utcnow
from tools import aizynth, chem, chemenv, opttddft, reactiont5, report
from tools.sandbox import create_sandbox

# ツール失敗の詳細（stderr / stdout / traceback）を後から確認できるように残すログ。
# trace（AgentEvent）には summary と error_type しか載らないため、原因調査には
# こちらを読む。1 行 1 失敗の JSON Lines で workspace 直下へ追記する。
ERROR_LOG_NAME = "tool_errors.jsonl"


def _clip(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + "... [truncated]"


def _loggable(value: Any, limit: int = 8000) -> Any:
    """JSON にできる値はそのまま、長い文字列や巨大な構造は切り詰めて残す。"""
    if isinstance(value, str):
        return _clip(value, limit)
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return _clip(value, limit)
    return value if len(text) <= limit else _clip(text, limit)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (properties/required)
    func: Callable[..., ToolResult]
    risk_level: str = "low"


class ToolRegistry:
    def __init__(self, error_log: Path | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        # 失敗の詳細を追記するファイル（None なら記録しない）
        self.error_log = Path(error_log) if error_log is not None else None

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[n] for n in self.names()]

    # ツール名は位置専用にする（`name` という引数を持つツール — 例
    # generate_3d_structure(name=...) — を呼べるようにするため）
    def call(self, name: str, /, **kwargs: Any) -> ToolResult:
        if name not in self._tools:
            result = ToolResult(status="failed", summary=f"unknown tool: {name}",
                                retryable=False, error_type="unknown_tool")
        else:
            try:
                result = self._tools[name].func(**kwargs)
            except Exception as e:
                result = ToolResult(
                    status="failed",
                    summary=f"{type(e).__name__}: {e}",
                    data={"traceback": traceback.format_exc()[-3000:]},
                    retryable=True,
                    error_type="tool_exception",
                )
        self.log_failure(name, kwargs, result)
        return result

    def log_failure(self, name: str, arguments: dict[str, Any], result: ToolResult) -> None:
        """成功以外の呼び出しを error_log へ 1 行追記する（stderr をここに残す）。

        ToolResult.data の stdout / stderr / traceback は agent には渡るが trace には
        残らないため、後から原因を調べられるようにここで永続化する。
        ログの失敗でツール実行を壊さないよう、例外は握りつぶす。
        """
        if (self.error_log is None or not isinstance(result, ToolResult)
                or result.status == "success"):
            return
        entry = {
            "at": utcnow().isoformat(),
            "tool": name,
            "status": result.status,
            "error_type": result.error_type,
            "retryable": result.retryable,
            "summary": result.summary,
            "arguments": {k: _loggable(v, 500) for k, v in arguments.items()},
            "data": {k: _loggable(v) for k, v in result.data.items()},
        }
        try:
            self.error_log.parent.mkdir(parents=True, exist_ok=True)
            with self.error_log.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


def build_default_registry(workspace: Path, sandbox_config: SandboxConfig, policy) -> ToolRegistry:
    """workspace / sandbox / policy を束縛した既定のツール群を構築する。"""
    workspace = Path(workspace)
    sandbox = create_sandbox(sandbox_config, workspace)
    # 失敗の詳細（stderr 等）は workspace/tool_errors.jsonl に残す
    registry = ToolRegistry(error_log=workspace / ERROR_LOG_NAME)
    bind = functools.partial

    smiles_list = {"type": "array", "items": {"type": "string"},
                   "description": "SMILES strings"}

    # RDKit / pandas / scikit-learn 系も専用環境で実行する（harness 環境に
    # これらの依存を持たせない。既定は conda env `pyscf`）
    chem_sandbox = create_sandbox(sandbox_config, workspace, env=chemenv.ENV_NAME)
    registry.register(ToolSpec(
        name="inspect_dataset",
        description=("CSVデータセットの行数・列型・欠損・統計量・先頭行を調べる"
                     "（pandas を持つ専用環境で実行）。"),
        parameters=_schema({"path": {"type": "string", "description": "CSV path"}}, ["path"]),
        func=lambda **kw: chemenv.inspect_dataset(workspace, sandbox=chem_sandbox, **kw),
    ))
    registry.register(ToolSpec(
        name="standardize_smiles",
        description=("SMILESをRDKitで標準化（Cleanup + FragmentParent）し正準SMILESを返す"
                     "（RDKit を持つ専用環境で実行）。"),
        parameters=_schema({"smiles": smiles_list}, ["smiles"]),
        func=lambda **kw: chemenv.standardize_smiles(workspace, sandbox=chem_sandbox, **kw),
    ))
    registry.register(ToolSpec(
        name="generate_3d_structure",
        description=("SMILESから3D構造を生成（ETKDGv3 + MMFF最適化）し、xyzファイルを保存する。"
                     "量子化学ツールは内部で構造生成するので、xyz を成果物として残したい場合や "
                     "scan_esipt_pes の初期構造を用意する場合に使う。"),
        parameters=_schema({
            "smiles": {"type": "string"},
            "name": {"type": "string", "description": "output basename (default: molecule)"},
        }, ["smiles"]),
        func=lambda **kw: chemenv.generate_3d_structure(workspace, sandbox=chem_sandbox, **kw),
    ))
    # 量子化学計算は OptTDDFT（tools/OptTDDFT）を専用の pyscf 環境で実行する。
    # 構造最適化（geomeTRIC）が必要な呼び出しだけ、geomeTRIC を持つ環境へ振り分ける
    qc_sandbox = create_sandbox(sandbox_config, workspace, env="opttddft")
    geom_sandbox = create_sandbox(sandbox_config, workspace, env="esipt")

    def qc_env(arguments: dict):
        """use_geom_opt=true なら geomeTRIC のある環境を使う。"""
        return geom_sandbox if arguments.get("use_geom_opt") else qc_sandbox
    solvent = {"type": "string", "enum": ["none", "pcm"], "default": "none",
               "description": "PCM 溶媒モデルを使うか"}
    threads = {"type": "integer", "default": 4,
               "description": "PySCF に許可するスレッド数（全コア占有を防ぐ）"}
    timeout = {"type": "integer",
               "description": "この呼び出しの実行上限秒（既定は sandbox の timeout）"}
    memory = {"type": "integer", "default": opttddft.DEFAULT_MEMORY_LIMIT_MB,
              "description": ("メモリ上限 (MB)。SIGSEGV / out_of_memory で落ちる場合は"
                              "上げる（基底関数を大きくすると必要量が増える）")}

    registry.register(ToolSpec(
        name="calculate_orbitals",
        description=(
            "OptTDDFT(PySCF) で各分子のHOMO/LUMO/gap (eV) と全エネルギーを計算し "
            "orbital_features.csv に保存する。method は 'HF' または DFT 汎関数名 "
            "(例 'b3lyp')。3D構造は RDKit の多コンフォマー探索+UFF で生成。"
            "励起状態は計算しない（必要なら calculate_tddft_spectrum）。"
            "同じ output_csv へ複数回呼ぶと結果は累積される（同じ smiles/method/basis の"
            "行だけ置換）ので、1分子ずつ呼んでも前の結果は消えない。専用conda環境(pyscf)で実行。"
        ),
        parameters=_schema({
            "smiles": smiles_list,
            "method": {"type": "string", "default": "HF"},
            "basis": {"type": "string", "default": "sto-3g"},
            "charge": {"type": "integer", "default": 0},
            "spin": {"type": "integer", "default": 0,
                     "description": "2S（不対電子数）。0 以外なら UHF/UKS で計算する"},
            "solvent_model": solvent,
            "solvent_eps": {"type": "number", "default": 4.7113},
            "use_geom_opt": {"type": "boolean", "default": False,
                             "description": ("SCF 前に構造最適化する（高コスト）。true にすると"
                                             "geomeTRIC を持つ専用環境で実行される")},
            "max_cycle": {"type": "integer", "default": 200},
            "output_csv": {"type": "string", "default": "orbital_features.csv"},
            "timeout_sec": timeout,
            "threads": threads,
            "memory_limit_mb": memory,
        }, ["smiles"]),
        func=lambda **kw: opttddft.calculate_orbitals(workspace, sandbox=qc_env(kw), **kw),
        risk_level="medium",
    ))
    registry.register(ToolSpec(
        name="calculate_tddft_spectrum",
        description=(
            "OptTDDFT の TDDFT で励起波長 (nm)・振動子強度・HOMO/LUMO を計算し、"
            "tddft_spectrum.csv（1状態1行）と orbital_features.csv、UV-Vis スペクトル画像を"
            "生成する。既定は CAMB3LYP/6-31g(d)。重いので分子数と nstates は控えめにし、"
            "必要に応じて timeout_sec を上げること。閉殻分子のみ（RKS）。"
            "同じ CSV へ複数回呼ぶと結果は累積される。専用conda環境(pyscf)で実行。"
        ),
        parameters=_schema({
            "smiles": smiles_list,
            "functional": {"type": "string", "default": "CAMB3LYP"},
            "basis": {"type": "string", "default": "6-31g(d)"},
            "nstates": {"type": "integer", "default": 10,
                        "description": "計算する励起状態数。状態が足りない/収束が悪いときは "
                                       "15〜24 へ増やす（減らすと収束が悪化することがある）"},
            "charge": {"type": "integer", "default": 0},
            "solvent_model": solvent,
            "solvent_eps": {"type": "number", "default": 4.7113},
            "use_geom_opt": {"type": "boolean", "default": False},
            "output_csv": {"type": "string", "default": "tddft_spectrum.csv"},
            "orbital_csv": {"type": "string", "default": "orbital_features.csv"},
            "plot": {"type": "boolean", "default": True},
            "timeout_sec": timeout,
            "threads": threads,
            "memory_limit_mb": memory,
        }, ["smiles"]),
        func=lambda **kw: opttddft.calculate_tddft_spectrum(
            workspace, sandbox=qc_env(kw), **kw),
        risk_level="medium",
    ))
    registry.register(ToolSpec(
        name="optimize_absorption_wavelength",
        description=(
            "骨格SMILES（ダミー原子 [*:1] [*:2] を持つ）と置換基候補から分子を組み立て、"
            "TDDFT の吸収波長が target_wavelength_nm に最も近い分子を Optuna(TPE) で探索する。"
            "既定では振動子強度が最大の吸収帯（= 実測される帯）を目標に合わせる"
            "（objective='longest' で最長波長にできるが、暗状態を追いやすい）。"
            "optuna_trials.csv / optimization_summary.json / <study_name>.db を生成し、"
            "generate_report=true で Excel・PowerPoint・スペクトル画像も出力する。"
            "1 trial が TDDFT 1 回分のコストなので n_trials と search_timeout_sec で予算を決める。"
            "専用conda環境(pyscf)で実行。"
        ),
        parameters=_schema({
            "scaffold": {"type": "string",
                         "description": "例 'c1cc([*:1])ccc1[*:2]'（[*:1] と [*:2] が必須）"},
            "side_chains_pos1": {"type": "array", "items": {"type": "string"},
                                 "description": "[*:1] 位の置換基候補（'H' は水素）"},
            "side_chains_pos2": {"type": "array", "items": {"type": "string"}},
            "target_wavelength_nm": {"type": "number", "default": 500.0},
            "objective": {"type": "string", "enum": ["strongest", "longest"],
                          "default": "strongest",
                          "description": ("目標に合わせる波長。'strongest'=振動子強度が"
                                          "最大の吸収帯（実測される帯・推奨）、"
                                          "'longest'=最長波長（暗状態になりやすい）")},
            "min_oscillator_strength": {"type": "number", "default": 0.01,
                                        "description": "この強度未満の状態は吸収帯として扱わない"},
            "n_trials": {"type": "integer", "default": 10},
            "study_name": {"type": "string", "default": "tddft_mi_optimization",
                           "description": "SQLite study 名。同名なら過去の探索を再開する"},
            "functional": {"type": "string", "default": "CAMB3LYP"},
            "basis": {"type": "string", "default": "6-31g(d)"},
            "nstates": {"type": "integer", "default": 10},
            "solvent_model": solvent,
            "search_timeout_sec": {"type": "integer", "default": 900,
                                   "description": "探索全体の打ち切り時間（打ち切っても結果は保存される）"},
            "generate_report": {"type": "boolean", "default": False},
            "timeout_sec": timeout,
            "threads": threads,
            "memory_limit_mb": memory,
        }, ["scaffold", "side_chains_pos1", "side_chains_pos2"]),
        func=lambda **kw: opttddft.optimize_absorption_wavelength(
            workspace, sandbox=qc_env(kw), **kw),
        risk_level="medium",
    ))
    registry.register(ToolSpec(
        name="scan_esipt_pes",
        description=(
            "ESIPT（励起状態分子内プロトン移動）の Relaxed PES スキャン。指定した2原子間距離を"
            "拘束して構造最適化 + TDDFT を繰り返し、esipt_scan_results.csv（S0/S1 エネルギー、"
            "相対 kcal/mol）・esipt_pes_profile.png・esipt_scan_summary.json を生成する。"
            "原子インデックスは 0 始まりで、SMILES ではなく既知の XYZ 座標から指定すること。"
            "拘束付き構造最適化を含むため、geomeTRIC を持つ専用conda環境(pyscf_esipt)で実行。"
        ),
        parameters=_schema({
            "atom_idx_1": {"type": "integer", "description": "移動するプロトン H の index（0始まり）"},
            "atom_idx_2": {"type": "integer", "description": "アクセプター原子（N/O 等）の index"},
            "xyz": {"type": "string",
                    "description": "初期構造（'元素記号 X Y Z' を改行区切り）。xyz_file と排他"},
            "xyz_file": {"type": "string", "description": "workspace 内の XYZ ファイル"},
            "start_dist": {"type": "number", "default": 1.0},
            "end_dist": {"type": "number", "default": 2.0},
            "step_size": {"type": "number", "default": 0.1},
            "functional": {"type": "string", "default": "CAMB3LYP"},
            "basis": {"type": "string", "default": "6-31g(d)"},
            "nstates": {"type": "integer", "default": 5},
            "solvent_model": solvent,
            "opt_max_steps": {"type": "integer", "default": 50},
            "timeout_sec": timeout,
            "threads": threads,
            "memory_limit_mb": memory,
        }, ["atom_idx_1", "atom_idx_2"]),
        # 拘束付き構造最適化が必須なので、常に geomeTRIC のある環境で実行する
        func=lambda **kw: opttddft.scan_esipt_pes(workspace, sandbox=geom_sandbox, **kw),
        risk_level="medium",
    ))
    registry.register(ToolSpec(
        name="calculate_rdkit_descriptors",
        description="RDKit記述子（MolWt, LogP, TPSA等）を計算しCSVに保存する。",
        parameters=_schema({
            "smiles": smiles_list,
            "output_csv": {"type": "string", "default": "rdkit_descriptors.csv"},
        }, ["smiles"]),
        func=lambda **kw: chemenv.calculate_rdkit_descriptors(
            workspace, sandbox=chem_sandbox, **kw),
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
            "memory_limit_mb": {"type": "integer",
                                "default": chemenv.DEFAULT_MEMORY_LIMIT_MB,
                                "description": "メモリ上限 (MB)。大きなデータでは上げる"},
        }, ["features_csv", "target_column"]),
        func=lambda **kw: chemenv.cross_validate_model(
            workspace, sandbox=chem_sandbox, **kw),
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
    # ReactionT5 は torch/transformers 依存のため、pyscf とは別の専用環境で実行する
    t5_sandbox = create_sandbox(sandbox_config, workspace, env="reactiont5")
    registry.register(ToolSpec(
        name="predict_reaction_t5",
        description=(
            "ReactionT5v2 学習済みモデルで反応を予測する。task: 'yield'=収率回帰 (0-100%), "
            "'forward'=生成物予測, 'retrosynthesis'=逆合成（前駆体予測）。"
            "入力形式 — yield: 'REACTANT:...REAGENT:...PRODUCT:...', "
            "forward: 'REACTANT:...REAGENT:...', retrosynthesis: 生成物SMILESのみ。"
            "結果は reactiont5_predictions.csv に保存される（1 反応ごとに途中結果を"
            "残すので、打ち切られても完了分は status='partial' + data.pending で返る）。"
            "専用conda環境(reactiont5)で実行。"
        ),
        parameters=_schema({
            "reactions": {"type": "array", "items": {"type": "string"},
                          "description": "モデル入力文字列のリスト（形式は task に依存）"},
            "task": {"type": "string", "enum": ["yield", "forward", "retrosynthesis"],
                     "default": "forward"},
            "num_beams": {"type": "integer", "default": 1,
                          "description": "forward/retrosynthesis のビーム幅（候補数）"},
            "output_csv": {"type": "string", "default": "reactiont5_predictions.csv"},
            "timeout_sec": timeout,
            "memory_limit_mb": {"type": "integer",
                                "default": reactiont5.DEFAULT_MEMORY_LIMIT_MB,
                                "description": ("メモリ上限 (MB)。torch は import と "
                                                "CUDA 初期化で大量のアドレス空間を要求"
                                                "するため、不足すると OpenBLAS の確保"
                                                "エラーや `CUDA error: out of memory` "
                                                "で落ちる（GPU 実行には 49152 以上）")},
        }, ["reactions", "task"]),
        func=lambda **kw: reactiont5.predict_reaction_t5(workspace, sandbox=t5_sandbox, **kw),
        risk_level="medium",
    ))
    # AiZynthFinder は ONNX モデル + stock DB を持つ別環境で実行する
    aizynth_sandbox = create_sandbox(sandbox_config, workspace, env="aizynth")
    registry.register(ToolSpec(
        name="plan_retrosynthesis",
        description=(
            "AiZynthFinder で目標分子の逆合成経路を多段探索する（expansion policy + stock）。"
            "purchasable な出発物質まで到達した経路を retrosynthesis_routes.json（経路木）と "
            "retrosynthesis_routes.csv（1経路1行の要約）に保存する。"
            "predict_reaction_t5(task='retrosynthesis') が1段階の前駆体予測なのに対し、"
            "こちらは経路全体の探索。1分子あたり time_limit_sec まで探索する。"
            "専用conda環境(aizynth)で実行し、学習済みモデル(config.yml)が必要。"
        ),
        parameters=_schema({
            "targets": {"type": "array", "items": {"type": "string"},
                        "description": "目標分子の SMILES（正準化済みが望ましい）"},
            "algorithm": {"type": "string", "enum": ["mcts", "retrostar"], "default": "mcts"},
            "iteration_limit": {"type": "integer", "default": 100,
                                "description": "探索の反復上限（増やすと解けやすいが遅い）"},
            "time_limit_sec": {"type": "integer", "default": 120,
                               "description": "1分子あたりの探索時間上限"},
            "max_transforms": {"type": "integer", "default": 6,
                               "description": "経路の最大段数"},
            "n_routes": {"type": "integer", "default": 5,
                         "description": "1分子あたり保存する上位経路数"},
            "stock": {"type": "array", "items": {"type": "string"},
                      "description": "使う stock 名（既定: config.yml の全件）"},
            "expansion": {"type": "array", "items": {"type": "string"},
                          "description": "使う expansion policy 名（既定: 先頭のもの）"},
            "filter_policy": {"type": "array", "items": {"type": "string"},
                              "description": "反応の実現性フィルタ（既定: 未使用）"},
            "config_yaml": {"type": "string",
                            "description": "AiZynthFinder の config.yml（既定: AIZYNTH_CONFIG 等から自動解決）"},
            "timeout_sec": {"type": "integer"},
            "memory_limit_mb": {"type": "integer",
                                "default": aizynth.DEFAULT_MEMORY_LIMIT_MB,
                                "description": "stock DB と ONNX を載せるためのメモリ上限"},
        }, ["targets"]),
        func=lambda **kw: aizynth.plan_retrosynthesis(
            workspace, sandbox=aizynth_sandbox, **kw),
        risk_level="medium",
    ))
    registry.register(ToolSpec(
        name="render_report_html",
        description=(
            "report_user.md を構造式付きの report_user.html に変換する。SmilesDrawer を"
            "HTML に埋め込むのでオフラインでも構造が描画される。Markdown 側では "
            "```smiles フェンス（1行 = 'SMILES ラベル'、`>` を含む行は反応式）と、"
            "文中の `smiles:<SMILES>` が構造になる。auto_structures=true（既定）なら "
            "workspace の CSV の SMILES 列からも構造一覧を自動生成する。"
        ),
        parameters=_schema({
            "markdown_path": {"type": "string", "default": "report_user.md"},
            "output_html": {"type": "string", "default": "report_user.html"},
            "auto_structures": {"type": "boolean", "default": True,
                                "description": "成果物CSVのSMILES列から構造一覧を追加する"},
            "inline_library": {"type": "boolean", "default": True,
                               "description": "SmilesDrawer を埋め込む（false なら Web UI 配信URLを参照）"},
        }, []),
        func=bind(report.render_report_html, workspace),
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
