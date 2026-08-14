"""共通スキーマ。

TaskSpec / ToolResult / VerificationResult / AgentEvent を定義する。
SDK固有のメッセージはすべて AgentEvent へ正規化される。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


TaskType = Literal[
    "orbital_calculation",      # HOMO/LUMO・TDDFT スペクトル（OptTDDFT）
    "molecular_design",         # 目標物性に近い分子の探索（Optuna + TDDFT）
    "pes_scan",                 # ESIPT 等の PES スキャン
    "molecular_regression",
    "reaction_prediction",      # ReactionT5（収率・生成物・1段階逆合成）
    "retrosynthesis_planning",  # AiZynthFinder による多段の逆合成経路探索
    "dataset_analysis",
    "code_editing",
    "long_running_research",
    "generic",
]

Provider = Literal["deepagents", "claude", "openai"]


class Artifact(BaseModel):
    path: str
    mime: str = "application/octet-stream"
    bytes: int = 0
    sha256: str | None = None
    kind: Literal["data", "figure", "report", "model", "log", "other"] = "other"


class ToolResult(BaseModel):
    status: Literal["success", "partial", "failed", "blocked"]
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    retryable: bool = False
    error_type: str | None = None


class TaskSpec(BaseModel):
    task_id: str = Field(default_factory=lambda: new_id("task"))
    description: str
    task_type: TaskType = "generic"
    # 複合タスク（例: 分子設計 + TDDFT + 逆合成 + 報告）で追加検出された側面。
    # Skill 選択と Verifier のドメイン検査は primary + secondary の両方を対象にする。
    # 期待出力（= 必須要件）は primary のものだけを使う（誤検出で不合格にしないため）
    secondary_task_types: list[TaskType] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    required_skills: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    # Verifier 不合格時に required_repairs がここへ入り、再計画プロンプトに反映される
    repairs: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    passed: bool
    requirements_satisfied: list[str] = Field(default_factory=list)
    requirements_missing: list[str] = Field(default_factory=list)
    scientific_warnings: list[str] = Field(default_factory=list)
    required_repairs: list[str] = Field(default_factory=list)


class AgentEvent(BaseModel):
    run_id: str
    event_type: Literal[
        "reasoning_summary",
        "tool_call",
        "tool_result",
        "delegation",
        "artifact",
        "error",
        "final",
    ]
    actor: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utcnow)


class SandboxConfig(BaseModel):
    type: Literal["docker", "local"] = "docker"
    image: str = "autoharnesschem/sandbox:latest"
    conda_env: str | None = "pyscf"  # local sandbox 用。None なら現在の python を使う
    timeout_sec: int = 600
    cpu_limit_sec: int = 300
    memory_limit_mb: int = 4096
    network: Literal["none", "bridge"] = "none"
    # ツール専用の実行環境。依存が競合するツール（例: ReactionT5 の torch/transformers、
    # AiZynthFinder の ONNX ランタイム）を既定環境から分離する。
    # local は conda_env、docker は image で解決される
    named_envs: dict[str, dict[str, str]] = Field(default_factory=lambda: {
        "opttddft": {"conda_env": "pyscf",
                     "image": "autoharnesschem/opttddft:latest"},
        # RDKit / pandas / scikit-learn 系ツール（量子化学環境に同居している）
        "rdkit": {"conda_env": "pyscf",
                  "image": "autoharnesschem/opttddft:latest"},
        # 構造最適化を伴う計算（ESIPT の PES スキャン / use_geom_opt）用。
        # geomeTRIC を含む環境を分けておく（既定環境の numpy ピンを崩さないため）
        "esipt": {"conda_env": "pyscf_esipt",
                  "image": "autoharnesschem/opttddft:latest"},
        "reactiont5": {"conda_env": "reactiont5",
                       "image": "autoharnesschem/reactiont5:latest"},
        "aizynth": {"conda_env": "aizynth",
                    "image": "autoharnesschem/aizynth:latest"},
    })


class RuntimeConfig(BaseModel):
    provider: Provider = "deepagents"
    model: str = "default"
    max_steps: int = 60
    max_tool_calls: int = 40
    max_replans: int = 3
    # 1 試行の実時間上限。超えたときの扱いは on_attempt_timeout で決める
    attempt_timeout_sec: int = 3600
    # ask   = ユーザに「さらに待つか」を確認して延長する（応答できない環境では stop）
    # extend= 確認せず自動で延長する（無人の長時間実行向け）
    # stop  = 打ち切り、その時点の成果物で検証・報告する
    on_attempt_timeout: Literal["ask", "extend", "stop"] = "ask"
    # 1 回の延長で追加する秒数（ユーザが秒数を指定した場合はそれが優先）
    timeout_extension_sec: int = 3600
    # 延長回数の上限。None なら無制限（数日かかる計算を待てるようにする）
    max_timeout_extensions: int | None = None
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)


class RunState(BaseModel):
    run_id: str = Field(default_factory=lambda: new_id("run"))
    provider: Provider
    status: Literal[
        "pending", "running", "awaiting_decision", "verifying", "repairing",
        "succeeded", "failed", "blocked",
    ] = "pending"
    attempts: int = 0
    workspace: str = ""
    session_ref: str | None = None  # SDK側の session/thread id（resume 用）
    # 実時間上限を延長した回数と、延長で足した合計秒数
    timeout_extensions: int = 0
    extended_sec: int = 0


class RunReport(BaseModel):
    run_id: str
    task: TaskSpec
    provider: Provider
    passed: bool
    attempts: int
    # 実時間上限を延長した回数 / 追加した秒数（長時間実行の記録）
    timeout_extensions: int = 0
    extended_sec: int = 0
    verification: VerificationResult
    artifacts: list[Artifact] = Field(default_factory=list)
    final_message: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
