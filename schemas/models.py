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
        "reactiont5": {"conda_env": "reactiont5",
                       "image": "autoharnesschem/reactiont5:latest"},
        "aizynth": {"conda_env": "aizynth",
                    "image": "autoharnesschem/aizynth:latest"},
    })


class RuntimeConfig(BaseModel):
    provider: Provider = "deepagents"
    model: str = "default"
    max_steps: int = 30
    max_tool_calls: int = 20
    max_replans: int = 3
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)


class RunState(BaseModel):
    run_id: str = Field(default_factory=lambda: new_id("run"))
    provider: Provider
    status: Literal[
        "pending", "running", "verifying", "repairing",
        "succeeded", "failed", "blocked",
    ] = "pending"
    attempts: int = 0
    workspace: str = ""
    session_ref: str | None = None  # SDK側の session/thread id（resume 用）


class RunReport(BaseModel):
    run_id: str
    task: TaskSpec
    provider: Provider
    passed: bool
    attempts: int
    verification: VerificationResult
    artifacts: list[Artifact] = Field(default_factory=list)
    final_message: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
