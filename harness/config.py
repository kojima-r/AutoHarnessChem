"""設定のロード。config/default.yaml をベースに、指定ファイルで上書きする。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from schemas import RuntimeConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class PromotionGate(BaseModel):
    minimum_task_success_improvement: float = 0.05
    maximum_cost_increase: float = 0.15
    maximum_latency_increase: float = 0.20
    zero_critical_regressions: bool = True
    security_tests_passed: bool = True
    human_approval_required: bool = True


class PathsConfig(BaseModel):
    root: Path = REPO_ROOT
    skills: Path = REPO_ROOT / "skills"
    workspaces: Path = REPO_ROOT / "workspaces"
    traces: Path = REPO_ROOT / "traces"
    benchmarks: Path = REPO_ROOT / "benchmarks"
    proposals: Path = REPO_ROOT / "evolver" / "proposals"


class HarnessConfig(BaseModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    # task_type -> provider。実測ベンチマーク結果から更新される
    routing: dict[str, str] = Field(default_factory=lambda: {
        "code_editing": "claude",
        "long_running_research": "deepagents",
        "hosted_sandbox_execution": "openai",
        "fallback": "deepagents",
    })
    promotion_gate: PromotionGate = Field(default_factory=PromotionGate)
    mode: Literal["development", "production"] = "development"
    self_improvement: bool = True
    skill_lockfile: str = "skills.lock"
    runtime_fallback: bool = True
    # high risk アクションの承認方式: interactive=CLIで確認 / deny=自動拒否 / allow=自動許可
    approval: Literal["interactive", "deny", "allow"] = "deny"
    paths: PathsConfig = Field(default_factory=PathsConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> HarnessConfig:
    data: dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if path is not None:
        override = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data = _deep_merge(data, override)
    config = HarnessConfig.model_validate(data)
    if config.mode == "production":
        # 本番では自己改善を強制無効化する
        config.self_improvement = False
    return config
