"""Adapter factory。provider 名から Adapter を生成し、フォールバック順を解決する。"""
from __future__ import annotations

from adapters.base import AdapterUnavailable, BaseAdapter

PROVIDERS = ("deepagents", "claude", "openai")


def create_adapter(provider: str, tracer, policy) -> BaseAdapter:
    if provider == "claude":
        from adapters.claude import ClaudeAgentAdapter
        return ClaudeAgentAdapter(tracer, policy)
    if provider == "openai":
        from adapters.openai import OpenAIAgentsAdapter
        return OpenAIAgentsAdapter(tracer, policy)
    if provider == "deepagents":
        from adapters.deepagents import DeepAgentsAdapter
        return DeepAgentsAdapter(tracer, policy)
    raise ValueError(f"unknown provider: {provider} (expected one of {PROVIDERS})")


__all__ = ["AdapterUnavailable", "BaseAdapter", "create_adapter", "PROVIDERS"]
