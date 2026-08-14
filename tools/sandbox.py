"""Sandbox — Agent 生成コードの実行環境。

DockerSandbox: ネットワーク遮断・workspace のみ mount のコンテナで実行（推奨）。
LocalSandbox : conda 環境 or 現在の python で subprocess 実行（rlimit 付き、開発用）。
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from schemas import SandboxConfig

SCRIPT_FILENAME = "script.py"


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    script_path: str
    new_files: list[str] = field(default_factory=list)


class BaseSandbox:
    def __init__(self, config: SandboxConfig, workspace: Path):
        self.config = config
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(self, code: str, script_name: str = SCRIPT_FILENAME) -> SandboxResult:
        before = self._snapshot()
        script_path = self.workspace / script_name
        script_path.write_text(code, encoding="utf-8")
        result = self._execute(script_path)
        result.new_files = sorted(self._snapshot() - before - {str(script_path)})
        return result

    def _snapshot(self) -> set[str]:
        return {str(p) for p in self.workspace.rglob("*") if p.is_file()}

    def _execute(self, script_path: Path) -> SandboxResult:
        raise NotImplementedError


class LocalSandbox(BaseSandbox):
    def _command(self, script_path: Path) -> list[str]:
        if self.config.conda_env and shutil.which("conda"):
            return ["conda", "run", "--no-capture-output", "-n", self.config.conda_env,
                    "python", str(script_path)]
        import sys
        return [sys.executable, str(script_path)]

    def _preexec(self):
        try:
            import resource

            def limits():
                cpu = self.config.cpu_limit_sec
                mem = self.config.memory_limit_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
                resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024,) * 2)

            return limits
        except ImportError:
            return None

    def _execute(self, script_path: Path) -> SandboxResult:
        try:
            cp = subprocess.run(
                self._command(script_path),
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_sec,
                preexec_fn=self._preexec(),
            )
            return SandboxResult(cp.stdout, cp.stderr, cp.returncode, False, str(script_path))
        except subprocess.TimeoutExpired as e:
            return SandboxResult(
                (e.stdout or ""), (e.stderr or "") + "\n[Timed out]",
                124, True, str(script_path),
            )


class DockerSandbox(BaseSandbox):
    def _execute(self, script_path: Path) -> SandboxResult:
        cmd = [
            "docker", "run", "--rm",
            "--network", self.config.network,
            "--memory", f"{self.config.memory_limit_mb}m",
            "--cpus", "2",
            "-v", f"{self.workspace.resolve()}:/workspace",
            "-w", "/workspace",
            self.config.image,
            "python", f"/workspace/{script_path.name}",
        ]
        try:
            cp = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config.timeout_sec
            )
            return SandboxResult(cp.stdout, cp.stderr, cp.returncode, False, str(script_path))
        except subprocess.TimeoutExpired as e:
            return SandboxResult(
                (e.stdout or ""), (e.stderr or "") + "\n[Timed out]",
                124, True, str(script_path),
            )


# docker image の可用性チェック結果（image 名 → 使えるか）。1 run で複数の sandbox を
# 作るため、同じ image を何度も問い合わせない
_DOCKER_IMAGE_CACHE: dict[str, bool] = {}


def docker_image_available(image: str) -> bool:
    """image がローカルにあり docker daemon に到達できるか（結果はキャッシュする）。

    `docker` バイナリがあってもイメージ未ビルド・daemon 停止・権限不足なら
    `docker run` は起動前に失敗する（returncode 125 等）ため、事前に確認して
    LocalSandbox へフォールバックできるようにする。
    """
    if image in _DOCKER_IMAGE_CACHE:
        return _DOCKER_IMAGE_CACHE[image]
    available = False
    if shutil.which("docker"):
        try:
            probe = subprocess.run(["docker", "image", "inspect", image],
                                   capture_output=True, text=True, timeout=20)
            available = probe.returncode == 0
        except (OSError, subprocess.SubprocessError):
            available = False
    _DOCKER_IMAGE_CACHE[image] = available
    return available


def create_sandbox(config: SandboxConfig, workspace: Path,
                   env: str | None = None) -> BaseSandbox:
    """env を指定すると named_envs の設定（conda_env / image）で上書きした
    専用サンドボックスを作る（例: env="reactiont5" → conda env reactiont5 で実行）。"""
    if env is not None:
        override = config.named_envs.get(env, {})
        config = config.model_copy(update={
            "conda_env": override.get("conda_env", env),
            "image": override.get("image", config.image),
        })
    if config.type == "docker":
        if docker_image_available(config.image):
            return DockerSandbox(config, workspace)
        # docker が使えない（未インストール / image 未ビルド / daemon 停止）場合は
        # conda 環境での実行へフォールバックする（runtime_fallback 相当）
        reason = ("docker not found" if not shutil.which("docker")
                  else f"image `{config.image}` is unavailable")
        print(f"[sandbox] {reason} — falling back to LocalSandbox "
              f"(conda env `{config.conda_env}`)")
    return LocalSandbox(config, workspace)
