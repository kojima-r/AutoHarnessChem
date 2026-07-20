"""Skill Registry / Skill Compiler。

Skill の正本は skills/ ディレクトリ（Agent Skills 標準の SKILL.md）。
SkillCompiler が各SDK向けに配置する:
  - deepagents → build/skills/deepagents/ へコピー（backend の skill directory として mount）
  - claude     → .claude/skills/ へ symlink
  - openai     → build/skills/openai/<name>-<version>.zip の Skill Bundle
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from schemas import TaskSpec

# タスクに関係なく常にロードする横断的 Skill
ALWAYS_ON_SKILLS = ["scientific-verification", "execution-recovery", "result-reporting"]


class Skill(BaseModel):
    name: str
    description: str = ""
    version: str = "0.0.0"
    risk_level: str = "low"
    required_tools: list[str] = Field(default_factory=list)
    allowed_paths: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    body: str = ""
    directory: str = ""


def parse_skill_md(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path}: SKILL.md must start with YAML frontmatter (---)")
    _, frontmatter, body = text.split("---", 2)
    meta = yaml.safe_load(frontmatter) or {}
    return Skill(**meta, body=body.strip(), directory=str(path.parent))


class SkillRegistry:
    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)
        self.skills: dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        self.skills = {}
        if not self.skills_dir.exists():
            return
        for skill_md in sorted(self.skills_dir.glob("*/SKILL.md")):
            skill = parse_skill_md(skill_md)
            self.skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        return self.skills[name]

    def names(self) -> list[str]:
        return sorted(self.skills)

    def select(self, task: TaskSpec) -> list[Skill]:
        """task_type と明示指定から使用 Skill を決める。横断 Skill は常に含める。"""
        selected: dict[str, Skill] = {}
        for name in task.required_skills:
            if name in self.skills:
                selected[name] = self.skills[name]
        for skill in self.skills.values():
            if task.task_type in skill.task_types:
                selected[skill.name] = skill
        for name in ALWAYS_ON_SKILLS:
            if name in self.skills:
                selected[name] = self.skills[name]
        return list(selected.values())

    def write_lockfile(self, path: Path) -> dict[str, dict]:
        """skills.lock — 検証済み Skill バージョンの固定（本番実行用）。"""
        lock: dict[str, dict] = {}
        for name, skill in sorted(self.skills.items()):
            skill_md = Path(skill.directory) / "SKILL.md"
            digest = hashlib.sha256(skill_md.read_bytes()).hexdigest()
            lock[name] = {"version": skill.version, "sha256": digest}
        Path(path).write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
        return lock

    def check_lockfile(self, path: Path) -> list[str]:
        """lock と現状の差分（改変された Skill 名のリスト）を返す。"""
        lock = json.loads(Path(path).read_text(encoding="utf-8"))
        current = {
            name: hashlib.sha256((Path(s.directory) / "SKILL.md").read_bytes()).hexdigest()
            for name, s in self.skills.items()
        }
        drift = [n for n, entry in lock.items() if current.get(n) != entry["sha256"]]
        drift += [n for n in current if n not in lock]
        return sorted(set(drift))


class SkillCompiler:
    def __init__(self, registry: SkillRegistry, repo_root: Path):
        self.registry = registry
        self.repo_root = Path(repo_root)

    def compile(self, provider: str) -> list[Path]:
        if provider == "claude":
            return self._compile_claude()
        if provider == "openai":
            return self._compile_openai()
        if provider == "deepagents":
            return self._compile_deepagents()
        raise ValueError(f"unknown provider: {provider}")

    def compile_all(self) -> dict[str, list[Path]]:
        return {p: self.compile(p) for p in ("deepagents", "claude", "openai")}

    def _compile_claude(self) -> list[Path]:
        dest_root = self.repo_root / ".claude" / "skills"
        dest_root.mkdir(parents=True, exist_ok=True)
        out = []
        for skill in self.registry.skills.values():
            dest = dest_root / skill.name
            if dest.is_symlink() or dest.exists():
                if dest.is_symlink():
                    dest.unlink()
                else:
                    shutil.rmtree(dest)
            dest.symlink_to(Path(skill.directory).resolve(), target_is_directory=True)
            out.append(dest)
        return out

    def _compile_deepagents(self) -> list[Path]:
        dest_root = self.repo_root / "build" / "skills" / "deepagents"
        dest_root.mkdir(parents=True, exist_ok=True)
        out = []
        for skill in self.registry.skills.values():
            dest = dest_root / skill.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(skill.directory, dest)
            out.append(dest)
        return out

    def _compile_openai(self) -> list[Path]:
        dest_root = self.repo_root / "build" / "skills" / "openai"
        dest_root.mkdir(parents=True, exist_ok=True)
        out = []
        for skill in self.registry.skills.values():
            bundle = dest_root / f"{skill.name}-{skill.version}.zip"
            with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in sorted(Path(skill.directory).rglob("*")):
                    if file.is_file():
                        zf.write(file, file.relative_to(skill.directory))
            out.append(bundle)
        return out
