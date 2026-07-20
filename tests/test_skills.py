from pathlib import Path

from harness.skill_registry import ALWAYS_ON_SKILLS, SkillRegistry
from schemas import TaskSpec

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def test_all_skills_parse():
    registry = SkillRegistry(SKILLS_DIR)
    assert len(registry.names()) >= 6
    for name in registry.names():
        skill = registry.get(name)
        assert skill.description
        assert skill.version
        assert "# Procedure" in skill.body or "# Recovery procedure" in skill.body


def test_selection_includes_always_on_and_task_type():
    registry = SkillRegistry(SKILLS_DIR)
    task = TaskSpec(description="HOMO/LUMO", task_type="orbital_calculation")
    names = {s.name for s in registry.select(task)}
    assert "pyscf-orbitals" in names
    assert set(ALWAYS_ON_SKILLS) <= names


def test_lockfile_roundtrip(tmp_path):
    registry = SkillRegistry(SKILLS_DIR)
    lock_path = tmp_path / "skills.lock"
    registry.write_lockfile(lock_path)
    assert registry.check_lockfile(lock_path) == []


def test_compile_openai_bundles(tmp_path):
    from harness.skill_registry import SkillCompiler

    registry = SkillRegistry(SKILLS_DIR)
    compiler = SkillCompiler(registry, tmp_path)
    bundles = compiler.compile("openai")
    assert len(bundles) == len(registry.names())
    assert all(b.suffix == ".zip" and b.exists() for b in bundles)
