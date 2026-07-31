"""ダミーAdapterで controller の自律ループと進行イベントを検証する（SDK不要）。"""
import asyncio
from pathlib import Path

from harness import controller as controller_mod
from harness.config import load_config


class DummyAdapter:
    """1回目は失敗（出力なし）、2回目で期待出力を生成するAdapter。"""
    name = "dummy"

    def __init__(self, tracer, policy):
        self.tracer = tracer
        self.calls = 0

    async def create(self, config, skills, tools):
        self.skills = skills

    async def run(self, task, state):
        self.calls += 1
        self.tracer.emit("tool_call", "dummy",
                         {"tool": "calculate_orbitals", "arguments": {"smiles": ["O"]}})
        if self.calls >= 2:
            Path(state.workspace, "orbital_features.csv").write_text(
                "smiles,homo_ev,lumo_ev\nO,-13.2,4.1\n")
            self.tracer.emit("tool_result", "dummy",
                             {"tool": "calculate_orbitals", "status": "success", "summary": "ok"})
        else:
            self.tracer.emit("tool_result", "dummy",
                             {"tool": "calculate_orbitals", "status": "failed",
                              "summary": "boom", "error_type": "scf_failed"})
        return f"attempt {self.calls} done"

    async def shutdown(self):
        pass


def test_task_interpreter_routes_new_tool_families():
    """OptTDDFT / AiZynthFinder を使うリクエストが専用の task_type になる。"""
    from harness.controller import interpret_task

    cases = {
        "ベンゼンの HOMO/LUMO を計算して": "orbital_calculation",
        "クマリンの吸収スペクトルを TDDFT で計算して": "orbital_calculation",
        "目標波長 500nm に近い分子を Optuna で探索して": "molecular_design",
        "ESIPT の PES スキャンを実行して": "pes_scan",
        "パラセタモールの合成経路を AiZynthFinder で探索して": "retrosynthesis_planning",
        "この反応の収率を予測して": "reaction_prediction",
    }
    for request, expected in cases.items():
        task = interpret_task(request)
        assert task.task_type == expected, f"{request} -> {task.task_type}"
        # 期待出力と達成条件が task_type から補完される
        assert task.expected_outputs
        assert task.success_criteria


def _config(tmp_path):
    config = load_config()
    config.paths.workspaces = tmp_path / "workspaces"
    config.paths.traces = tmp_path / "traces"
    config.runtime.sandbox.type = "local"
    return config


def test_loop_repairs_then_passes_and_streams_events(monkeypatch, tmp_path):
    adapters = []

    def fake_create(provider, tracer, policy):
        adapter = DummyAdapter(tracer, policy)
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(controller_mod, "create_adapter", fake_create)
    events = []
    controller = controller_mod.HarnessController(_config(tmp_path))
    report = asyncio.run(controller.run("水のHOMO/LUMOを計算して",
                                        on_event=events.append))

    assert report.passed and report.attempts == 2
    assert report.task.task_type == "orbital_calculation"
    # 進行フェーズイベントが両インターフェース共通の trace に流れていること
    phases = [e.payload.get("phase") for e in events if e.actor == "controller"]
    assert phases.count("attempt") == 2
    assert "start" in phases and "verifying" in phases and "finished" in phases
    # Adapter のツールイベントもリスナーに届く
    assert any(e.event_type == "tool_call" for e in events)
    # 2回目の試行には Verifier の修復指示が渡っている
    attempt2 = next(e for e in events
                    if e.payload.get("phase") == "attempt" and e.payload.get("attempt") == 2)
    assert attempt2.payload["repairs"]
    # 選択された Skill に pyscf-orbitals が含まれる
    assert any(s.name == "pyscf-orbitals" for s in adapters[0].skills)
