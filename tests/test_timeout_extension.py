"""実時間上限に達したときの「さらに待つか確認して延長」の動作。

数日かかる計算があるため、上限に達しても即打ち切らず、
  - ユーザに確認して延長できる（CLI の対話 / Web UI のボタン）
  - 延長中もエージェントは動き続ける（計算をやり直さない）
  - 無人実行では config の方針（extend / stop）に従う
ことを検証する。
"""
import asyncio
import json
from pathlib import Path

import pytest

from app.cli import _parse_duration
from harness import controller as controller_mod
from harness.config import load_config


class SlowAdapter:
    """指定回数だけ「まだ終わらない」状態を続けてから完了する Adapter。"""
    name = "dummy"

    def __init__(self, tracer, policy, sleep_sec=0.25, cancelled=None):
        self.tracer = tracer
        self.sleep_sec = sleep_sec
        self.runs = 0
        self.cancelled = cancelled if cancelled is not None else []

    async def create(self, config, skills, tools):
        self.skills = skills

    async def run(self, task, state):
        self.runs += 1
        try:
            await asyncio.sleep(self.sleep_sec)
        except asyncio.CancelledError:
            self.cancelled.append(state.attempts)
            raise
        Path(state.workspace, "orbital_features.csv").write_text(
            "smiles,homo_ev,lumo_ev\nO,-13.2,4.1\n")
        return "done"

    async def shutdown(self):
        pass


def _config(tmp_path, **runtime):
    config = load_config()
    config.paths.workspaces = tmp_path / "workspaces"
    config.paths.traces = tmp_path / "traces"
    config.runtime.sandbox.type = "local"
    config.runtime.attempt_timeout_sec = 1        # テスト用に極端に短くする
    config.runtime.timeout_extension_sec = 1
    for key, value in runtime.items():
        setattr(config.runtime, key, value)
    return config


def _run(config, monkeypatch, adapter_factory, **kwargs):
    events = []
    monkeypatch.setattr(controller_mod, "create_adapter",
                        lambda provider, tracer, policy: adapter_factory(tracer, policy))
    controller = controller_mod.HarnessController(config)
    report = asyncio.run(controller.run("水のHOMO/LUMOを計算して",
                                        on_event=events.append, **kwargs))
    return report, events


def _phases(events):
    return [e.payload.get("phase") for e in events if e.payload.get("phase")]


# --- ユーザに確認して延長する -----------------------------------------------

def test_asks_user_and_keeps_waiting(monkeypatch, tmp_path):
    """上限に達したら確認し、延長すると同じ試行を続ける（やり直さない）。"""
    asked = []
    adapters = []

    def factory(tracer, policy):
        adapter = SlowAdapter(tracer, policy, sleep_sec=2.5)
        adapters.append(adapter)
        return adapter

    def on_timeout(info):
        asked.append(info)
        return True                     # timeout_extension_sec だけ延長

    report, events = _run(_config(tmp_path), monkeypatch, factory,
                          on_timeout=on_timeout)

    assert report.passed and report.attempts == 1        # 1 試行で完了
    assert adapters[0].runs == 1                        # 再実行していない
    assert adapters[0].cancelled == []                  # 打ち切っていない
    assert len(asked) >= 2                              # 上限ごとに確認している
    assert asked[0]["waited_sec"] == 1 and asked[1]["waited_sec"] == 2
    assert asked[1]["extensions"] == 1
    assert report.timeout_extensions == len(asked)
    assert report.extended_sec == len(asked)
    phases = _phases(events)
    assert "attempt_timeout_pending" in phases and "attempt_timeout_extended" in phases
    assert "attempt_timeout" not in phases              # 打ち切りイベントは出ない


def test_user_can_specify_seconds(monkeypatch, tmp_path):
    """秒数を返すとその分だけ待つ。"""
    report, events = _run(_config(tmp_path), monkeypatch,
                          lambda t, p: SlowAdapter(t, p, sleep_sec=1.5),
                          on_timeout=lambda info: 5)
    assert report.passed and report.timeout_extensions == 1
    assert report.extended_sec == 5
    extended = next(e for e in events
                    if e.payload.get("phase") == "attempt_timeout_extended")
    assert extended.payload["extend_sec"] == 5


def test_async_asker_is_supported(monkeypatch, tmp_path):
    """Web UI 側は非同期に応答するため、awaitable も受け付ける。"""
    async def on_timeout(info):
        await asyncio.sleep(0)
        return 3

    report, _ = _run(_config(tmp_path), monkeypatch,
                     lambda t, p: SlowAdapter(t, p, sleep_sec=1.5),
                     on_timeout=on_timeout)
    assert report.passed and report.extended_sec == 3


def test_declining_stops_the_attempt(monkeypatch, tmp_path):
    """『待たない』と答えたら打ち切り、その時点の成果物で検証・報告する。"""
    cancelled = []
    report, events = _run(
        _config(tmp_path, max_replans=0), monkeypatch,
        lambda t, p: SlowAdapter(t, p, sleep_sec=30, cancelled=cancelled),
        on_timeout=lambda info: False)

    assert not report.passed                    # 成果物が作られる前に打ち切られた
    assert report.timeout_extensions == 0
    assert cancelled == [1]                     # 実行中のタスクは cancel される
    assert "attempt_timeout" in _phases(events)
    workspace = Path(report.artifacts[0].path).parent if report.artifacts else \
        Path(load_config().paths.workspaces)
    assert (tmp_path / "workspaces" / report.run_id / "report.json").exists()


# --- 無人実行のときの方針 ---------------------------------------------------

def test_policy_extend_waits_without_asking(monkeypatch, tmp_path):
    """on_attempt_timeout=extend なら確認せず延長する（無人の長時間実行）。"""
    report, events = _run(_config(tmp_path, on_attempt_timeout="extend"), monkeypatch,
                          lambda t, p: SlowAdapter(t, p, sleep_sec=2.5))
    assert report.passed and report.attempts == 1
    assert report.timeout_extensions >= 2
    assert "attempt_timeout_pending" not in _phases(events)   # 確認していない


def test_policy_stop_cuts_immediately(monkeypatch, tmp_path):
    report, events = _run(_config(tmp_path, on_attempt_timeout="stop", max_replans=0),
                          monkeypatch, lambda t, p: SlowAdapter(t, p, sleep_sec=30))
    assert not report.passed
    assert "attempt_timeout" in _phases(events)


def test_ask_without_asker_stops_and_warns(monkeypatch, tmp_path):
    """ask なのに確認手段が無い場合は、無限に待たず打ち切って理由を残す。"""
    report, events = _run(_config(tmp_path, max_replans=0), monkeypatch,
                          lambda t, p: SlowAdapter(t, p, sleep_sec=30))
    assert not report.passed
    phases = _phases(events)
    assert "attempt_timeout_unanswered" in phases and "attempt_timeout" in phases


def test_max_extensions_is_respected(monkeypatch, tmp_path):
    """延長回数の上限を超えたら確認せず打ち切る。"""
    asked = []
    report, events = _run(
        _config(tmp_path, max_timeout_extensions=2, max_replans=0), monkeypatch,
        lambda t, p: SlowAdapter(t, p, sleep_sec=30),
        on_timeout=lambda info: asked.append(info) or True)
    assert report.timeout_extensions == 2
    assert len(asked) == 2                       # 3 回目は確認せず打ち切り
    assert "attempt_timeout_limit" in _phases(events)


# --- CLI / API のインターフェース -------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("", 3600), ("7200", 7200), ("30m", 1800), ("12h", 43200), ("2d", 172800),
    ("1.5h", 5400), ("なんとなく", 3600),
])
def test_cli_duration_parsing(text, expected):
    assert _parse_duration(text, 3600) == expected


def test_cli_exposes_timeout_options():
    from app.cli import main

    with pytest.raises(SystemExit):
        main(["run", "x", "--on-timeout", "unknown"])       # choices で弾かれる


def test_api_timeout_endpoint(tmp_path):
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from app.api import create_app

    config = load_config()
    config.paths.workspaces = tmp_path / "workspaces"
    config.paths.traces = tmp_path / "traces"
    config.paths.workspaces.mkdir()
    config.paths.traces.mkdir()
    client = TestClient(create_app(config))

    # 判断待ちでない run への決定は 404
    assert client.post("/api/runs/run-nope/timeout", json={"stop": True}).status_code == 404


def test_api_pending_then_extend_completes_the_run(tmp_path, monkeypatch):
    """Web UI 経路: 上限で判断待ちになり、延長 API を叩くと同じ試行が完走する。"""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import time

    from fastapi.testclient import TestClient

    from app.api import create_app

    monkeypatch.setattr(controller_mod, "create_adapter",
                        lambda provider, tracer, policy: SlowAdapter(tracer, policy,
                                                                     sleep_sec=3))
    config = load_config()
    config.paths.workspaces = tmp_path / "workspaces"
    config.paths.traces = tmp_path / "traces"
    config.paths.workspaces.mkdir()
    config.paths.traces.mkdir()
    config.runtime.attempt_timeout_sec = 1
    config.runtime.timeout_extension_sec = 10

    with TestClient(create_app(config)) as client:
        run_id = client.post("/api/tasks", json={"request": "水のHOMO/LUMOを計算して"}
                             ).json()["run_id"]
        # 上限に達して判断待ちになるのを待つ
        for _ in range(80):
            detail = client.get(f"/api/runs/{run_id}").json()
            if detail.get("status") == "awaiting_decision":
                break
            time.sleep(0.25)
        assert detail["status"] == "awaiting_decision"
        assert detail["timeout_info"]["waited_sec"] == 1

        answer = client.post(f"/api/runs/{run_id}/timeout", json={"extend_sec": 20})
        assert answer.status_code == 200 and answer.json()["decision"] == "extend"

        for _ in range(80):
            detail = client.get(f"/api/runs/{run_id}").json()
            if detail.get("status") in ("succeeded", "failed", "error"):
                break
            time.sleep(0.25)
        assert detail["status"] == "succeeded"       # 延長して完走した
        assert detail["report"]["timeout_extensions"] == 1
        assert detail["report"]["extended_sec"] == 20


def test_web_ui_offers_extension_buttons():
    index = (Path(__file__).resolve().parent.parent / "app" / "web"
             / "index.html").read_text(encoding="utf-8")
    assert "awaiting_decision" in index
    assert "decideTimeout" in index and "/timeout" in index
    for label in ("+1 時間", "+1 日", "打ち切って報告する"):
        assert label in index


def test_report_and_ledger_record_extensions(monkeypatch, tmp_path):
    report, _ = _run(_config(tmp_path), monkeypatch,
                     lambda t, p: SlowAdapter(t, p, sleep_sec=1.5),
                     on_timeout=lambda info: 2)
    saved = json.loads((tmp_path / "workspaces" / report.run_id / "report.json")
                       .read_text(encoding="utf-8"))
    assert saved["timeout_extensions"] == 1 and saved["extended_sec"] == 2


# --- 長い計算中でも上限のチェックが効くこと --------------------------------

def test_tool_calls_do_not_block_the_event_loop(tmp_path):
    """ツール実行は別スレッドで走るので、計算中でも上限（延長の確認）が時間通り効く。

    同期のツールをそのまま await すると、controller の asyncio.wait が計算終了まで
    起きられず、上限が実質無効になる（実測で 60s 上限が 141s まで遅れた）。
    """
    import time

    from harness.policy import PolicyGate
    from harness.traces import TraceWriter
    from schemas import ToolResult
    from tools.registry import ToolRegistry, ToolSpec

    from adapters.base import BaseAdapter

    class Dummy(BaseAdapter):
        name = "dummy"

        async def _create_impl(self):
            pass

        async def run(self, task, state):
            return ""

    registry = ToolRegistry()
    registry.register(ToolSpec(name="slow", description="", parameters={},
                               func=lambda: (time.sleep(0.6),
                                             ToolResult(status="success",
                                                        summary="done"))[1]))
    adapter = Dummy(TraceWriter(tmp_path, "run-loop"), PolicyGate())
    adapter.tools = registry

    async def scenario():
        ticks = 0
        tool = asyncio.ensure_future(adapter.call_tool_async("slow", {}))
        while not tool.done():                 # ツール実行中もループが回る
            await asyncio.sleep(0.05)
            ticks += 1
        return ticks, tool.result()

    ticks, result = asyncio.run(scenario())
    assert result.status == "success"
    assert ticks >= 5, "ツール実行中にイベントループが止まっている"
