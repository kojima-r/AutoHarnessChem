"""plan_retrosynthesis（AiZynthFinder）の単体テスト（モデル・aizynthfinder 不要）。"""
import json
from pathlib import Path

from schemas import SandboxConfig, TaskSpec
from tools import aizynth
from tools.sandbox import SandboxResult

CONFIG_YAML = """expansion:
  uspto:
    - /models/uspto_model.onnx
    - /models/uspto_templates.csv.gz
stock:
  zinc: /models/zinc_stock.hdf5
"""


class StubSandbox:
    def __init__(self, workspace, stdout="", stderr="", returncode=0, timed_out=False,
                 payload=None, files=(), sandbox_type="local"):
        self.config = SandboxConfig(type=sandbox_type, conda_env="aizynth",
                                    timeout_sec=600, memory_limit_mb=4096)
        self.workspace = Path(workspace)
        self._result = SandboxResult(stdout, stderr, returncode, timed_out, "")
        self._payload = payload
        self._files = files
        self.ran = []

    def run(self, code, script_name="script.py"):
        self.ran.append((script_name, code))
        if self._payload is not None:
            (self.workspace / aizynth.SCRIPT.output_json).write_text(
                json.dumps(self._payload), encoding="utf-8")
        for name in self._files:
            (self.workspace / name).write_text("x", encoding="utf-8")
        return self._result


def _config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(CONFIG_YAML, encoding="utf-8")
    return path


def _payload(solved=True, n_routes=2) -> dict:
    routes = [{
        "rank": rank, "score": 0.99, "n_steps": 1, "n_precursors": 2,
        "solved": solved, "precursors": ["CC(=O)OC(C)=O", "Nc1ccc(O)cc1"],
        "precursors_in_stock": [solved, solved],
        "tree": {"type": "mol", "smiles": "CC(=O)Nc1ccc(O)cc1"},
    } for rank in range(1, n_routes + 1)]
    return {
        "targets": [{"target": "CC(=O)Nc1ccc(O)cc1", "solved": solved,
                     "search_time_s": 2.8, "n_routes": len(routes), "routes": routes,
                     "statistics": {"is_solved": solved}}],
        "selection": {"stock": ["zinc"], "expansion_policy": ["uspto"],
                      "filter_policy": []},
        "search": {"algorithm": "mcts"},
        "n_solved": 1 if solved else 0,
    }


# --- スクリプトの静的な健全性 ---------------------------------------------

def test_script_is_valid_python():
    rendered = aizynth.SCRIPT.render()
    compile(rendered, "<aizynth>", "exec")
    assert "AiZynthFinder" in rendered and "__INPUT_JSON__" not in rendered


# --- config.yml の解決 -----------------------------------------------------

def test_resolve_config_prefers_explicit_then_env(tmp_path, monkeypatch):
    explicit = _config_file(tmp_path)
    env_config = tmp_path / "env" / "config.yml"
    env_config.parent.mkdir()
    env_config.write_text(CONFIG_YAML, encoding="utf-8")

    monkeypatch.setenv("AIZYNTH_CONFIG", str(env_config))
    assert aizynth.resolve_config(tmp_path, str(explicit)) == explicit.resolve()
    assert aizynth.resolve_config(tmp_path) == env_config.resolve()

    monkeypatch.delenv("AIZYNTH_CONFIG")
    monkeypatch.setenv("AIZYNTH_DATA", str(env_config.parent))
    assert aizynth.resolve_config(tmp_path) == env_config.resolve()


def test_blocked_when_model_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("AIZYNTH_CONFIG", raising=False)
    monkeypatch.delenv("AIZYNTH_DATA", raising=False)
    monkeypatch.setattr(aizynth, "_CONFIG_FALLBACK_DIRS", (tmp_path / "nope",))

    sandbox = StubSandbox(tmp_path)
    result = aizynth.plan_retrosynthesis(tmp_path, ["CCO"], sandbox=sandbox)
    assert result.status == "blocked" and result.error_type == "model_unavailable"
    assert "download_public_data" in result.summary
    assert sandbox.ran == []          # 探索を始めない


def test_docker_defers_config_resolution_to_image(tmp_path, monkeypatch):
    """docker sandbox では image 内の AIZYNTH_CONFIG を使うのでブロックしない。"""
    monkeypatch.delenv("AIZYNTH_CONFIG", raising=False)
    monkeypatch.delenv("AIZYNTH_DATA", raising=False)
    monkeypatch.setattr(aizynth, "_CONFIG_FALLBACK_DIRS", (tmp_path / "nope",))

    sandbox = StubSandbox(tmp_path, payload=_payload(), sandbox_type="docker",
                          files=("retrosynthesis_routes.json",
                                 "retrosynthesis_routes.csv"))
    result = aizynth.plan_retrosynthesis(tmp_path, ["CCO"], sandbox=sandbox)
    assert result.status == "success"
    spec = json.loads((tmp_path / aizynth.SCRIPT.input_json).read_text())
    assert spec["config_yaml"] is None


# --- 探索結果の整形 --------------------------------------------------------

def test_success_summary_and_artifacts(tmp_path):
    sandbox = StubSandbox(tmp_path, payload=_payload(),
                          files=("retrosynthesis_routes.json",
                                 "retrosynthesis_routes.csv"))
    result = aizynth.plan_retrosynthesis(
        tmp_path, ["CC(=O)Nc1ccc(O)cc1"], config_yaml=str(_config_file(tmp_path)),
        iteration_limit=50, time_limit_sec=60, n_routes=2, sandbox=sandbox)

    assert result.status == "success" and result.error_type is None
    assert result.data["n_solved"] == 1
    assert {Path(a.path).name for a in result.artifacts} == {
        "retrosynthesis_routes.json", "retrosynthesis_routes.csv"}
    # 巨大な経路木は ToolResult に載せず、要約だけ返す
    digest = result.data["targets"][0]
    assert "tree" not in digest["top_route"]
    assert digest["top_route"]["n_steps"] == 1
    # 探索設定がスクリプトへ渡っている
    spec = json.loads((tmp_path / aizynth.SCRIPT.input_json).read_text())
    assert spec["search"] == {"algorithm": "mcts", "iteration_limit": 50,
                              "time_limit": 60, "max_transforms": 6}


def test_unsolved_is_partial_with_error_type(tmp_path):
    sandbox = StubSandbox(tmp_path, payload=_payload(solved=False),
                          files=("retrosynthesis_routes.json",))
    result = aizynth.plan_retrosynthesis(tmp_path, ["CCO"],
                                         config_yaml=str(_config_file(tmp_path)),
                                         sandbox=sandbox)
    assert result.status == "partial" and result.error_type == "no_route_found"
    assert result.retryable


def test_no_routes_at_all_fails(tmp_path):
    payload = _payload(solved=False, n_routes=0)
    payload["targets"][0]["routes"] = []
    payload["targets"][0]["n_routes"] = 0
    result = aizynth.plan_retrosynthesis(
        tmp_path, ["CCO"], config_yaml=str(_config_file(tmp_path)),
        sandbox=StubSandbox(tmp_path, payload=payload))
    assert result.status == "failed" and result.error_type == "no_route_found"


def test_invalid_inputs(tmp_path):
    sandbox = StubSandbox(tmp_path)
    config = str(_config_file(tmp_path))
    empty = aizynth.plan_retrosynthesis(tmp_path, [], config_yaml=config, sandbox=sandbox)
    assert empty.error_type == "invalid_input"
    bad_algorithm = aizynth.plan_retrosynthesis(tmp_path, ["CCO"], algorithm="dijkstra",
                                                config_yaml=config, sandbox=sandbox)
    assert bad_algorithm.error_type == "invalid_input"
    assert sandbox.ran == []


def test_single_target_string_is_accepted(tmp_path):
    sandbox = StubSandbox(tmp_path, payload=_payload())
    result = aizynth.plan_retrosynthesis(tmp_path, "CC(=O)Nc1ccc(O)cc1",
                                         config_yaml=str(_config_file(tmp_path)),
                                         sandbox=sandbox)
    assert result.status == "success"


def test_failure_classification(tmp_path):
    config = str(_config_file(tmp_path))
    missing_model = aizynth.plan_retrosynthesis(
        tmp_path, ["CCO"], config_yaml=config,
        sandbox=StubSandbox(tmp_path, returncode=1,
                            stderr="FileNotFoundError: [Errno 2] No such file or "
                                   "directory: '/models/uspto_model.onnx'"))
    assert missing_model.error_type == "model_unavailable"
    assert "download_public_data" in missing_model.summary

    unknown_key = aizynth.plan_retrosynthesis(
        tmp_path, ["CCO"], config_yaml=config, stock=["emolecules"],
        sandbox=StubSandbox(tmp_path, returncode=1,
                            stderr="ValueError: The key 'emolecules' is not in the "
                                   "collection"))
    assert unknown_key.error_type == "invalid_input"

    timeout = aizynth.plan_retrosynthesis(
        tmp_path, ["CCO"], config_yaml=config,
        sandbox=StubSandbox(tmp_path, timed_out=True, returncode=124))
    assert timeout.error_type == "timeout" and timeout.retryable


def test_memory_limit_is_raised_for_stock_database(tmp_path):
    """zinc_stock.hdf5 (~650MB) を載せるため既定より大きいメモリ上限を使う。"""
    captured = {}
    original = aizynth.run_env_script

    def spy(sandbox, workspace, script, spec, **kwargs):
        captured.update(kwargs)
        return original(sandbox, workspace, script, spec, **kwargs)

    aizynth.run_env_script = spy
    try:
        aizynth.plan_retrosynthesis(tmp_path, ["CCO"],
                                    config_yaml=str(_config_file(tmp_path)),
                                    sandbox=StubSandbox(tmp_path, payload=_payload()))
    finally:
        aizynth.run_env_script = original
    assert captured["memory_limit_mb"] >= 8192


# --- Verifier / Task Interpreter との結合 ---------------------------------

def test_verifier_checks_routes(tmp_path):
    from harness.verifier import ScientificVerifier

    (tmp_path / "retrosynthesis_routes.json").write_text(
        json.dumps(_payload(solved=False)), encoding="utf-8")
    (tmp_path / "retrosynthesis_routes.csv").write_text(
        "target_smiles,route_rank,solved\nCCO,1,False\n", encoding="utf-8")
    task = TaskSpec(description="逆合成経路", task_type="retrosynthesis_planning",
                    expected_outputs=["retrosynthesis_routes.json",
                                      "retrosynthesis_routes.csv"])
    verification = ScientificVerifier().verify(task, tmp_path)
    assert verification.passed          # 未解決は警告（不合格ではない）
    assert any("未解決" in w for w in verification.scientific_warnings)


def test_verifier_flags_inconsistent_solved_flag(tmp_path):
    from harness.verifier import ScientificVerifier

    payload = _payload(solved=True)
    payload["targets"][0]["routes"][0]["precursors_in_stock"] = [True, False]
    (tmp_path / "retrosynthesis_routes.json").write_text(json.dumps(payload),
                                                         encoding="utf-8")
    task = TaskSpec(description="逆合成経路", task_type="retrosynthesis_planning")
    verification = ScientificVerifier().verify(task, tmp_path)
    assert not verification.passed
    assert any("solved" in r for r in verification.required_repairs)


def test_route_planning_requests_route_to_aizynth_task_type():
    from harness.controller import interpret_task

    planning = interpret_task("パラセタモールの合成経路を提案してください")
    assert planning.task_type == "retrosynthesis_planning"
    assert "retrosynthesis_routes.json" in planning.expected_outputs

    # 1段階の逆合成予測は従来どおり ReactionT5 側のタスク種別
    single_step = interpret_task("この分子の逆合成（前駆体）を予測してください")
    assert single_step.task_type == "reaction_prediction"
