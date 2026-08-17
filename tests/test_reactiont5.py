"""predict_reaction_t5 の単体テスト（torch 不要 — sandbox はスタブ）。"""
import json
from pathlib import Path

from schemas import SandboxConfig
from tools.reactiont5 import (DEFAULT_MEMORY_LIMIT_MB, INPUT_JSON, OUTPUT_JSON,
                              PARTIAL_JSON, SCRIPT, SCRIPT_NAME, predict_reaction_t5)
from tools.sandbox import SandboxResult


class StubSandbox:
    """run() で受け取ったスクリプトを記録し、あらかじめ決めた結果を返す sandbox。

    envrun.run_env_script は実行前に古い出力を消すので、payload は run() の中で書く。
    """

    def __init__(self, workspace, stdout="", stderr="", returncode=0,
                 timed_out=False, output_payload=None, partial_payload=None,
                 config=None):
        self.config = config or SandboxConfig(type="local", conda_env="reactiont5")
        self.workspace = Path(workspace)
        self._result = SandboxResult(stdout, stderr, returncode, timed_out, "")
        self._output = output_payload
        self._partial = partial_payload
        # copy.copy された複製（with_limits が上限を差し替えたもの）でも同じ list を
        # 共有するので、実際に適用された上限をここで観測できる
        self.ran_scripts = []
        self.seen_limits = []

    def run(self, code, script_name="script.py"):
        self.ran_scripts.append((script_name, code))
        self.seen_limits.append(self.config.model_dump())
        if self._output is not None:
            (self.workspace / OUTPUT_JSON).write_text(json.dumps(self._output),
                                                      encoding="utf-8")
        if self._partial is not None:
            (self.workspace / PARTIAL_JSON).write_text(json.dumps(self._partial),
                                                       encoding="utf-8")
        return self._result


def _spec(sandbox) -> dict:
    """スクリプトへ渡された入力 JSON を読む。"""
    return json.loads((sandbox.workspace / INPUT_JSON).read_text(encoding="utf-8"))


def test_script_is_valid_python():
    script = SCRIPT.render()
    compile(script, "<reactiont5>", "exec")
    assert "ReactionT5Yield2" in script and "AutoModelForSeq2SeqLM" in script
    # プレースホルダは実ファイル名に置換されている
    assert "__INPUT_JSON__" not in script and "__PARTIAL_JSON__" not in script
    # 1 反応ごとに途中結果を書く（打ち切られても完了分を回収できる）
    assert script.count("dump_partial()") >= 3


def test_invalid_inputs_fail_without_running_sandbox(tmp_path):
    sandbox = StubSandbox(tmp_path)
    bad_task = predict_reaction_t5(tmp_path, ["x"], task="oracle", sandbox=sandbox)
    assert bad_task.status == "failed" and bad_task.error_type == "invalid_input"
    empty = predict_reaction_t5(tmp_path, [], task="forward", sandbox=sandbox)
    assert empty.status == "failed"
    assert sandbox.ran_scripts == []


def test_success_writes_csv(tmp_path):
    payload = {"task": "forward", "model": "sagawa/ReactionT5v2-forward",
               "results": [{"input": "REACTANT:CCOREAGENT:",
                            "prediction": "CCO", "candidates": ["CCO", "CC"]}]}
    sandbox = StubSandbox(tmp_path, stdout="predicted 1 entries", output_payload=payload)
    result = predict_reaction_t5(tmp_path, ["REACTANT:CCOREAGENT:"],
                                 task="forward", sandbox=sandbox)
    assert result.status == "success"
    assert sandbox.ran_scripts[0][0] == SCRIPT_NAME
    csv_text = (tmp_path / "reactiont5_predictions.csv").read_text()
    assert "CCO|CC" in csv_text
    assert result.artifacts and result.artifacts[0].path.endswith(".csv")
    assert result.data["pending"] == []
    # 入力仕様が専用環境スクリプトへ渡っている
    spec = _spec(sandbox)
    assert spec["task"] == "forward" and spec["num_beams"] == 1
    assert spec["model_name"] == "sagawa/ReactionT5v2-forward"


def test_memory_and_timeout_limits_are_applied(tmp_path):
    """既定のメモリ上限を上げ、指定した timeout が sandbox へ反映されること。"""
    payload = {"task": "yield", "model": "sagawa/ReactionT5v2-yield",
               "results": [{"input": "r1", "predicted_yield": 55.2}]}
    sandbox = StubSandbox(tmp_path, output_payload=payload,
                          config=SandboxConfig(type="local", conda_env="reactiont5",
                                               timeout_sec=600, cpu_limit_sec=300,
                                               memory_limit_mb=1024))
    result = predict_reaction_t5(tmp_path, ["r1"], task="yield", timeout_sec=1800,
                                 sandbox=sandbox)

    assert result.status == "success"
    seen = sandbox.seen_limits[0]
    assert seen["memory_limit_mb"] == DEFAULT_MEMORY_LIMIT_MB >= 16384
    assert seen["timeout_sec"] == 1800
    # CPU 時間上限は実時間より先に効いてはいけない（torch は全コアを使う）
    assert seen["cpu_limit_sec"] > 1800


def test_partial_result_is_returned_when_interrupted(tmp_path):
    """打ち切られても、完了分は partial + pending として返る。"""
    partial = {"task": "yield", "model": "sagawa/ReactionT5v2-yield", "partial": True,
               "results": [{"input": "r1", "predicted_yield": 42.0}]}
    sandbox = StubSandbox(tmp_path, timed_out=True, returncode=124,
                          partial_payload=partial)
    result = predict_reaction_t5(tmp_path, ["r1", "r2"], task="yield", sandbox=sandbox)

    assert result.status == "partial" and result.retryable
    assert result.error_type == "timeout"
    assert result.data["pending"] == ["r2"]
    assert "42.0" in (tmp_path / "reactiont5_predictions.csv").read_text()


def test_failure_classification(tmp_path):
    missing_env = predict_reaction_t5(
        tmp_path, ["x"], task="yield",
        sandbox=StubSandbox(tmp_path, stderr="EnvironmentLocationNotFound: reactiont5",
                            returncode=1))
    assert missing_env.error_type == "missing_environment"
    assert not missing_env.retryable

    missing_dep = predict_reaction_t5(
        tmp_path, ["x"], task="yield",
        sandbox=StubSandbox(tmp_path, stderr="ModuleNotFoundError: No module named 'torch'",
                            returncode=1))
    assert missing_dep.error_type == "missing_dependency"

    # run-5956e69a1dd4 で実際に起きた失敗（メモリ上限不足）が OOM として分類される
    oom = predict_reaction_t5(
        tmp_path, ["x"], task="yield",
        sandbox=StubSandbox(
            tmp_path, returncode=1,
            stderr="OpenBLAS error: Memory allocation still failed after 10 retries, "
                   "giving up."))
    assert oom.error_type == "out_of_memory" and oom.retryable
    assert "memory_limit_mb" in oom.summary

    timeout = predict_reaction_t5(
        tmp_path, ["x"], task="yield",
        sandbox=StubSandbox(tmp_path, timed_out=True, returncode=124))
    assert timeout.error_type == "timeout" and timeout.retryable


def test_registered_with_dedicated_env(tmp_path, monkeypatch):
    from harness.policy import PolicyGate
    from tools.registry import build_default_registry

    created_envs = []
    import tools.registry as registry_mod
    original = registry_mod.create_sandbox

    def spy(config, workspace, env=None):
        created_envs.append(env)
        return original(config, workspace, env=env)

    monkeypatch.setattr(registry_mod, "create_sandbox", spy)
    registry = build_default_registry(tmp_path, SandboxConfig(type="local"), PolicyGate())
    assert "predict_reaction_t5" in registry.names()
    assert "reactiont5" in created_envs  # pyscf 既定環境とは別の専用環境
    # メモリ上限と実行上限をエージェントが指定できる（run-5956e69a1dd4 の失敗要因）
    properties = registry.get("predict_reaction_t5").parameters["properties"]
    assert properties["memory_limit_mb"]["default"] == DEFAULT_MEMORY_LIMIT_MB
    assert "timeout_sec" in properties


def test_verifier_checks_yield_range(tmp_path):
    from harness.verifier import ScientificVerifier
    from schemas import TaskSpec

    (tmp_path / "reactiont5_predictions.csv").write_text(
        "input,predicted_yield\nREACTANT:xREAGENT:,150.0\n")
    task = TaskSpec(description="収率予測", task_type="reaction_prediction",
                    expected_outputs=["reactiont5_predictions.csv"])
    verification = ScientificVerifier().verify(task, tmp_path)
    assert any("0–100%" in w for w in verification.scientific_warnings)
