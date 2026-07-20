"""predict_reaction_t5 の単体テスト（torch 不要 — sandbox はスタブ）。"""
from pathlib import Path

from tools.reactiont5 import (INPUT_JSON, OUTPUT_JSON, SCRIPT_NAME, _SCRIPT_TEMPLATE,
                              predict_reaction_t5)
from tools.sandbox import SandboxResult


class StubSandbox:
    def __init__(self, workspace, stdout="", stderr="", returncode=0,
                 timed_out=False, output_payload=None):
        import json
        from schemas import SandboxConfig
        self.config = SandboxConfig(type="local", conda_env="reactiont5")
        self.workspace = Path(workspace)
        self._result = SandboxResult(stdout, stderr, returncode, timed_out, "")
        self.ran_scripts = []
        if output_payload is not None:
            (self.workspace / OUTPUT_JSON).write_text(json.dumps(output_payload))

    def run(self, code, script_name="script.py"):
        self.ran_scripts.append((script_name, code))
        return self._result


def test_script_template_is_valid_python():
    script = (_SCRIPT_TEMPLATE.replace("__INPUT_JSON__", INPUT_JSON)
              .replace("__OUTPUT_JSON__", OUTPUT_JSON))
    compile(script, "<reactiont5>", "exec")
    assert "ReactionT5Yield2" in script and "AutoModelForSeq2SeqLM" in script


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
    # 入力仕様が専用環境スクリプトへ渡っている
    assert (tmp_path / INPUT_JSON).exists()


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

    timeout = predict_reaction_t5(
        tmp_path, ["x"], task="yield",
        sandbox=StubSandbox(tmp_path, timed_out=True, returncode=124))
    assert timeout.error_type == "timeout" and timeout.retryable


def test_registered_with_dedicated_env(tmp_path, monkeypatch):
    from harness.policy import PolicyGate
    from schemas import SandboxConfig
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


def test_verifier_checks_yield_range(tmp_path):
    from harness.verifier import ScientificVerifier
    from schemas import TaskSpec

    (tmp_path / "reactiont5_predictions.csv").write_text(
        "input,predicted_yield\nREACTANT:xREAGENT:,150.0\n")
    task = TaskSpec(description="収率予測", task_type="reaction_prediction",
                    expected_outputs=["reactiont5_predictions.csv"])
    verification = ScientificVerifier().verify(task, tmp_path)
    assert any("0–100%" in w for w in verification.scientific_warnings)
