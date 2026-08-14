"""複雑（複合）クエリが完了しなかった原因に対する回帰テスト。

run-620297cc1ccb（「骨格をベースに紫外域に吸収を持ち、合成可能（経路を出力）な
化合物を探す」）で起きたこと:
  1. docker バイナリはあるが image が無く、専用環境ツールが returncode=125 で
     全滅した（fallback は「docker が無い」ときだけ動いていた）
  2. 失敗が runtime_error に分類され、原因が分からなかった
  3. 複合要求なのに task_type が 1 つしか付かず、逆合成・分子設計の Skill が
     ロードされなかった
  4. 重い TDDFT が途中で打ち切られると結果が丸ごと失われた
  5. 試行に実時間の上限が無く、run が終わらないまま report も作られなかった

再実行（run-9850117a6f29）で見つかった追加の原因:
  6. 実用的な TDDFT（CAM-B3LYP/6-31G(d)）が sandbox のメモリ上限 4096MB では
     SIGSEGV になり、runtime_error として原因不明のまま報告されていた
  7. harness 環境に rdkit が無く standardize_smiles が使えなかった
  8. `name` 引数を持つツール（generate_3d_structure）が ToolRegistry.call() の
     引数名と衝突して呼べなかった
"""
import asyncio
import json
from pathlib import Path

import pytest

from harness import controller as controller_mod
from harness.config import load_config
from harness.controller import detect_task_types, interpret_task
from harness.skill_registry import SkillRegistry
from schemas import SandboxConfig, TaskSpec
from tools import envrun, opttddft, sandbox as sandbox_mod
from tools.sandbox import SandboxResult

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
COMPLEX_REQUEST = ("O=c1ccc2ccccc2o1の骨格をベースに吸収スペクトルにおいて"
                   "紫外線領域に吸収を持ち、合成可能（経路を出力）な化合物を探してください")


# --- 1 & 2. docker が使えないときのフォールバックと失敗分類 -------------------

def test_create_sandbox_falls_back_when_image_missing(tmp_path, monkeypatch, capsys):
    """docker バイナリがあっても image が無ければ LocalSandbox を使う。"""
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(sandbox_mod, "_DOCKER_IMAGE_CACHE", {})
    monkeypatch.setattr(sandbox_mod.subprocess, "run",
                        lambda *a, **k: SandboxResult("", "no such image", 1, False, ""))
    config = SandboxConfig(type="docker")
    box = sandbox_mod.create_sandbox(config, tmp_path, env="opttddft")
    assert isinstance(box, sandbox_mod.LocalSandbox)
    assert box.config.conda_env == "pyscf"          # named_envs の conda 環境で実行
    assert "falling back" in capsys.readouterr().out


def test_create_sandbox_uses_docker_when_image_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(sandbox_mod, "_DOCKER_IMAGE_CACHE", {})
    monkeypatch.setattr(sandbox_mod.subprocess, "run",
                        lambda *a, **k: SandboxResult("[]", "", 0, False, ""))
    box = sandbox_mod.create_sandbox(SandboxConfig(type="docker"), tmp_path)
    assert isinstance(box, sandbox_mod.DockerSandbox)


def test_docker_image_availability_is_cached(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(sandbox_mod, "_DOCKER_IMAGE_CACHE", {})

    def fake_run(*args, **kwargs):
        calls.append(args)
        return SandboxResult("", "", 1, False, "")

    monkeypatch.setattr(sandbox_mod.subprocess, "run", fake_run)
    for _ in range(3):
        sandbox_mod.create_sandbox(SandboxConfig(type="docker"), tmp_path)
    assert len(calls) == 1        # 同じ image は 1 回だけ問い合わせる


@pytest.mark.parametrize("stderr,expected", [
    ("Unable to find image 'autoharnesschem/opttddft:latest' locally",
     "missing_environment"),
    ("docker: Error response from daemon: pull access denied for autoharnesschem/x",
     "missing_environment"),
    ("Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
     "missing_environment"),
])
def test_docker_failures_are_classified_with_hint(stderr, expected):
    error_type, retryable, hint = envrun.classify_failure(stderr, "pyscf")
    assert error_type == expected and not retryable
    assert "docker" in hint and ("build" in hint or "local" in hint)


# --- 3. 複合クエリのタスク解釈と Skill 選択 ---------------------------------

def test_complex_request_detects_all_aspects():
    detected = detect_task_types(COMPLEX_REQUEST)
    assert "retrosynthesis_planning" in detected      # 「合成可能（経路を出力）」
    assert "molecular_design" in detected             # 「化合物を探す」
    assert "orbital_calculation" in detected          # 「吸収スペクトル」

    task = interpret_task(COMPLEX_REQUEST)
    assert task.secondary_task_types                  # primary 以外も保持する
    assert set(task.secondary_task_types) | {task.task_type} == set(detected)
    # 必須要件は primary の成果物 + 報告（他の側面はここでは必須にしない）
    primary_outputs = controller_mod._DEFAULT_EXPECTED_OUTPUTS[task.task_type]
    assert task.expected_outputs == primary_outputs + ["report_user.md",
                                                       "report_user.html"]
    assert any("他の側面" in c for c in task.success_criteria)


def test_composite_task_loads_every_relevant_skill():
    task = interpret_task(COMPLEX_REQUEST)
    names = {s.name for s in SkillRegistry(SKILLS_DIR).select(task)}
    # 逆合成と分子設計の手順書が両方入る（以前は片方だけだった）
    assert {"aizynth-retrosynthesis", "tddft-molecular-design"} <= names
    # 複合タスクの進め方（分解・安い順・部分結果で報告）も常時ロードされる
    assert "complex-task-planning" in names


def test_explicit_task_type_keeps_detected_aspects_as_secondary():
    task = interpret_task(COMPLEX_REQUEST, task_type="pes_scan")
    assert task.task_type == "pes_scan"               # 明示指定が primary になる
    assert "orbital_calculation" in task.secondary_task_types
    assert "retrosynthesis_planning" in task.secondary_task_types
    assert task.expected_outputs[:2] == ["esipt_scan_results.csv",
                                         "esipt_pes_profile.png"]

    # 期待出力を明示した場合は、報告の自動追加もしない（ユーザ指定を尊重する）
    explicit = interpret_task(COMPLEX_REQUEST, expected_outputs=["only_this.csv"])
    assert explicit.expected_outputs == ["only_this.csv"]


def test_verifier_checks_secondary_aspects(tmp_path):
    """secondary の側面のファイルもドメイン検査の対象になる。"""
    from harness.verifier import ScientificVerifier

    (tmp_path / "orbital_features.csv").write_text(
        "smiles,homo_ev,lumo_ev\nO,4.1,-13.2\n", encoding="utf-8")   # HOMO > LUMO
    task = TaskSpec(description="x", task_type="retrosynthesis_planning",
                    secondary_task_types=["orbital_calculation"])
    verification = ScientificVerifier().verify(task, tmp_path)
    assert not verification.passed
    assert any("HOMO" in r for r in verification.required_repairs)


# --- 4. 打ち切られても部分結果を返す ---------------------------------------

class TimingOutSandbox:
    """タイムアウトする直前に途中結果（1 分子分）を書き出す sandbox。"""

    def __init__(self, workspace, done_smiles, timed_out=True, returncode=124):
        self.config = SandboxConfig(type="local", conda_env="pyscf", timeout_sec=600)
        self.workspace = Path(workspace)
        self.done_smiles = done_smiles
        self.timed_out = timed_out
        self.returncode = returncode

    def run(self, code, script_name="script.py"):
        name = script_name.replace(".py", "").replace("_opttddft_", "")
        partial = self.workspace / f"_opttddft_{name}_partial.json"
        rows = [{"smiles": s, "homo_ev": -7.0, "lumo_ev": 1.0, "gap_ev": 8.0,
                 "max_wavelength_nm": 320.0, "strongest_wavelength_nm": 300.0}
                for s in self.done_smiles]
        partial.write_text(json.dumps({
            "partial": True, "results": rows, "molecules": rows, "states": [],
            "failures": [], "images": [],
        }), encoding="utf-8")
        return SandboxResult("", "", self.returncode, self.timed_out, "")


def test_orbitals_returns_partial_results_on_timeout(tmp_path):
    requested = ["c1ccccc1", "CCO", "C=O"]
    result = opttddft.calculate_orbitals(
        tmp_path, requested, sandbox=TimingOutSandbox(tmp_path, ["c1ccccc1"]))

    assert result.status == "partial"          # 完了分は失われない
    assert result.error_type == "timeout" and result.retryable
    assert [r["smiles"] for r in result.data["results"]] == ["c1ccccc1"]
    assert result.data["pending"] == ["CCO", "C=O"]     # 残りが分かる
    assert result.data["interrupted"] is True
    assert "打ち切られました" in result.summary and "CCO" in result.summary


def test_tddft_spectrum_returns_partial_results_on_timeout(tmp_path):
    result = opttddft.calculate_tddft_spectrum(
        tmp_path, ["O=c1ccc2ccccc2o1", "CC1=CC(=O)Oc2ccccc21"], nstates=5,
        sandbox=TimingOutSandbox(tmp_path, ["O=c1ccc2ccccc2o1"]))
    assert result.status == "partial" and result.error_type == "timeout"
    assert len(result.data["molecules"]) == 1
    assert result.data["pending"] == ["CC1=CC(=O)Oc2ccccc21"]


def test_partial_is_not_used_for_environment_errors(tmp_path):
    """環境不備のときは部分結果扱いにせず、修復不能として返す。"""
    sandbox = TimingOutSandbox(tmp_path, ["c1ccccc1"], timed_out=False, returncode=125)
    original_run = sandbox.run

    def run(code, script_name="script.py"):
        result = original_run(code, script_name)
        return SandboxResult("", "Unable to find image 'autoharnesschem/opttddft:latest'",
                             125, False, "")

    sandbox.run = run
    result = opttddft.calculate_orbitals(tmp_path, ["c1ccccc1"], sandbox=sandbox)
    assert result.status == "failed"
    assert result.error_type == "missing_environment" and not result.retryable


def test_scripts_checkpoint_after_each_item():
    """スクリプトが 1 件ごとに checkpoint を呼ぶこと（打ち切り対策の要）。"""
    for body, name in ((opttddft._ORBITALS_BODY, "orbitals"),
                       (opttddft._SPECTRUM_BODY, "spectrum"),
                       (opttddft._OPTUNA_BODY, "optuna"),
                       (opttddft._ESIPT_BODY, "esipt")):
        rendered = opttddft._script(body, name).render()
        assert "checkpoint(" in rendered, name
        assert f"_opttddft_{name}_partial.json" in rendered, name


# --- 5. 試行の実時間上限と、必ず report を残すこと -------------------------

class HangingAdapter:
    """1 回目はハングし、2 回目で期待出力を作る Adapter。"""
    name = "dummy"

    def __init__(self, tracer, policy):
        self.tracer = tracer
        self.calls = 0

    async def create(self, config, skills, tools):
        self.skills = skills

    async def run(self, task, state):
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(30)          # attempt_timeout_sec で打ち切られる
            return "should not get here"
        Path(state.workspace, "orbital_features.csv").write_text(
            "smiles,homo_ev,lumo_ev\nO,-13.2,4.1\n")
        return "done"

    async def shutdown(self):
        pass


def _config(tmp_path):
    config = load_config()
    config.paths.workspaces = tmp_path / "workspaces"
    config.paths.traces = tmp_path / "traces"
    config.runtime.sandbox.type = "local"
    config.runtime.attempt_timeout_sec = 1      # テスト用に極端に短くする
    return config


def test_attempt_timeout_recovers_and_still_reports(monkeypatch, tmp_path):
    adapters = []

    def fake_create(provider, tracer, policy):
        adapter = HangingAdapter(tracer, policy)
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(controller_mod, "create_adapter", fake_create)
    events = []
    controller = controller_mod.HarnessController(_config(tmp_path))
    report = asyncio.run(controller.run("水のHOMO/LUMOを計算して",
                                        on_event=events.append))

    # 1 回目は時間切れ → 2 回目で成功し、run 自体は完了する
    assert report.passed and report.attempts == 2
    assert any(e.payload.get("phase") == "attempt_timeout" for e in events)
    # 時間切れの試行には、分割・部分結果保存を促す修復指示が渡る
    attempt2 = next(e for e in events if e.payload.get("phase") == "attempt"
                    and e.payload.get("attempt") == 2)
    assert any("打ち切られました" in r for r in attempt2.payload["repairs"])
    # report は必ず書かれている
    workspace = Path(report.artifacts[0].path).parent
    assert (workspace / "report.json").exists() and (workspace / "report.md").exists()


class BrokenAdapter(HangingAdapter):
    async def run(self, task, state):
        raise RuntimeError("SDK crashed")


def test_report_is_written_even_when_the_adapter_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(controller_mod, "create_adapter",
                        lambda provider, tracer, policy: BrokenAdapter(tracer, policy))
    controller = controller_mod.HarnessController(_config(tmp_path))
    with pytest.raises(RuntimeError, match="SDK crashed"):
        asyncio.run(controller.run("水のHOMO/LUMOを計算して"))

    workspaces = list((tmp_path / "workspaces").glob("run-*"))
    assert workspaces, "workspace が作られていること"
    report = json.loads((workspaces[0] / "report.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["verification"]["requirements_missing"]      # 何が足りないか残る
    ledger = json.loads((workspaces[0] / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["status"] != "running"                       # 開いたままにしない


# --- 6. メモリ上限による SIGSEGV -------------------------------------------

class SegfaultingSandbox:
    """C 拡張がアドレス空間上限に達して SIGSEGV する状況を再現する。"""

    def __init__(self, workspace, memory_limit_mb=4096):
        self.config = SandboxConfig(type="local", conda_env="pyscf",
                                    memory_limit_mb=memory_limit_mb)
        self.workspace = Path(workspace)
        self.limits = []

    def run(self, code, script_name="script.py"):
        self.limits.append(self.config.memory_limit_mb)
        return SandboxResult("", "", 139, False, "")


def test_segfault_is_reported_as_out_of_memory(tmp_path):
    result = opttddft.calculate_tddft_spectrum(
        tmp_path, ["O=c1ccc2ccccc2o1"], sandbox=SegfaultingSandbox(tmp_path))
    assert result.status == "failed"
    assert result.error_type == "out_of_memory" and result.retryable
    assert "memory_limit_mb" in result.summary and "SIGSEGV" in result.summary


def test_quantum_tools_raise_the_default_memory_limit(tmp_path):
    """4096MB では実用的な TDDFT が落ちるため、ツール既定は大きめにする。"""
    assert opttddft.DEFAULT_MEMORY_LIMIT_MB >= 8192

    captured = {}
    original = opttddft.run_env_script

    def spy(sandbox, workspace, script, spec, **kwargs):
        captured.update(kwargs)
        return original(sandbox, workspace, script, spec, **kwargs)

    opttddft.run_env_script = spy
    try:
        opttddft.calculate_tddft_spectrum(tmp_path, ["CCO"],
                                          sandbox=SegfaultingSandbox(tmp_path))
    finally:
        opttddft.run_env_script = original
    assert captured["memory_limit_mb"] == opttddft.DEFAULT_MEMORY_LIMIT_MB


# --- 7 & 8. RDKit 系ツールの専用環境化とツール呼び出しの引数衝突 ------------

def test_rdkit_tools_run_in_a_dedicated_env(tmp_path, monkeypatch):
    """harness 環境に rdkit が無くても standardize_smiles 等が使えること。"""
    from harness.policy import PolicyGate
    from tools.registry import build_default_registry
    import tools.registry as registry_mod

    created = []
    original = registry_mod.create_sandbox
    monkeypatch.setattr(registry_mod, "create_sandbox",
                        lambda config, ws, env=None: (created.append(env),
                                                      original(config, ws, env=env))[1])
    registry = build_default_registry(tmp_path, SandboxConfig(type="local"), PolicyGate())
    assert "rdkit" in created          # chemenv 用の専用環境が作られる
    for name in ("standardize_smiles", "generate_3d_structure",
                 "calculate_rdkit_descriptors", "inspect_dataset",
                 "cross_validate_model"):
        assert name in registry.names()

    from tools import chem
    for gone in ("standardize_smiles", "generate_3d_structure",
                 "calculate_rdkit_descriptors", "inspect_dataset",
                 "cross_validate_model"):
        assert not hasattr(chem, gone), gone      # 実装は chemenv.py が正本


def test_tool_named_argument_does_not_collide_with_registry_call(tmp_path):
    """generate_3d_structure(name=...) のように `name` 引数を持つツールを呼べること。"""
    from harness.policy import PolicyGate
    from tools.registry import ToolRegistry, ToolSpec, build_default_registry

    registry = ToolRegistry()
    registry.register(ToolSpec(name="echo", description="", parameters={},
                               func=lambda name=None: __import__("schemas").ToolResult(
                                   status="success", summary=f"name={name}")))
    assert registry.call("echo", name="coumarin").summary == "name=coumarin"

    real = build_default_registry(tmp_path, SandboxConfig(type="local"), PolicyGate())
    result = real.call("generate_3d_structure", smiles="CCO", name="ethanol")
    # 専用環境が無い CI では missing_environment 等になるが、引数衝突では落ちない
    assert result.error_type != "tool_exception"


# --- 構造最適化を伴う計算の環境振り分け（geomeTRIC 専用環境） ----------------

def test_geometry_optimization_uses_the_esipt_env(tmp_path, monkeypatch):
    """geomeTRIC が必要な呼び出しだけ pyscf_esipt 環境へ振り分けること。"""
    from harness.policy import PolicyGate
    from tools.registry import build_default_registry
    import tools.registry as registry_mod

    boxes = {}
    original = registry_mod.create_sandbox

    def spy(config, ws, env=None):
        box = original(config, ws, env=env)
        boxes[env] = box
        return box

    monkeypatch.setattr(registry_mod, "create_sandbox", spy)
    registry = build_default_registry(tmp_path, SandboxConfig(type="local"), PolicyGate())
    assert boxes["esipt"].config.conda_env == "pyscf_esipt"
    assert boxes["opttddft"].config.conda_env == "pyscf"

    used = []
    monkeypatch.setattr(opttddft, "calculate_orbitals",
                        lambda ws, sandbox=None, **kw: used.append(sandbox.config.conda_env))
    registry.call("calculate_orbitals", smiles=["CCO"])                    # 既定
    registry.call("calculate_orbitals", smiles=["CCO"], use_geom_opt=True)  # 構造最適化あり
    assert used == ["pyscf", "pyscf_esipt"]

    monkeypatch.setattr(opttddft, "scan_esipt_pes",
                        lambda ws, sandbox=None, **kw: used.append(sandbox.config.conda_env))
    registry.call("scan_esipt_pes", atom_idx_1=1, atom_idx_2=2, xyz="O 0 0 0")
    assert used[-1] == "pyscf_esipt"      # 拘束付き最適化は常に geomeTRIC 環境


def test_chemenv_raises_memory_limit_for_heavy_imports(tmp_path, monkeypatch):
    """rdkit / pandas / sklearn の import は 4096MB では足りない。"""
    from tools import chemenv

    assert chemenv.DEFAULT_MEMORY_LIMIT_MB > 4096
    captured = {}
    original = chemenv.run_env_script

    def spy(sandbox, workspace, script, spec, **kwargs):
        captured.update(kwargs)
        return original(sandbox, workspace, script, spec, **kwargs)

    monkeypatch.setattr(chemenv, "run_env_script", spy)
    chemenv.standardize_smiles(tmp_path, ["CCO"],
                               sandbox=SegfaultingSandbox(tmp_path))
    assert captured["memory_limit_mb"] == chemenv.DEFAULT_MEMORY_LIMIT_MB


# --- 設計タスクの目的関数（暗状態を追わない） ------------------------------

def test_design_objective_targets_the_bright_band(tmp_path):
    """既定は振動子強度が最大の吸収帯。最長波長（暗状態）は明示指定したときだけ。"""
    sandbox = SegfaultingSandbox(tmp_path)
    opttddft.optimize_absorption_wavelength(
        tmp_path, "c1cc([*:1])ccc1[*:2]", ["H"], ["H"], target_wavelength_nm=330.0,
        sandbox=sandbox)
    spec = json.loads((tmp_path / "_opttddft_optuna_input.json").read_text())
    assert spec["objective"] == "strongest"
    assert spec["min_oscillator_strength"] == 0.01

    opttddft.optimize_absorption_wavelength(
        tmp_path, "c1cc([*:1])ccc1[*:2]", ["H"], ["H"], objective="longest",
        sandbox=sandbox)
    assert json.loads(
        (tmp_path / "_opttddft_optuna_input.json").read_text())["objective"] == "longest"

    bad = opttddft.optimize_absorption_wavelength(
        tmp_path, "c1cc([*:1])ccc1[*:2]", ["H"], ["H"], objective="brightest",
        sandbox=sandbox)
    assert bad.status == "failed" and bad.error_type == "invalid_input"


def test_optuna_script_picks_wavelength_by_intensity():
    """スクリプト側が振動子強度で波長を選び、強度も trial に残すこと。"""
    rendered = opttddft._script(opttddft._OPTUNA_BODY, "optuna").render()
    assert "def pick_wavelength" in rendered
    assert 'set_user_attr("oscillator_strengths"' in rendered
    assert 'set_user_attr("objective_wavelength_nm"' in rendered
    # opt_tddft の既定（最長波長）をそのまま使っていない
    assert "self.objective" in rendered


# --- 1分子ずつ呼んでも CSV が消えない（分割実行との整合） ------------------

class MergingSandbox:
    """スクリプトを実際には動かさず、渡された rows を merge_csv 相当で書く sandbox。"""

    def __init__(self, workspace, rows):
        self.config = SandboxConfig(type="local", conda_env="pyscf")
        self.workspace = Path(workspace)
        self.rows = rows

    def run(self, code, script_name="script.py"):
        spec = json.loads((self.workspace / "_opttddft_orbitals_input.json")
                          .read_text(encoding="utf-8"))
        # 生成スクリプトの merge_csv 部分だけを取り出して実行する
        namespace = {"csv": __import__("csv"), "Path": Path}
        marker = "def merge_csv"
        body = code[code.index(marker):code.index("def finish(")]
        exec(compile(body, "<merge>", "exec"), namespace)
        # 実ツールと同じく smiles を先頭列にする
        rows = [{"smiles": s, **r} for s, r in zip(spec["smiles"], self.rows)]
        n = namespace["merge_csv"](self.workspace / spec["output_csv"], rows,
                                   ("smiles", "method", "basis"))
        (self.workspace / "_opttddft_orbitals_output.json").write_text(json.dumps(
            {"results": rows, "failures": [], "n_rows_in_csv": n}), encoding="utf-8")
        return SandboxResult("", "", 0, False, "")


def test_incremental_calls_do_not_overwrite_the_csv(tmp_path):
    """重い計算は1分子ずつ呼ぶ運用なので、同じ output_csv でも前の分子を消さない。"""
    for smiles, homo in (("O", -10.6), ("CCO", -9.4), ("C=O", -9.7)):
        row = {"method": "HF", "basis": "sto-3g", "homo_ev": homo, "lumo_ev": 5.0}
        result = opttddft.calculate_orbitals(
            tmp_path, [smiles], sandbox=MergingSandbox(tmp_path, [row]))
        assert result.status == "success"
    text = (tmp_path / "orbital_features.csv").read_text(encoding="utf-8")
    assert text.count("\n") == 4          # ヘッダ + 3 分子
    for smiles in ("O", "CCO", "C=O"):
        assert f"{smiles},HF,sto-3g" in text
    assert "累計 3 行" in result.summary

    # 同じ分子を別の手法で計算すると行は置き換わらず追加される
    row = {"method": "b3lyp", "basis": "sto-3g", "homo_ev": -3.9, "lumo_ev": 8.9}
    opttddft.calculate_orbitals(tmp_path, ["O"], method="b3lyp",
                                sandbox=MergingSandbox(tmp_path, [row]))
    text = (tmp_path / "orbital_features.csv").read_text(encoding="utf-8")
    assert "O,HF,sto-3g" in text and "O,b3lyp,sto-3g" in text


def test_scripts_merge_instead_of_overwrite():
    for name, body in (("orbitals", opttddft._ORBITALS_BODY),
                       ("spectrum", opttddft._SPECTRUM_BODY)):
        rendered = opttddft._script(body, name).render()
        assert "merge_csv(" in rendered, name
        # 同じ CSV を write_csv で上書きしない
        assert 'write_csv(spec["output_csv"]' not in rendered, name
