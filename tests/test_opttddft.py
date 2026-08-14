"""OptTDDFT ベースの量子化学ツールの単体テスト（pyscf 不要 — sandbox はスタブ）。"""
import json
from pathlib import Path

import pytest

from schemas import SandboxConfig
from tools import opttddft
from tools.envrun import with_limits
from tools.sandbox import SandboxResult


class StubSandbox:
    """run() で受け取ったスクリプトを記録し、あらかじめ決めた結果を返す sandbox。"""

    def __init__(self, workspace, stdout="", stderr="", returncode=0, timed_out=False,
                 payload=None, files=(), config=None):
        self.config = config or SandboxConfig(type="local", conda_env="pyscf",
                                              timeout_sec=600, cpu_limit_sec=300)
        self.workspace = Path(workspace)
        self._result = SandboxResult(stdout, stderr, returncode, timed_out, "")
        self._payload = payload
        self._files = files
        self.ran = []

    def run(self, code, script_name="script.py"):
        self.ran.append((script_name, code))
        if self._payload is not None:
            output = next(p for p in self.workspace.glob("_opttddft_*_input.json"))
            (self.workspace / output.name.replace("_input", "_output")).write_text(
                json.dumps(self._payload), encoding="utf-8")
        for name in self._files:
            (self.workspace / name).write_text("x", encoding="utf-8")
        return self._result


def _spec(sandbox) -> dict:
    """スクリプトへ渡された入力 JSON を読む。"""
    path = next(sandbox.workspace.glob("_opttddft_*_input.json"))
    return json.loads(path.read_text(encoding="utf-8"))


# --- スクリプトの静的な健全性 ---------------------------------------------

@pytest.mark.parametrize("body,name", [
    (opttddft._ORBITALS_BODY, "orbitals"),
    (opttddft._SPECTRUM_BODY, "spectrum"),
    (opttddft._OPTUNA_BODY, "optuna"),
    (opttddft._ESIPT_BODY, "esipt"),
])
def test_generated_scripts_are_valid_python(body, name):
    script = opttddft._script(body, name)
    compile(script.render(), f"<{name}>", "exec")
    assert "__INPUT_JSON__" not in script.render()
    assert "opt_tddft" in script.render()


def test_prelude_limits_threads_before_importing_pyscf():
    rendered = opttddft._script(opttddft._ORBITALS_BODY, "orbitals").render()
    assert rendered.index("OMP_NUM_THREADS") < rendered.index("from pyscf import")


# --- calculate_orbitals ----------------------------------------------------

def test_calculate_orbitals_success(tmp_path):
    payload = {"results": [{"smiles": "O", "homo_ev": -10.6, "lumo_ev": 15.6,
                            "gap_ev": 26.2, "total_energy_hartree": -74.9}],
               "failures": []}
    sandbox = StubSandbox(tmp_path, payload=payload, files=("orbital_features.csv",))
    result = opttddft.calculate_orbitals(tmp_path, ["O"], sandbox=sandbox)

    assert result.status == "success"
    assert result.data["engine"] == "opt_tddft"
    assert result.artifacts and result.artifacts[0].path.endswith("orbital_features.csv")
    # OptTDDFT の SolverConfig に渡す設定がスクリプトへ届いている
    spec = _spec(sandbox)
    assert spec["method"] == "HF" and spec["basis"] == "sto-3g"
    assert spec["nstates"] == 0          # 軌道のみ（TDDFT なし）
    assert spec["solvent_model"] is None
    assert spec["opt_tddft_root"].endswith("OptTDDFT")


def test_calculate_orbitals_partial_and_all_failed(tmp_path):
    partial = opttddft.calculate_orbitals(
        tmp_path, ["O", "bad"], sandbox=StubSandbox(
            tmp_path, payload={"results": [{"smiles": "O", "homo_ev": -1, "lumo_ev": 1}],
                               "failures": [{"smiles": "bad", "error": "x"}]}))
    assert partial.status == "partial" and partial.error_type == "scf_failed"

    all_failed = opttddft.calculate_orbitals(
        tmp_path, ["bad"], sandbox=StubSandbox(
            tmp_path, payload={"results": [], "failures": [{"smiles": "bad"}]}))
    assert all_failed.status == "failed" and all_failed.retryable


def test_calculate_orbitals_rejects_empty_smiles(tmp_path):
    sandbox = StubSandbox(tmp_path)
    result = opttddft.calculate_orbitals(tmp_path, [], sandbox=sandbox)
    assert result.status == "failed" and result.error_type == "invalid_input"
    assert sandbox.ran == []


def test_solvent_and_charge_are_passed_through(tmp_path):
    sandbox = StubSandbox(tmp_path, payload={"results": [{"smiles": "O", "homo_ev": -1,
                                                          "lumo_ev": 1}], "failures": []})
    opttddft.calculate_orbitals(tmp_path, ["[O-]S(=O)(=O)[O-]"], charge=-2, spin=0,
                                solvent_model="pcm", solvent_eps=78.4, method="b3lyp",
                                basis="6-31g(d)", sandbox=sandbox)
    spec = _spec(sandbox)
    assert (spec["charge"], spec["spin"]) == (-2, 0)
    assert spec["solvent_model"] == "pcm" and spec["solvent_eps"] == 78.4
    assert spec["functional"] == "b3lyp"


# --- 失敗の分類 ------------------------------------------------------------

def test_failure_classification(tmp_path):
    missing_env = opttddft.calculate_orbitals(tmp_path, ["O"], sandbox=StubSandbox(
        tmp_path, stderr="EnvironmentLocationNotFound: pyscf", returncode=1))
    assert missing_env.error_type == "missing_environment" and not missing_env.retryable

    geometric = opttddft.scan_esipt_pes(
        tmp_path, atom_idx_1=1, atom_idx_2=2, xyz="O 0 0 0\nH 1 0 0\nN 2 0 0\nC 3 0 0",
        sandbox=StubSandbox(tmp_path, returncode=1,
                            stderr="ModuleNotFoundError: No module named 'geometric'"))
    assert geometric.error_type == "missing_dependency"
    assert "geomeTRIC" in geometric.summary

    basis = opttddft.calculate_orbitals(tmp_path, ["Br"], basis="6-31g(d)",
                                        sandbox=StubSandbox(
        tmp_path, returncode=1, stderr="BasisNotFoundError: Basis set not found for Br"))
    assert basis.error_type == "invalid_input"

    timeout = opttddft.calculate_orbitals(tmp_path, ["O"], sandbox=StubSandbox(
        tmp_path, timed_out=True, returncode=124))
    assert timeout.error_type == "timeout" and timeout.retryable

    killed = opttddft.calculate_orbitals(tmp_path, ["O"], sandbox=StubSandbox(
        tmp_path, returncode=137, stderr="<locale-dependent kill message>"))
    assert killed.error_type == "out_of_memory" and killed.retryable


# --- calculate_tddft_spectrum ---------------------------------------------

def test_tddft_spectrum_success(tmp_path):
    payload = {
        "states": [{"smiles": "C=O", "state_index": 1, "wavelength_nm": 321.8,
                    "oscillator_strength": 0.0}],
        "molecules": [{"smiles": "C=O", "homo_ev": -4.4, "lumo_ev": 1.8, "gap_ev": 6.2,
                       "max_wavelength_nm": 321.8, "strongest_wavelength_nm": 105.7}],
        "failures": [], "images": ["tddft_spectrum_1.png"],
    }
    sandbox = StubSandbox(tmp_path, payload=payload,
                          files=("tddft_spectrum.csv", "orbital_features.csv",
                                 "tddft_spectrum_1.png"))
    result = opttddft.calculate_tddft_spectrum(tmp_path, ["C=O"], functional="b3lyp",
                                               basis="sto-3g", nstates=5,
                                               sandbox=sandbox)
    assert result.status == "success"
    assert "322nm" in result.summary  # λmax は四捨五入して表示する
    # 状態リスト・軌道 CSV・スペクトル画像がすべて artifact になる
    names = {Path(a.path).name for a in result.artifacts}
    assert names == {"tddft_spectrum.csv", "orbital_features.csv", "tddft_spectrum_1.png"}
    assert _spec(sandbox)["nstates"] == 5


def test_tddft_spectrum_requires_states(tmp_path):
    result = opttddft.calculate_tddft_spectrum(tmp_path, ["C=O"], nstates=0,
                                               sandbox=StubSandbox(tmp_path))
    assert result.status == "failed" and result.error_type == "invalid_input"


# --- optimize_absorption_wavelength ---------------------------------------

def test_optimization_validates_scaffold(tmp_path):
    sandbox = StubSandbox(tmp_path)
    bad = opttddft.optimize_absorption_wavelength(
        tmp_path, "c1ccccc1", ["H"], ["H"], sandbox=sandbox)
    assert bad.status == "failed" and bad.error_type == "invalid_input"
    assert sandbox.ran == []


def test_optimization_success_and_no_valid_trial(tmp_path):
    summary = {"n_completed_trials": 2, "n_trials_total": 3, "best_smiles": "Nc1ccccc1",
               "best_wavelength_nm": 228.9, "difference_from_target_nm": 71.1}
    sandbox = StubSandbox(tmp_path, payload={"summary": summary, "trials": [],
                                             "reports": []},
                          files=("optuna_trials.csv", "optimization_summary.json"))
    result = opttddft.optimize_absorption_wavelength(
        tmp_path, "c1cc([*:1])ccc1[*:2]", ["H", "C"], ["H", "N"],
        target_wavelength_nm=300.0, n_trials=3, sandbox=sandbox)
    assert result.status == "success" and "Nc1ccccc1" in result.summary
    spec = _spec(sandbox)
    assert spec["side_chains_pos1"] == ["H", "C"]
    assert spec["target_wavelength_nm"] == 300.0

    pruned = opttddft.optimize_absorption_wavelength(
        tmp_path, "c1cc([*:1])ccc1[*:2]", ["H"], ["H"],
        sandbox=StubSandbox(tmp_path, payload={"summary": {"n_completed_trials": 0,
                                                          "n_trials_total": 5}}))
    assert pruned.status == "failed" and pruned.error_type == "no_valid_trial"


def test_optimization_gives_sandbox_time_to_write_outputs(tmp_path):
    """探索の打ち切り時間より sandbox の timeout が長いこと。"""
    sandbox = StubSandbox(tmp_path, payload={"summary": {"n_completed_trials": 1,
                                                         "best_smiles": "C",
                                                         "best_wavelength_nm": 300.0,
                                                         "difference_from_target_nm": 0.0}})
    captured = {}
    original = opttddft.run_env_script

    def spy(sb, ws, script, spec, **kwargs):
        captured.update(kwargs)
        return original(sb, ws, script, spec, **kwargs)

    opttddft.run_env_script = spy
    try:
        opttddft.optimize_absorption_wavelength(
            tmp_path, "c1cc([*:1])ccc1[*:2]", ["H"], ["H"], search_timeout_sec=900,
            sandbox=sandbox)
    finally:
        opttddft.run_env_script = original
    assert captured["timeout_sec"] > 900


# --- scan_esipt_pes --------------------------------------------------------

XYZ = """O   -1.3   1.2   0.0
H   -0.5   1.7   0.0
N    1.2   0.5   0.0
C   -1.2  -0.1   0.0"""


def test_esipt_validates_indices_and_range(tmp_path):
    sandbox = StubSandbox(tmp_path)
    out_of_range = opttddft.scan_esipt_pes(tmp_path, atom_idx_1=1, atom_idx_2=9,
                                           xyz=XYZ, sandbox=sandbox)
    assert out_of_range.error_type == "invalid_input"

    same = opttddft.scan_esipt_pes(tmp_path, atom_idx_1=1, atom_idx_2=1, xyz=XYZ,
                                   sandbox=sandbox)
    assert same.error_type == "invalid_input"

    bad_range = opttddft.scan_esipt_pes(tmp_path, atom_idx_1=1, atom_idx_2=2, xyz=XYZ,
                                        start_dist=2.0, end_dist=1.0, sandbox=sandbox)
    assert bad_range.error_type == "invalid_input"

    no_geometry = opttddft.scan_esipt_pes(tmp_path, atom_idx_1=1, atom_idx_2=2,
                                          sandbox=sandbox)
    assert no_geometry.error_type == "invalid_input"
    assert sandbox.ran == []


def test_esipt_reads_xyz_file_and_strips_header(tmp_path):
    (tmp_path / "mol.xyz").write_text("4\ncomment\n" + XYZ + "\n", encoding="utf-8")
    payload = {"summary": {"n_points": 3, "s0_barrier_kcal": 1.2, "s1_barrier_kcal": 0.4},
               "points": [], "failures": []}
    sandbox = StubSandbox(tmp_path, payload=payload,
                          files=("esipt_scan_results.csv", "esipt_pes_profile.png"))
    result = opttddft.scan_esipt_pes(tmp_path, atom_idx_1=1, atom_idx_2=2,
                                     xyz_file="mol.xyz", start_dist=1.0, end_dist=1.2,
                                     step_size=0.1, sandbox=sandbox)
    assert result.status == "success"
    spec = _spec(sandbox)
    assert spec["initial_xyz"].splitlines()[0].startswith("O")   # ヘッダが落ちている
    assert len(spec["initial_xyz"].splitlines()) == 4


def test_esipt_missing_xyz_file(tmp_path):
    result = opttddft.scan_esipt_pes(tmp_path, atom_idx_1=1, atom_idx_2=2,
                                     xyz_file="nope.xyz", sandbox=StubSandbox(tmp_path))
    assert result.error_type == "input_not_found"


# --- 実行上限の差し替え ----------------------------------------------------

def test_with_limits_scales_cpu_limit_by_threads(tmp_path):
    sandbox = StubSandbox(tmp_path)
    tightened = with_limits(sandbox, timeout_sec=1200, threads=4, memory_limit_mb=8192)
    assert tightened.config.timeout_sec == 1200
    # 全スレッド分 + 1 スレッド分の余裕（実時間の timeout を先に効かせる）
    assert tightened.config.cpu_limit_sec == 1200 * 5
    assert tightened.config.memory_limit_mb == 8192
    assert sandbox.config.timeout_sec == 600           # 元の sandbox は変えない
    assert with_limits(sandbox) is sandbox


# --- レジストリへの登録 ----------------------------------------------------

def test_registered_in_dedicated_env(tmp_path, monkeypatch):
    from harness.policy import PolicyGate
    from tools.registry import build_default_registry
    import tools.registry as registry_mod

    created = []
    original = registry_mod.create_sandbox
    monkeypatch.setattr(registry_mod, "create_sandbox",
                        lambda config, ws, env=None: (created.append(env),
                                                      original(config, ws, env=env))[1])
    registry = build_default_registry(tmp_path, SandboxConfig(type="local"), PolicyGate())

    for name in ("calculate_orbitals", "calculate_tddft_spectrum",
                 "optimize_absorption_wavelength", "scan_esipt_pes"):
        assert name in registry.names()
    # 量子化学は opttddft 環境、ReactionT5 と AiZynth はそれぞれ専用環境
    assert {"opttddft", "reactiont5", "aizynth"} <= set(created)
    assert registry.get("calculate_orbitals").risk_level == "medium"


def test_chem_module_no_longer_owns_orbitals():
    """pyscf 実装は tools/chem.py から opttddft へ置き換えられている。"""
    from tools import chem

    assert not hasattr(chem, "calculate_orbitals")
