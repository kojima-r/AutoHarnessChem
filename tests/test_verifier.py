import json

from harness.verifier import ScientificVerifier
from schemas import TaskSpec


def test_missing_output_fails(tmp_path):
    task = TaskSpec(description="x", task_type="orbital_calculation",
                    expected_outputs=["orbital_features.csv"])
    v = ScientificVerifier().verify(task, tmp_path)
    assert not v.passed
    assert v.requirements_missing
    assert v.required_repairs


def test_valid_orbitals_pass(tmp_path):
    (tmp_path / "orbital_features.csv").write_text(
        "smiles,homo_ev,lumo_ev\nO,-13.2,4.1\nc1ccccc1,-6.7,2.9\n")
    task = TaskSpec(description="x", task_type="orbital_calculation",
                    expected_outputs=["orbital_features.csv"])
    v = ScientificVerifier().verify(task, tmp_path)
    assert v.passed
    assert v.scientific_warnings == []


def test_homo_above_lumo_requires_repair(tmp_path):
    (tmp_path / "orbital_features.csv").write_text(
        "smiles,homo_ev,lumo_ev\nO,4.1,-13.2\n")
    task = TaskSpec(description="x", task_type="orbital_calculation",
                    expected_outputs=["orbital_features.csv"])
    v = ScientificVerifier().verify(task, tmp_path)
    assert not v.passed


def test_hartree_scale_values_warn(tmp_path):
    # eV変換忘れ（Hartreeのまま≒-500eV相当にならないが範囲外の例）
    (tmp_path / "orbital_features.csv").write_text(
        "smiles,homo_ev,lumo_ev\nO,-360.0,110.0\n")
    task = TaskSpec(description="x", task_type="orbital_calculation",
                    expected_outputs=["orbital_features.csv"])
    v = ScientificVerifier().verify(task, tmp_path)
    assert v.scientific_warnings


def test_regression_leak_warning(tmp_path):
    (tmp_path / "cv_metrics.json").write_text(json.dumps({"r2": 0.9999, "rmse": 0.001}))
    task = TaskSpec(description="x", task_type="molecular_regression",
                    expected_outputs=["cv_metrics.json"])
    v = ScientificVerifier().verify(task, tmp_path)
    assert any("リーク" in w for w in v.scientific_warnings)


def test_invalid_r2_fails(tmp_path):
    (tmp_path / "cv_metrics.json").write_text(json.dumps({"r2": 1.7}))
    task = TaskSpec(description="x", task_type="molecular_regression",
                    expected_outputs=["cv_metrics.json"])
    v = ScientificVerifier().verify(task, tmp_path)
    assert not v.passed


# --- TDDFT スペクトル（軌道計算タスクの一部として検査される） ---------------

def test_valid_spectrum_passes(tmp_path):
    (tmp_path / "tddft_spectrum.csv").write_text(
        "smiles,state_index,wavelength_nm,oscillator_strength\n"
        "C=O,1,321.8,0.0001\nC=O,2,134.4,0.02\n")
    task = TaskSpec(description="x", task_type="orbital_calculation",
                    expected_outputs=["tddft_spectrum.csv"])
    v = ScientificVerifier().verify(task, tmp_path)
    assert v.passed and v.scientific_warnings == []


def test_spectrum_out_of_range_requires_repair(tmp_path):
    # 励起エネルギー (eV) をそのまま波長として書いてしまったケース
    (tmp_path / "tddft_spectrum.csv").write_text(
        "smiles,state_index,wavelength_nm,oscillator_strength\nC=O,1,3.85,0.01\n")
    task = TaskSpec(description="x", task_type="orbital_calculation",
                    expected_outputs=["tddft_spectrum.csv"])
    v = ScientificVerifier().verify(task, tmp_path)
    assert not v.passed
    assert any("1240" in r for r in v.required_repairs)


# --- 分子設計（Optuna 探索） -----------------------------------------------

def test_molecular_design_summary_passes(tmp_path):
    (tmp_path / "optimization_summary.json").write_text(json.dumps({
        "n_completed_trials": 3, "best_smiles": "Nc1ccccc1",
        "best_wavelength_nm": 228.9, "target_wavelength_nm": 300.0,
        "difference_from_target_nm": 71.1}))
    task = TaskSpec(description="x", task_type="molecular_design",
                    expected_outputs=["optimization_summary.json"])
    v = ScientificVerifier().verify(task, tmp_path)
    assert v.passed and v.scientific_warnings == []


def test_molecular_design_without_valid_trial_fails(tmp_path):
    (tmp_path / "optimization_summary.json").write_text(json.dumps({
        "n_completed_trials": 0, "n_trials_total": 5, "best_smiles": None}))
    task = TaskSpec(description="x", task_type="molecular_design",
                    expected_outputs=["optimization_summary.json"])
    v = ScientificVerifier().verify(task, tmp_path)
    assert not v.passed
    assert any("n_completed_trials=0" in r for r in v.required_repairs)


def test_molecular_design_objective_mismatch_warns(tmp_path):
    (tmp_path / "optimization_summary.json").write_text(json.dumps({
        "n_completed_trials": 1, "best_smiles": "C", "best_wavelength_nm": 400.0,
        "target_wavelength_nm": 300.0, "difference_from_target_nm": 5.0}))
    task = TaskSpec(description="x", task_type="molecular_design")
    v = ScientificVerifier().verify(task, tmp_path)
    assert any("目的関数" in w for w in v.scientific_warnings)


# --- PES スキャン ----------------------------------------------------------

def test_pes_scan_passes(tmp_path):
    (tmp_path / "esipt_scan_results.csv").write_text(
        "distance_angstrom,s0_energy_hartree,s1_energy_hartree\n"
        "1.0,-300.10,-299.98\n1.1,-300.12,-300.01\n1.2,-300.09,-299.99\n")
    task = TaskSpec(description="x", task_type="pes_scan",
                    expected_outputs=["esipt_scan_results.csv"])
    v = ScientificVerifier().verify(task, tmp_path)
    assert v.passed and v.scientific_warnings == []


def test_pes_scan_excited_below_ground_requires_repair(tmp_path):
    (tmp_path / "esipt_scan_results.csv").write_text(
        "distance_angstrom,s0_energy_hartree,s1_energy_hartree\n"
        "1.0,-300.10,-300.50\n1.1,-300.12,-300.01\n1.2,-300.09,-299.99\n")
    task = TaskSpec(description="x", task_type="pes_scan")
    v = ScientificVerifier().verify(task, tmp_path)
    assert not v.passed
    assert any("S1" in r for r in v.required_repairs)


def test_pes_scan_too_few_points_requires_repair(tmp_path):
    (tmp_path / "esipt_scan_results.csv").write_text(
        "distance_angstrom,s0_energy_hartree,s1_energy_hartree\n1.0,-300.1,-299.9\n")
    task = TaskSpec(description="x", task_type="pes_scan")
    v = ScientificVerifier().verify(task, tmp_path)
    assert not v.passed
    assert any("3 点以上" in r for r in v.required_repairs)
