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
