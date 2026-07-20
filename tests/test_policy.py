from pathlib import Path

from harness.policy import PolicyGate


def test_blocks_socket_and_eval(monkeypatch):
    monkeypatch.delenv("ENABLE_UNSAFE_PY_RUNNER", raising=False)
    gate = PolicyGate()
    assert gate.check_code("import socket\n").status == "blocked"
    assert gate.check_code("x = eval('1+1')").status == "blocked"
    assert gate.check_code("import numpy as np\nprint(np.pi)") is None


def test_unsafe_override(monkeypatch):
    monkeypatch.setenv("ENABLE_UNSAFE_PY_RUNNER", "1")
    gate = PolicyGate()
    assert gate.check_code("import socket") is None


def test_write_path_restriction(tmp_path):
    gate = PolicyGate(allowed_write_roots=[tmp_path])
    assert gate.check_write_path(tmp_path / "out.csv") is None
    assert gate.check_write_path(Path("/etc/passwd")).status == "blocked"


def test_approval_modes():
    assert PolicyGate(approval="deny").approve("dangerous", "high") is False
    assert PolicyGate(approval="allow").approve("dangerous", "high") is True
    assert PolicyGate(approval="deny").approve("routine", "low") is True
