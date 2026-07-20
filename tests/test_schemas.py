from schemas import AgentEvent, TaskSpec, ToolResult, VerificationResult


def test_tool_result_roundtrip():
    result = ToolResult(status="partial", summary="2/3 ok", retryable=True,
                        error_type="scf_failed")
    restored = ToolResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_task_spec_defaults():
    task = TaskSpec(description="calc HOMO/LUMO")
    assert task.task_id.startswith("task-")
    assert task.task_type == "generic"
    assert task.repairs == []


def test_agent_event_serializes_timestamp():
    event = AgentEvent(run_id="run-x", event_type="tool_call", actor="claude",
                       payload={"tool": "calculate_orbitals"})
    data = event.model_dump_json()
    assert "tool_call" in data and "run-x" in data


def test_verification_result_shape():
    v = VerificationResult(passed=False, requirements_missing=["output missing: a.csv"],
                           required_repairs=["make a.csv"])
    assert not v.passed
    assert v.scientific_warnings == []
