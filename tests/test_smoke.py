import pytest

from injection_guard.trace import Trace


def test_trace_schema_loads_tool_call() -> None:
    trace = Trace.from_dict(
        {
            "trace_id": "smoke-1",
            "split": "train",
            "label": "benign",
            "attack_type": "none",
            "source_tool": "search_docs",
            "tool_output": "The deployment guide lists a health-check endpoint.",
            "requested_action": {"name": "read_file", "arguments": {"path": "docs/guide.md"}},
            "granted_capabilities": ["fs:read"],
        }
    )

    assert trace.requested_action.name == "read_file"
    assert not trace.is_malicious


def test_trace_schema_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="label"):
        Trace.from_dict(
            {
                "trace_id": "smoke-2",
                "split": "test",
                "label": "unknown",
                "tool_output": "text",
                "requested_action": {"name": "read_file", "arguments": {}},
            }
        )
