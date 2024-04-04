from pathlib import Path

from injection_guard.permissions import InMemoryToolExecutor
from injection_guard.trace import load_traces

DATA = Path(__file__).parents[1] / "data" / "traces.json"


def test_local_trace_splits_and_attack_coverage() -> None:
    traces = load_traces(DATA)
    train = [trace for trace in traces if trace.split == "train"]
    test = [trace for trace in traces if trace.split == "test"]

    assert len(train) == 32
    assert len(test) == 24
    assert sum(trace.is_malicious for trace in train) == 16
    assert sum(trace.is_malicious for trace in test) == 12
    assert {
        trace.attack_type for trace in test if trace.is_malicious
    } == {"confused_deputy", "data_exfiltration", "privilege_escalation"}
    assert len({trace.tool_output for trace in traces}) == len(traces)


def test_unprotected_executor_demonstrates_every_held_out_attack_effect() -> None:
    executor = InMemoryToolExecutor()
    attacks = [
        trace for trace in load_traces(DATA) if trace.split == "test" and trace.is_malicious
    ]

    results = [executor.execute(trace.requested_action) for trace in attacks]

    assert all(result.executed for result in results)
    assert executor.state.outbound_requests
    assert executor.state.principal_grants
