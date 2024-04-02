"""Agent pipeline that keeps statistical detection separate from authorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from injection_guard.permissions import InMemoryToolExecutor, PermissionPolicy
from injection_guard.trace import Trace


class Detector(Protocol):
    threshold: float

    def score(self, text: str) -> float: ...


@dataclass(frozen=True)
class AgentDecision:
    trace_id: str
    detector_score: float
    detector_flagged: bool
    executed: bool
    denial_source: str | None = None
    denial_reason: str | None = None


class ProtectedAgent:
    def __init__(self, detector: Detector, executor: InMemoryToolExecutor) -> None:
        self.detector = detector
        self.executor = executor

    def process(self, trace: Trace) -> AgentDecision:
        score = self.detector.score(trace.tool_output)
        if score >= self.detector.threshold:
            return AgentDecision(
                trace.trace_id, score, True, False, "classifier", "injection_detected"
            )
        policy = PermissionPolicy(self.executor, trace.granted_capabilities)
        result = policy.execute(trace.requested_action)
        return AgentDecision(
            trace_id=trace.trace_id,
            detector_score=score,
            detector_flagged=False,
            executed=result.executed,
            denial_source=None if result.executed else "permission",
            denial_reason=result.denial_reason,
        )
