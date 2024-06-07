"""Indirect prompt-injection detection and independent tool permissions."""

from injection_guard.agent import AgentDecision, ProtectedAgent
from injection_guard.classifier import InjectionClassifier
from injection_guard.permissions import InMemoryToolExecutor, PermissionPolicy, SandboxState
from injection_guard.trace import ToolCall, Trace, load_traces

__all__ = [
    "AgentDecision",
    "InjectionClassifier",
    "InMemoryToolExecutor",
    "PermissionPolicy",
    "ProtectedAgent",
    "SandboxState",
    "ToolCall",
    "Trace",
    "load_traces",
]
__version__ = "0.1.0"
