"""Indirect prompt-injection detection and independent tool permissions."""

from injection_guard.trace import ToolCall, Trace, load_traces

__all__ = ["ToolCall", "Trace", "load_traces"]
__version__ = "0.1.0"
