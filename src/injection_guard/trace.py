"""Typed records for local tool-output traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ToolCall:
        if not isinstance(value.get("name"), str) or not value["name"]:
            raise ValueError("tool call name must be a non-empty string")
        arguments = value.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must be an object")
        return cls(name=value["name"], arguments=dict(arguments))


@dataclass(frozen=True)
class Trace:
    trace_id: str
    split: str
    label: str
    attack_type: str
    source_tool: str
    tool_output: str
    requested_action: ToolCall
    granted_capabilities: tuple[str, ...]

    @property
    def is_malicious(self) -> bool:
        return self.label == "malicious"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Trace:
        split = value.get("split")
        label = value.get("label")
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if label not in {"benign", "malicious"}:
            raise ValueError("label must be 'benign' or 'malicious'")
        output = value.get("tool_output")
        if not isinstance(output, str) or not output.strip():
            raise ValueError("tool_output must be a non-empty string")
        grants = value.get("granted_capabilities", [])
        if not isinstance(grants, list) or not all(isinstance(item, str) for item in grants):
            raise ValueError("granted_capabilities must be a list of strings")
        return cls(
            trace_id=str(value["trace_id"]),
            split=split,
            label=label,
            attack_type=str(value.get("attack_type", "none")),
            source_tool=str(value.get("source_tool", "unknown")),
            tool_output=output,
            requested_action=ToolCall.from_dict(value["requested_action"]),
            granted_capabilities=tuple(grants),
        )


def load_traces(path: str | Path) -> list[Trace]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("trace file must contain a list")
    traces = [Trace.from_dict(record) for record in records]
    identifiers = [trace.trace_id for trace in traces]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("trace_id values must be unique")
    return traces
