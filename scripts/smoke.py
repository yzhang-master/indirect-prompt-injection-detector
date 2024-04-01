from injection_guard.trace import Trace


def main() -> None:
    trace = Trace.from_dict(
        {
            "trace_id": "smoke",
            "split": "train",
            "label": "benign",
            "tool_output": "The local document is ready.",
            "requested_action": {"name": "read_file", "arguments": {"path": "docs/guide.md"}},
            "granted_capabilities": ["fs:read"],
        }
    )
    assert trace.requested_action.arguments["path"] == "docs/guide.md"
    print("smoke ok: local tool trace parsed")


if __name__ == "__main__":
    main()
