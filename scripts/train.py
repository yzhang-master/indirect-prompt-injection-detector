import argparse
import json
from pathlib import Path

from injection_guard.classifier import InjectionClassifier
from injection_guard.trace import load_traces

ROOT = Path(__file__).parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on the local train split only")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "traces.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "model.json")
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    traces = [trace for trace in load_traces(args.data) if trace.split == "train"]
    model, history = InjectionClassifier.fit(
        [trace.tool_output for trace in traces],
        [int(trace.is_malicious) for trace in traces],
        training_ids=tuple(trace.trace_id for trace in traces),
        epochs=args.epochs,
    )
    model.save(args.output)
    report = {
        "training_examples": len(traces),
        "training_attacks": sum(trace.is_malicious for trace in traces),
        "epochs": history.epochs,
        "initial_loss": history.initial_loss,
        "final_loss": history.final_loss,
        "model": str(args.output),
    }
    report_path = ROOT / "results" / "training.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
