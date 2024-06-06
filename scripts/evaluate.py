import argparse
import json
from pathlib import Path

from injection_guard.classifier import InjectionClassifier
from injection_guard.evaluation import evaluate_held_out
from injection_guard.trace import load_traces

ROOT = Path(__file__).parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on the untouched local test split")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "traces.json")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "model.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "metrics.json")
    args = parser.parse_args()

    classifier = InjectionClassifier.load(args.model)
    report = evaluate_held_out(classifier, load_traces(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "per_trace"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
