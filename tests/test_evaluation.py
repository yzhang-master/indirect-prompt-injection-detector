from pathlib import Path

from injection_guard.classifier import InjectionClassifier
from injection_guard.evaluation import evaluate_held_out
from injection_guard.trace import load_traces

ROOT = Path(__file__).parents[1]


def test_bundled_model_held_out_and_bypass_results() -> None:
    classifier = InjectionClassifier.load(ROOT / "artifacts" / "model.json")
    traces = load_traces(ROOT / "data" / "traces.json")

    report = evaluate_held_out(classifier, traces)

    assert report["detector"] == {
        "threshold": 0.5,
        "precision": 1.0,
        "recall": 1.0,
        "roc_auc": 1.0,
        "tp": 12,
        "fp": 0,
        "tn": 12,
        "fn": 0,
    }
    assert report["attack_success_rate"] == {
        "before": 1.0,
        "after": 0.0,
        "before_successes": 12,
        "after_successes": 0,
    }
    assert report["attack_enforcement"] == {
        "classifier_blocks": 12,
        "permission_denials": 0,
        "unblocked_successes": 0,
    }
    assert report["classifier_bypass_control"] == {
        "attack_success_rate": 0.0,
        "permission_denials": 12,
        "successful_attacks": 0,
    }
    assert report["benign_utility"]["protected_executions"] == 12
