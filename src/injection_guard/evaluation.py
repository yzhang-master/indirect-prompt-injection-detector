"""Held-out detector and end-to-end attack evaluation."""

from __future__ import annotations

from typing import Any

from injection_guard.agent import ProtectedAgent
from injection_guard.classifier import InjectionClassifier
from injection_guard.metrics import confusion_counts, precision_recall, roc_auc_score
from injection_guard.permissions import InMemoryToolExecutor
from injection_guard.trace import Trace


def evaluate_held_out(
    classifier: InjectionClassifier, traces: list[Trace]
) -> dict[str, Any]:
    held_out = [trace for trace in traces if trace.split == "test"]
    attacks = [trace for trace in held_out if trace.is_malicious]
    benign = [trace for trace in held_out if not trace.is_malicious]
    scores = [classifier.score(trace.tool_output) for trace in held_out]
    labels = [int(trace.is_malicious) for trace in held_out]
    predictions = [score >= classifier.threshold for score in scores]
    counts = confusion_counts(labels, predictions)
    precision, recall = precision_recall(counts)

    baseline_executor = InMemoryToolExecutor()
    baseline_results = [
        baseline_executor.execute(trace.requested_action) for trace in attacks
    ]
    protected_executor = InMemoryToolExecutor()
    agent = ProtectedAgent(classifier, protected_executor)
    decisions = [agent.process(trace) for trace in held_out]
    attack_decisions = [
        decision for decision, trace in zip(decisions, held_out, strict=True) if trace.is_malicious
    ]
    benign_decisions = [
        decision
        for decision, trace in zip(decisions, held_out, strict=True)
        if not trace.is_malicious
    ]
    before_successes = sum(result.executed for result in baseline_results)
    after_successes = sum(decision.executed for decision in attack_decisions)
    classifier_blocks = sum(
        decision.denial_source == "classifier" for decision in attack_decisions
    )
    permission_denials = sum(
        decision.denial_source == "permission" for decision in attack_decisions
    )

    return {
        "dataset": {
            "held_out_examples": len(held_out),
            "held_out_benign": len(benign),
            "held_out_attacks": len(attacks),
            "attack_types": sorted({trace.attack_type for trace in attacks}),
        },
        "detector": {
            "threshold": classifier.threshold,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc_score(labels, scores),
            **counts,
        },
        "attack_success_rate": {
            "before": before_successes / len(attacks),
            "after": after_successes / len(attacks),
            "before_successes": before_successes,
            "after_successes": after_successes,
        },
        "attack_enforcement": {
            "classifier_blocks": classifier_blocks,
            "permission_denials": permission_denials,
            "unblocked_successes": after_successes,
        },
        "benign_utility": {
            "protected_executions": sum(decision.executed for decision in benign_decisions),
            "total": len(benign),
            "protected_execution_rate": sum(decision.executed for decision in benign_decisions)
            / len(benign),
        },
        "per_trace": [
            {
                "trace_id": trace.trace_id,
                "label": trace.label,
                "attack_type": trace.attack_type,
                "score": score,
                "detector_flagged": decision.detector_flagged,
                "executed": decision.executed,
                "denial_source": decision.denial_source,
                "denial_reason": decision.denial_reason,
            }
            for trace, score, decision in zip(held_out, scores, decisions, strict=True)
        ],
    }
