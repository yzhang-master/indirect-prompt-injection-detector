import pytest

from injection_guard.metrics import confusion_counts, precision_recall, roc_auc_score


def test_auc_matches_known_ranking() -> None:
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75


def test_auc_handles_tied_scores() -> None:
    assert roc_auc_score([0, 1], [0.5, 0.5]) == 0.5


def test_auc_rejects_single_class_input() -> None:
    with pytest.raises(ValueError, match="both classes"):
        roc_auc_score([1, 1], [0.2, 0.8])


def test_confusion_and_precision_recall() -> None:
    counts = confusion_counts([1, 1, 0, 0], [True, False, True, False])

    assert counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert precision_recall(counts) == (0.5, 0.5)
