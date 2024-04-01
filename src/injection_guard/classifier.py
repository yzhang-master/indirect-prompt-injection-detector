"""A small, fully local logistic-regression injection classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from injection_guard.features import HashedFeatureExtractor


@dataclass(frozen=True)
class TrainingHistory:
    initial_loss: float
    final_loss: float
    epochs: int


class InjectionClassifier:
    def __init__(
        self,
        extractor: HashedFeatureExtractor,
        weights: NDArray[np.float64],
        bias: float = 0.0,
        threshold: float = 0.5,
        training_ids: tuple[str, ...] = (),
    ) -> None:
        if weights.shape != (extractor.dimensions,):
            raise ValueError("weight shape does not match feature dimensions")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between zero and one")
        self.extractor = extractor
        self.weights = weights
        self.bias = float(bias)
        self.threshold = float(threshold)
        self.training_ids = training_ids

    @staticmethod
    def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))

    def decision_function(self, texts: list[str] | tuple[str, ...]) -> NDArray[np.float64]:
        return self.extractor.transform(texts) @ self.weights + self.bias

    def predict_proba(self, texts: list[str] | tuple[str, ...]) -> NDArray[np.float64]:
        return self._sigmoid(self.decision_function(texts))

    def score(self, text: str) -> float:
        return float(self.predict_proba([text])[0])

    def is_malicious(self, text: str) -> bool:
        return self.score(text) >= self.threshold

    @classmethod
    def fit(
        cls,
        texts: list[str] | tuple[str, ...],
        labels: list[int] | tuple[int, ...],
        *,
        training_ids: tuple[str, ...] = (),
        dimensions: int = 4096,
        epochs: int = 500,
        learning_rate: float = 0.03,
        l2: float = 0.0001,
    ) -> tuple[InjectionClassifier, TrainingHistory]:
        if len(texts) != len(labels) or not texts:
            raise ValueError("texts and labels must have the same non-zero length")
        if epochs < 1 or learning_rate <= 0 or l2 < 0:
            raise ValueError("invalid optimization hyperparameters")
        targets = np.asarray(labels, dtype=np.float64)
        if not np.all((targets == 0.0) | (targets == 1.0)):
            raise ValueError("labels must be binary")
        if len(set(targets.tolist())) != 2:
            raise ValueError("training requires both classes")

        extractor = HashedFeatureExtractor(dimensions)
        matrix = extractor.transform(texts)
        weights = np.zeros(dimensions, dtype=np.float64)
        bias = 0.0
        first_moment = np.zeros(dimensions, dtype=np.float64)
        second_moment = np.zeros(dimensions, dtype=np.float64)
        bias_first_moment = 0.0
        bias_second_moment = 0.0

        positives = max(float(targets.sum()), 1.0)
        negatives = max(float(len(targets) - targets.sum()), 1.0)
        sample_weights = np.where(targets == 1.0, 0.5 / positives, 0.5 / negatives)
        sample_weights *= len(targets)

        def loss(current_weights: NDArray[np.float64], current_bias: float) -> float:
            logits = matrix @ current_weights + current_bias
            per_example = np.maximum(logits, 0.0) - logits * targets + np.log1p(
                np.exp(-np.abs(logits))
            )
            regularization = 0.5 * l2 * current_weights @ current_weights
            return float(np.mean(sample_weights * per_example) + regularization)

        initial_loss = loss(weights, bias)
        for step in range(1, epochs + 1):
            probabilities = cls._sigmoid(matrix @ weights + bias)
            residual = sample_weights * (probabilities - targets) / len(targets)
            gradient = matrix.T @ residual + l2 * weights
            bias_gradient = float(residual.sum())

            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
            bias_first_moment = 0.9 * bias_first_moment + 0.1 * bias_gradient
            bias_second_moment = 0.999 * bias_second_moment + 0.001 * bias_gradient**2
            corrected_first = first_moment / (1.0 - 0.9**step)
            corrected_second = second_moment / (1.0 - 0.999**step)
            weights -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            bias -= (
                learning_rate
                * (bias_first_moment / (1.0 - 0.9**step))
                / (np.sqrt(bias_second_moment / (1.0 - 0.999**step)) + 1e-8)
            )

        model = cls(extractor, weights, bias, training_ids=training_ids)
        return model, TrainingHistory(initial_loss, loss(weights, bias), epochs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "model_type": "logistic_regression",
            "dimensions": self.extractor.dimensions,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "threshold": self.threshold,
            "training_ids": list(self.training_ids),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> InjectionClassifier:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format_version") != 1 or value.get("model_type") != "logistic_regression":
            raise ValueError("unsupported classifier artifact")
        extractor = HashedFeatureExtractor(int(value["dimensions"]))
        return cls(
            extractor=extractor,
            weights=np.asarray(value["weights"], dtype=np.float64),
            bias=float(value["bias"]),
            threshold=float(value["threshold"]),
            training_ids=tuple(value.get("training_ids", [])),
        )
