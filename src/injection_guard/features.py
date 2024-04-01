"""Deterministic hashed features for untrusted tool output."""

from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np
from numpy.typing import NDArray

_TOKEN = re.compile(r"[a-z0-9]+|https?|[:/@._?=&%-]")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if unicodedata.category(character) != "Cf")


class HashedFeatureExtractor:
    def __init__(self, dimensions: int = 4096) -> None:
        if dimensions < 64:
            raise ValueError("dimensions must be at least 64")
        self.dimensions = dimensions

    def _index_and_sign(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dimensions, 1.0 if value & (1 << 63) else -1.0

    def transform_one(self, text: str) -> NDArray[np.float64]:
        normalized = normalize_text(text)
        tokens = _TOKEN.findall(normalized)
        features: list[str] = [f"w:{token}" for token in tokens]
        features.extend(
            f"b:{left} {right}" for left, right in zip(tokens, tokens[1:], strict=False)
        )
        padded = f"  {normalized}  "
        for size in (3, 4, 5):
            features.extend(
                f"c:{padded[index:index + size]}"
                for index in range(max(0, len(padded) - size + 1))
            )
        features.extend(
            (
                f"meta:length-{min(len(normalized) // 40, 20)}",
                f"meta:urls-{min(normalized.count('http'), 5)}",
                f"meta:exclamations-{min(normalized.count('!'), 5)}",
            )
        )

        vector = np.zeros(self.dimensions, dtype=np.float64)
        for feature in features:
            index, sign = self._index_and_sign(feature)
            vector[index] += sign
        np.log1p(np.abs(vector), out=vector)
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector

    def transform(self, texts: list[str] | tuple[str, ...]) -> NDArray[np.float64]:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float64)
        return np.stack([self.transform_one(text) for text in texts])
