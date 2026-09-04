from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class Match:
    item_id: str
    similarity: float
    interpretation: str = "Similarity is an investigative indicator and requires human review."


class VectorIndex:
    """Small zero-cost fallback; swap in FAISS behind this interface when available."""

    def __init__(self) -> None:
        self._items: dict[str, list[float]] = {}

    def add(self, item_id: str, vector: list[float]) -> None:
        self._items[item_id] = vector

    def search(self, query: list[float], limit: int = 5) -> list[Match]:
        def cosine(vector: list[float]) -> float:
            numerator = sum(a * b for a, b in zip(query, vector))
            denominator = math.sqrt(sum(a * a for a in query)) * math.sqrt(sum(b * b for b in vector))
            return numerator / denominator if denominator else 0.0

        return [Match(item_id, round(cosine(vector), 4)) for item_id, vector in sorted(self._items.items(), key=lambda pair: cosine(pair[1]), reverse=True)[:limit]]
