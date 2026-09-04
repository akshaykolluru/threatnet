from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:  # FAISS is optional so the local demo remains easy to run.
    import faiss  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised when FAISS is installed.
    faiss = None


EMBEDDING_DIMENSION = 512
DEFAULT_INTERPRETATION = "Similarity is an investigative indicator and requires human verification. It is not proof of identity."


@dataclass(frozen=True)
class Match:
    """A ranked vector-search candidate returned to an investigator."""

    item_id: str
    similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)
    interpretation: str = DEFAULT_INTERPRETATION


class FaceVectorIndex:
    """Persistent 512-d cosine index backed by FAISS when available, otherwise NumPy.

    SQL stores the provenance metadata. This sidecar index persists normalized vectors,
    item identifiers, and a metadata copy so that a restarted API can retrieve frames
    without keeping model output in process memory.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.matrix_path = self.directory / "face_embeddings.npz"
        self.metadata_path = self.directory / "face_embeddings.metadata.json"
        self.faiss_path = self.directory / "face_embeddings.faiss"

    @property
    def backend(self) -> str:
        return "faiss" if faiss is not None else "numpy"

    def upsert(self, item_id: str, vector: Sequence[float], metadata: dict[str, Any] | None = None) -> None:
        """Insert or replace one normalized embedding and persist it immediately."""

        vectors, identifiers, metadata_by_id = self._load()
        normalized = self._normalize(vector)
        if item_id in identifiers:
            vectors[identifiers.index(item_id)] = normalized
        else:
            identifiers.append(item_id)
            vectors = np.vstack((vectors, normalized.reshape(1, -1)))
        metadata_by_id[item_id] = metadata or {}
        self._persist(vectors, identifiers, metadata_by_id)

    def add(self, item_id: str, vector: Sequence[float]) -> None:
        """Backward-compatible alias for code that used the original VectorIndex API."""

        self.upsert(item_id, vector)

    def remove(self, item_ids: Iterable[str]) -> None:
        """Remove vectors when their associated evidence or case is deleted."""

        requested = set(item_ids)
        if not requested:
            return
        vectors, identifiers, metadata_by_id = self._load()
        keep_indices = [index for index, item_id in enumerate(identifiers) if item_id not in requested]
        if len(keep_indices) == len(identifiers):
            return
        kept_vectors = vectors[keep_indices] if keep_indices else np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        kept_identifiers = [identifiers[index] for index in keep_indices]
        kept_metadata = {item_id: metadata_by_id.get(item_id, {}) for item_id in kept_identifiers}
        self._persist(kept_vectors, kept_identifiers, kept_metadata)

    def remove_by_metadata(self, **criteria: Any) -> None:
        """Remove vectors belonging to a deleted evidence record or case."""

        _, identifiers, metadata_by_id = self._load()
        self.remove(
            item_id
            for item_id in identifiers
            if all(metadata_by_id.get(item_id, {}).get(key) == value for key, value in criteria.items())
        )

    def search(
        self,
        query: Sequence[float],
        limit: int = 5,
        case_id: str | None = None,
    ) -> list[Match]:
        """Return cosine-similarity candidates in descending order."""

        if limit < 1:
            return []
        vectors, identifiers, metadata_by_id = self._load()
        if not identifiers:
            return []

        normalized_query = self._normalize(query)
        if self.backend == "faiss" and case_id is None and self.faiss_path.exists():
            search_index = faiss.read_index(str(self.faiss_path))
            scores, positions = search_index.search(normalized_query.reshape(1, -1), min(limit, len(identifiers)))
            ranked = [
                (identifiers[position], float(score))
                for score, position in zip(scores[0], positions[0])
                if position >= 0
            ]
        else:
            candidate_indices = [
                index
                for index, item_id in enumerate(identifiers)
                if case_id is None or metadata_by_id.get(item_id, {}).get("case_id") == case_id
            ]
            if not candidate_indices:
                return []
            candidate_matrix = vectors[candidate_indices]
            scores = candidate_matrix @ normalized_query
            ordering = np.argsort(-scores)[:limit]
            ranked = [
                (identifiers[candidate_indices[int(position)]], float(scores[int(position)]))
                for position in ordering
            ]

        return [
            Match(
                item_id=item_id,
                similarity=round(max(-1.0, min(1.0, score)), 4),
                metadata=metadata_by_id.get(item_id, {}),
            )
            for item_id, score in ranked
        ]

    def count(self, case_id: str | None = None) -> int:
        """Return the number of persisted descriptors, optionally scoped to one case."""

        _, identifiers, metadata_by_id = self._load()
        if case_id is None:
            return len(identifiers)
        return sum(metadata_by_id.get(item_id, {}).get("case_id") == case_id for item_id in identifiers)

    def count_by_metadata(self, **criteria: Any) -> int:
        """Count vectors with an exact metadata match without exposing index internals."""

        _, identifiers, metadata_by_id = self._load()
        return sum(
            all(metadata_by_id.get(item_id, {}).get(key) == value for key, value in criteria.items())
            for item_id in identifiers
        )

    def _normalize(self, vector: Sequence[float]) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        if array.size != EMBEDDING_DIMENSION:
            raise ValueError(f"Face embeddings must have exactly {EMBEDDING_DIMENSION} dimensions")
        norm = float(np.linalg.norm(array))
        if norm <= 0:
            raise ValueError("Face embedding cannot be all zeros")
        return array / norm

    def _load(self) -> tuple[np.ndarray, list[str], dict[str, dict[str, Any]]]:
        if not self.matrix_path.exists():
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32), [], {}
        try:
            with np.load(self.matrix_path, allow_pickle=False) as persisted:
                vectors = np.asarray(persisted["vectors"], dtype=np.float32)
                identifiers = [str(value) for value in persisted["identifiers"].tolist()]
        except (KeyError, OSError, ValueError) as exc:
            raise RuntimeError("The persisted face-vector index is unreadable; rebuild it from frame evidence.") from exc

        if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIMENSION or vectors.shape[0] != len(identifiers):
            raise RuntimeError("The persisted face-vector index has an invalid shape; rebuild it from frame evidence.")
        metadata_by_id: dict[str, dict[str, Any]] = {}
        if self.metadata_path.exists():
            try:
                raw_metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                if isinstance(raw_metadata, dict):
                    metadata_by_id = {str(key): value for key, value in raw_metadata.items() if isinstance(value, dict)}
            except (json.JSONDecodeError, OSError):
                metadata_by_id = {}
        return vectors, identifiers, metadata_by_id

    def _persist(
        self,
        vectors: np.ndarray,
        identifiers: list[str],
        metadata_by_id: dict[str, dict[str, Any]],
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        matrix = np.asarray(vectors, dtype=np.float32).reshape((-1, EMBEDDING_DIMENSION))
        self._atomic_npz_write(matrix, identifiers)
        self._atomic_json_write(metadata_by_id)
        if faiss is not None:
            index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            if len(identifiers):
                index.add(matrix)
            faiss.write_index(index, str(self.faiss_path))

    def _atomic_npz_write(self, vectors: np.ndarray, identifiers: list[str]) -> None:
        with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".npz", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, vectors=vectors, identifiers=np.asarray(identifiers, dtype=np.str_))
        os.replace(temporary_path, self.matrix_path)

    def _atomic_json_write(self, metadata_by_id: dict[str, dict[str, Any]]) -> None:
        temporary_path = self.metadata_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(metadata_by_id, sort_keys=True), encoding="utf-8")
        os.replace(temporary_path, self.metadata_path)


class VectorIndex(FaceVectorIndex):
    """Compatibility wrapper around the durable V2 implementation."""

    def __init__(self) -> None:
        super().__init__(Path(tempfile.gettempdir()) / "threatnet-legacy-vector-index")
