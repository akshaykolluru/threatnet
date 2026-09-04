"""Cached multimodal retrieval over sampled CCTV frames.

The optional OpenCLIP dependency gives text and images a shared 512-dimensional
embedding space.  Frame vectors are generated once at ingestion and persisted in
the existing NumPy/FAISS sidecar index; a later query never re-runs inference over
the video frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .vector_search import EMBEDDING_DIMENSION, FaceVectorIndex


# Cosine similarities from one video are comparable to each other but are not
# calibrated probabilities. Keep only the near-best band instead of displaying
# every frame that happens to be rankable as a supposed match.
NEAR_BEST_COSINE_WINDOW = 0.006
MAX_REVIEW_CANDIDATES = 6


class FrameSearchEmbedder:
    """Lazy CPU OpenCLIP wrapper so ordinary CCTV ingestion remains available."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._attempted = False

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    def _load(self) -> None:
        if self._attempted:
            return
        self._attempted = True
        try:
            import open_clip  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"
            )
            model.eval()
            self._model, self._preprocess, self._tokenizer, self._torch = model, preprocess, open_clip.get_tokenizer("ViT-B-32"), torch
        except Exception:
            self._model = None

    def _require_model(self) -> None:
        if not self.available:
            raise RuntimeError(
                "Semantic frame search is unavailable. Install the optional open_clip_torch dependency "
                "and ensure its pretrained model can be downloaded once."
            )

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("The model returned an empty visual representation")
        return (vector / norm).astype(np.float32)

    def image_embedding(self, image: np.ndarray) -> np.ndarray:
        self._require_model()
        if image is None or image.size == 0:
            raise ValueError("The supplied image could not be decoded")
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self._preprocess(Image.fromarray(rgb)).unsqueeze(0)
        with self._torch.no_grad():
            vector = self._model.encode_image(tensor).cpu().numpy().reshape(-1)
        if vector.size != EMBEDDING_DIMENSION:
            raise RuntimeError("The selected visual-search model returned an unexpected embedding size")
        return self._normalize(vector)

    def image_bytes_embedding(self, content: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        return self.image_embedding(image)

    def text_embedding(self, prompt: str) -> np.ndarray:
        self._require_model()
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("Enter a search prompt or upload a reference image")
        tokens = self._tokenizer([clean_prompt])
        with self._torch.no_grad():
            vector = self._model.encode_text(tokens).cpu().numpy().reshape(-1)
        if vector.size != EMBEDDING_DIMENSION:
            raise RuntimeError("The selected visual-search model returned an unexpected embedding size")
        return self._normalize(vector)


_embedder: FrameSearchEmbedder | None = None


def get_frame_search_embedder() -> FrameSearchEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = FrameSearchEmbedder()
    return _embedder


def frame_index(directory: Path | str) -> FaceVectorIndex:
    """Use the project's durable cosine index in a distinct frame-search namespace."""

    return FaceVectorIndex(Path(directory) / "frame-search-index")


def index_frames(index: FaceVectorIndex, frames: Sequence[dict[str, Any]], case_id: str, evidence_id: str) -> int:
    """Embed sampled frames once and retain only reviewable provenance in index metadata."""

    embedder = get_frame_search_embedder()
    if not embedder.available:
        return 0
    indexed = 0
    for frame in frames:
        frame_path = Path(str(frame["path"]))
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        vector = embedder.image_embedding(image)
        item_id = f"FRAME-{evidence_id}-{int(frame['frame_index'])}"
        index.upsert(
            item_id,
            vector,
            {
                "case_id": case_id,
                "evidence_id": evidence_id,
                "frame_path": str(frame_path),
                "frame_index": int(frame["frame_index"]),
                "timestamp_seconds": float(frame["timestamp_seconds"]),
            },
        )
        indexed += 1
    return indexed


def search_frames(
    index: FaceVectorIndex,
    case_id: str,
    prompt: str = "",
    reference_image: bytes | None = None,
    evidence_id: str | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Rank cached frame embeddings by text, image, or their equally weighted blend."""

    if not prompt.strip() and not reference_image:
        raise ValueError("Enter a search prompt or upload a reference image")
    embedder = get_frame_search_embedder()
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    if prompt.strip():
        vectors.append(embedder.text_embedding(prompt))
        labels.append("text prompt")
    if reference_image:
        vectors.append(embedder.image_bytes_embedding(reference_image))
        labels.append("reference image")
    # All vectors are normalized. Their mean is normalized again so a combined
    # request assigns equal contribution to each available modality.
    query = np.mean(np.vstack(vectors), axis=0)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    candidates = index.search(query, limit=max(1, min(limit * 4, 100)), case_id=case_id)
    ranked_results: list[dict[str, Any]] = []
    for candidate in candidates:
        metadata = candidate.metadata
        if evidence_id and metadata.get("evidence_id") != evidence_id:
            continue
        ranked_results.append(
            {
                "frame_path": metadata.get("frame_path", ""),
                "frame_index": metadata.get("frame_index"),
                "timestamp_seconds": metadata.get("timestamp_seconds"),
                "score": round(max(0.0, (candidate.similarity + 1) / 2), 4),
                "reason": f"Ranked by {' and '.join(labels)} visual similarity.",
                "evidence_id": metadata.get("evidence_id"),
            }
        )
    if not ranked_results:
        return []
    best_cosine = (float(ranked_results[0]["score"]) * 2) - 1
    return [
        result
        for result in ranked_results
        if ((float(result["score"]) * 2) - 1) >= best_cosine - NEAR_BEST_COSINE_WINDOW
    ][: min(limit, MAX_REVIEW_CANDIDATES)]
