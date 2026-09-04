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
MIN_BEST_TO_MEDIAN_SCORE = 0.01
PERSON_CONTEXT_NEAR_BEST_WINDOW = 0.025

# COCO labels that a lightweight YOLO detector can verify reliably in CCTV
# frames. Natural-language searches outside this vocabulary still use semantic
# retrieval, but these targets must be detector-confirmed.
QUERY_CLASS_ALIASES: dict[str, set[str]] = {
    "person": {"person", "people", "man", "woman", "human"},
    "backpack": {"backpack", "rucksack", "bag", "carrybag", "carry-bag"},
    "handbag": {"handbag", "purse", "bag", "carrybag", "carry-bag"},
    "suitcase": {"suitcase", "luggage", "bag"},
    "car": {"car", "vehicle", "auto", "hatchback", "sedan"},
    "motorcycle": {"motorcycle", "motorbike", "bike"},
    "bicycle": {"bicycle", "cycle", "bike"},
    "bus": {"bus"},
    "truck": {"truck", "lorry"},
    "cell phone": {"phone", "mobile"},
}
CLOTHING_COLORS: dict[str, tuple[tuple[int, int], ...]] = {
    "red": ((0, 10), (170, 180)),
    "blue": ((95, 130),),
    "green": ((38, 85),),
    "yellow": ((20, 37),),
    "black": ((0, 180),),
    "white": ((0, 180),),
    "gray": ((0, 180),),
    "grey": ((0, 180),),
}
CLOTHING_TERMS = {"shirt", "jacket", "t-shirt", "tshirt", "top", "pants", "trousers", "clothes", "wearing", "dress"}
BAG_TERMS = {"bag", "carrybag", "carry-bag", "handbag", "purse", "backpack", "rucksack", "suitcase", "luggage"}
BAG_CLASSES = {"backpack", "handbag", "suitcase"}
PERSON_TERMS = QUERY_CLASS_ALIASES["person"]


class CctvObjectVerifier:
    """Optional YOLO verifier for common, detector-supported CCTV objects."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._attempted = False
        self._cache: dict[str, list[dict[str, Any]]] = {}

    def _load(self) -> None:
        if self._attempted:
            return
        self._attempted = True
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]

            # Downloads the small pretrained checkpoint only once when absent.
            self._model = YOLO("yolo11n.pt")
        except Exception:
            self._model = None

    def supported_classes(self, prompt: str) -> set[str]:
        words = {word.strip(".,!?;:").casefold() for word in prompt.split()}
        classes = {label for label, aliases in QUERY_CLASS_ALIASES.items() if words & aliases}
        if words & CLOTHING_TERMS and words & set(CLOTHING_COLORS):
            classes.add("person")
        return classes

    def detect(self, frame_path: Path) -> list[dict[str, Any]]:
        key = str(frame_path.resolve())
        if key in self._cache:
            return self._cache[key]
        self._load()
        if self._model is None:
            return []
        try:
            prediction = self._model(str(frame_path), verbose=False)[0]
            names = prediction.names
            detections = [
                {
                    "label": str(names[int(box.cls[0])]),
                    "confidence": float(box.conf[0]),
                    "bbox": [int(value) for value in box.xyxy[0].tolist()],
                }
                for box in prediction.boxes
                if float(box.conf[0]) >= 0.35
            ]
        except Exception:
            detections = []
        self._cache[key] = detections
        return detections


_object_verifier: CctvObjectVerifier | None = None


def get_object_verifier() -> CctvObjectVerifier:
    global _object_verifier
    if _object_verifier is None:
        _object_verifier = CctvObjectVerifier()
    return _object_verifier


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

    def _text_embeddings(self, prompts: list[str]) -> np.ndarray:
        """Encode and average prompt templates in the shared visual space."""

        self._require_model()
        tokens = self._tokenizer(prompts)
        with self._torch.no_grad():
            vectors = self._model.encode_text(tokens).cpu().numpy()
        if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIMENSION:
            raise RuntimeError("The selected visual-search model returned an unexpected embedding size")
        normalized = np.vstack([self._normalize(vector) for vector in vectors])
        return self._normalize(normalized.mean(axis=0))

    def text_embedding(self, prompt: str) -> np.ndarray:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("Enter a search prompt or upload a reference image")
        # Raw search terms are weak CLIP prompts. These CCTV-specific variants
        # reduce accidental matches to generic backgrounds and lighting.
        positive = self._text_embeddings(
            [
                f"a CCTV security-camera frame showing {clean_prompt}",
                f"surveillance footage of {clean_prompt}",
            ]
        )
        negative = self._text_embeddings(
            [
                f"a CCTV security-camera frame without {clean_prompt}",
                "an unrelated empty indoor CCTV security-camera frame",
            ]
        )
        # Contrastive direction favors frames more like the requested evidence
        # than an explicitly unrelated frame; it is not an identity decision.
        return self._normalize(positive - negative)


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
    # A flat result distribution means the model has no discriminative signal.
    # Return no candidates rather than presenting arbitrary nearest frames as
    # matches. Scores are transformed cosine values solely for display.
    median_score = float(np.median([float(item["score"]) for item in ranked_results]))
    if float(ranked_results[0]["score"]) - median_score < MIN_BEST_TO_MEDIAN_SCORE:
        return []
    best_cosine = (float(ranked_results[0]["score"]) * 2) - 1
    return [
        result
        for result in ranked_results
        if ((float(result["score"]) * 2) - 1) >= best_cosine - NEAR_BEST_COSINE_WINDOW
    ][: min(limit, MAX_REVIEW_CANDIDATES)]


def detector_verified_results(frames: Sequence[dict[str, Any]], prompt: str, limit: int = MAX_REVIEW_CANDIDATES) -> list[dict[str, Any]] | None:
    """Return confirmed common-object hits, or None when the prompt is not COCO-verifiable."""

    verifier = get_object_verifier()
    target_classes = verifier.supported_classes(prompt)
    if not target_classes:
        return None
    words = {word.strip(".,!?;:").casefold() for word in prompt.split()}
    requested_colors = [color for color in CLOTHING_COLORS if color in words]
    clothing_query = bool(requested_colors and words & CLOTHING_TERMS)
    bag_query = bool(words & BAG_TERMS)
    person_query = bool(words & PERSON_TERMS)
    matches: list[dict[str, Any]] = []
    for frame in frames:
        path = Path(str(frame.get("path", "")))
        if not path.is_file():
            continue
        detections = verifier.detect(path)
        people = [detection for detection in detections if detection["label"] == "person"]
        bags = [detection for detection in detections if detection["label"] in BAG_CLASSES]
        if person_query and not people:
            continue
        if bag_query and not bags:
            continue
        if person_query and bag_query:
            # A carry bag must be spatially associated with the requested person,
            # not merely be elsewhere in a crowded frame.
            paired_bags = [bag for bag in bags if any(_boxes_are_near(person["bbox"], bag["bbox"]) for person in people)]
            if not paired_bags:
                continue
            bags = paired_bags
        matched = [detection for detection in detections if detection["label"] in target_classes]
        if bag_query:
            matched = bags
        color_score = 1.0
        if bag_query and requested_colors:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            color_score = max(_bbox_color_score(image, bag["bbox"], color) for bag in bags for color in requested_colors)
            if color_score < 0.08:
                continue
            matched = bags
        elif clothing_query:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            colored_people = []
            for detection in matched:
                if detection["label"] != "person" or image is None:
                    continue
                score = max(_upper_body_color_score(image, detection["bbox"], color) for color in requested_colors)
                if score >= 0.08:
                    colored_people.append((detection, score))
            if not colored_people:
                continue
            matched = [detection for detection, _ in colored_people]
            color_score = max(score for _, score in colored_people)
        if not matched:
            continue
        confidence = max(float(detection["confidence"]) for detection in matched)
        labels = ", ".join(sorted({str(detection["label"]) for detection in matched}))
        matches.append(
            {
                "frame_path": str(path),
                "frame_index": frame.get("frame_index"),
                "timestamp_seconds": frame.get("timestamp_seconds"),
                "score": round(confidence * color_score, 4),
                "reason": (
                    f"Detector verified: person carrying {labels} with {', '.join(requested_colors)} color."
                    if bag_query and requested_colors
                    else f"Detector verified: person carrying {labels}."
                    if bag_query and person_query
                    else f"Detector verified: {labels}" + (f" with {', '.join(requested_colors)} upper-body clothing." if clothing_query else ".")
                ),
                "detector_verified": True,
            }
        )
    return sorted(matches, key=lambda item: float(item["score"]), reverse=True)[:limit]


def is_person_carried_item_query(prompt: str) -> bool:
    """Whether a prompt needs person-centred retrieval when a small item is missed."""

    words = {word.strip(".,!?;:").casefold() for word in prompt.split()}
    return bool(words & PERSON_TERMS) and bool(words & BAG_TERMS)


def person_context_search_results(
    frames: Sequence[dict[str, Any]], prompt: str, limit: int = MAX_REVIEW_CANDIDATES
) -> list[dict[str, Any]]:
    """Rank detected people with their hand/carry area, not an entire crowded frame.

    Small shopping bags are frequently below the COCO detector's resolution.  This
    is deliberately labelled as a visual candidate rather than a detector-verified
    finding: the crop gives the text/image model a fair comparison without claiming
    it has identified a bag that the detector did not see.
    """

    verifier = get_object_verifier()
    embedder = get_frame_search_embedder()
    query_vector = embedder.text_embedding(_normalized_carried_item_prompt(prompt))
    candidates: list[dict[str, Any]] = []
    for frame in frames:
        path = Path(str(frame.get("path", "")))
        if not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        person_scores: list[float] = []
        people = [person for person in verifier.detect(path) if person["label"] == "person"]
        # Tiny pedestrians do not carry enough pixels to distinguish a bag; they
        # also make a CPU-only search needlessly slow.  The nearest/largest people
        # are the useful candidates for this evidence review.
        people.sort(
            key=lambda person: (person["bbox"][2] - person["bbox"][0]) * (person["bbox"][3] - person["bbox"][1]),
            reverse=True,
        )
        for person in people[:3]:
            crop = _person_context_crop(image, person["bbox"])
            if crop is None:
                continue
            similarity = float(np.dot(embedder.image_embedding(crop), query_vector))
            person_scores.append(similarity)
        if not person_scores:
            continue
        best_similarity = max(person_scores)
        candidates.append(
            {
                "frame_path": str(path),
                "frame_index": frame.get("frame_index"),
                "timestamp_seconds": frame.get("timestamp_seconds"),
                "score": round(max(0.0, (best_similarity + 1) / 2), 4),
                "reason": "Person-centred visual candidate; review the carried item manually.",
                "detector_verified": False,
            }
        )
    if not candidates:
        return []
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    scores = [float(item["score"]) for item in candidates]
    if scores[0] - float(np.median(scores)) < MIN_BEST_TO_MEDIAN_SCORE:
        return []
    best_cosine = scores[0] * 2 - 1
    return [
        item
        for item in candidates
        if (float(item["score"]) * 2 - 1) >= best_cosine - PERSON_CONTEXT_NEAR_BEST_WINDOW
    ][: min(limit, MAX_REVIEW_CANDIDATES)]


def _normalized_carried_item_prompt(prompt: str) -> str:
    """Give the text model the conventional spelling it was trained on."""

    return prompt.casefold().replace("carrybag", "shopping bag").replace("carry-bag", "shopping bag")


def _person_context_crop(image: np.ndarray, bbox: Sequence[int]) -> np.ndarray | None:
    """Include a person's hands and immediately adjacent carried-object area."""

    x1, y1, x2, y2 = [int(value) for value in bbox]
    height, width = image.shape[:2]
    person_width = max(1, x2 - x1)
    person_height = max(1, y2 - y1)
    # Shopping bags commonly hang at either side of the body.  A modest margin
    # preserves that evidence while excluding most of a busy pavement/street.
    left = max(0, x1 - int(person_width * 0.45))
    right = min(width, x2 + int(person_width * 0.45))
    top = max(0, y1 - int(person_height * 0.06))
    bottom = min(height, y2 + int(person_height * 0.08))
    crop = image[top:bottom, left:right]
    return crop if crop.size else None


def _upper_body_color_score(image: np.ndarray, bbox: Sequence[int], color: str) -> float:
    """Measure a requested clothing color in a detected person's torso region."""

    x1, y1, x2, y2 = [int(value) for value in bbox]
    height, width = image.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    person_height = y2 - y1
    # Head and legs are excluded; this is where a shirt/jacket is usually seen.
    torso = image[y1 + int(person_height * 0.22) : y1 + int(person_height * 0.62), x1:x2]
    if torso.size == 0:
        return 0.0
    return _color_score(torso, color)


def _bbox_color_score(image: np.ndarray, bbox: Sequence[int], color: str) -> float:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    height, width = image.shape[:2]
    crop = image[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
    return _color_score(crop, color) if crop.size else 0.0


def _color_score(image: np.ndarray, color: str) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    if color == "black":
        mask = value < 75
    elif color == "white":
        mask = (saturation < 45) & (value > 150)
    elif color in {"gray", "grey"}:
        mask = (saturation < 45) & (value >= 55) & (value <= 190)
    else:
        hue_match = np.zeros_like(hue, dtype=bool)
        for low, high in CLOTHING_COLORS[color]:
            hue_match |= (hue >= low) & (hue <= high)
        mask = hue_match & (saturation >= 70) & (value >= 55)
    return float(mask.mean())


def _boxes_are_near(person_bbox: Sequence[int], object_bbox: Sequence[int]) -> bool:
    px1, py1, px2, py2 = [float(value) for value in person_bbox]
    ox1, oy1, ox2, oy2 = [float(value) for value in object_bbox]
    person_width, person_height = max(1.0, px2 - px1), max(1.0, py2 - py1)
    object_x, object_y = (ox1 + ox2) / 2, (oy1 + oy2) / 2
    return px1 - person_width * 0.35 <= object_x <= px2 + person_width * 0.35 and py1 <= object_y <= py2 + person_height * 0.2
