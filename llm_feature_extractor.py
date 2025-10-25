"""
LLM semantic feature extraction utilities.

This module provides a thin wrapper around either a remote LLM
service (Google Generative AI) or a deterministic heuristic fallback
so we can obtain structured semantic annotations for trajectory
prediction scenes without hard dependencies on network access.

The output schema is aligned with the rest of the project – especially
the calibrated override workflow – via the ``SemanticFeatures`` dataclass.
"""

from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional


try:  # Optional dependency – only needed when we really call the API.
    import google.generativeai as genai  # type: ignore
except ImportError:  # pragma: no cover - handled via offline fallback.
    genai = None  # type: ignore


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SemanticFeatures:
    """
    Structured semantic description for a traffic scene.

    The field names mirror the expectations inside ``override_calibrator``
    and documentation captured in the project reports.
    """

    scene_type: str = "other"
    scene_subtype: str = "unspecified"
    risk_level: str = "low"
    risk_score: float = 0.2
    conflict_potential: float = 0.1
    n_active_agents: int = 0
    has_occlusion: bool = False
    has_yielding: bool = False
    has_pedestrian: bool = False
    has_sudden_maneuver: bool = False
    semantic_keywords: list[str] = field(default_factory=list)
    interaction_topology: str = "sparse"  # sparse | moderate | dense
    speed_variation: str = "low"          # low | medium | high
    extraction_confidence: float = 0.5
    recommended_model: Optional[str] = None
    reasoning: str = ""

    @property
    def risk_category(self) -> str:
        """Alias maintained for backwards compatibility."""
        return self.risk_level

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _hash_context(scene_context: Dict[str, Any]) -> str:
    """Create a deterministic cache key for a scene context."""
    serialized = json.dumps(scene_context, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def _estimate_risk_level(score: float) -> str:
    """Convert a numeric risk score into a qualitative label."""
    if score >= 0.75:
        return "critical"
    if score >= 0.55:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


# ---------------------------------------------------------------------------
# LLM Feature Extractor
# ---------------------------------------------------------------------------


class LLMFeatureExtractor:
    """
    Extract semantic features for a scene via an LLM or heuristics.

    Parameters
    ----------
    api_key:
        Google Generative AI key. If omitted, we attempt to read the
        ``GOOGLE_API_KEY`` environment variable. When no key is available
        we transparently fall back to the heuristic mode.
    model:
        Remote model name. Defaults to ``gemini-1.5-flash`` which offers a
        good latency/price trade-off for semantic analysis.
    cache_path:
        Optional JSON file used to persist query results. Re-using cached
        responses dramatically reduces network usage during large eval runs.
    offline:
        Force heuristic mode regardless of key availability. Useful for tests.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        cache_path: Optional[str] = "eval_out/llm_semantic_cache.json",
        offline: Optional[bool] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, Dict[str, Any]] = {}

        self.offline = offline if offline is not None else not (
            self.api_key and genai is not None
        )

        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text())
            except json.JSONDecodeError:
                self.cache = {}

        if not self.offline and genai is not None and self.api_key:
            genai.configure(api_key=self.api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(
        self,
        scene_context: Dict[str, Any],
        *,
        force_refresh: bool = False,
    ) -> SemanticFeatures:
        """
        Return semantic descriptors for a scene.

        The method first checks the on-disk cache, then decides between
        heuristic or remote inference depending on the operating mode.
        """
        cache_key = _hash_context(scene_context)
        if not force_refresh and cache_key in self.cache:
            return SemanticFeatures(**self.cache[cache_key])

        if self.offline or genai is None or not self.api_key:
            features = self._heuristic_features(scene_context)
        else:
            try:
                features = self._call_remote_model(scene_context)
            except Exception:
                features = self._heuristic_features(scene_context)

        if self.cache_path:
            self.cache[cache_key] = features.to_dict()
            self._persist_cache()

        return features

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _heuristic_features(self, scene_context: Dict[str, Any]) -> SemanticFeatures:
        """
        A deterministic rule-based approximation used when the real LLM
        is unavailable. The rules cover the most common scenarios observed
        in the project (e.g. intersections, cut-ins, congestion).
        """
        neighbors = float(scene_context.get("n_neighbors", 0))
        min_dist = float(scene_context.get("min_neighbor_dist", 999.0) or 999.0)
        crosswalks = float(scene_context.get("n_crosswalks", 0))
        complexity = float(scene_context.get("complexity_score", 0.0))
        speed_var = float(scene_context.get("speed_variance", 0.0))

        scene_type = "other"
        keywords: list[str] = []
        has_yielding = False
        has_occlusion = bool(scene_context.get("has_occlusion", False))

        if crosswalks >= 1:
            scene_type = "intersection"
            keywords.extend(["intersection", "crosswalk"])
            has_yielding = True
        elif neighbors >= 4 and min_dist < 8.0:
            scene_type = "congestion"
            keywords.extend(["congestion", "dense_traffic"])
        elif neighbors >= 2 and min_dist < 5.0:
            scene_type = "cut_in"
            keywords.extend(["lane_change", "cut_in"])
        elif speed_var > 3.0:
            scene_type = "high_speed"
            keywords.append("high_speed")

        if "intersection" in keywords and neighbors >= 3:
            keywords.append("multi_agent_interaction")

        interaction_topology = "sparse"
        if neighbors >= 6:
            interaction_topology = "dense"
        elif neighbors >= 3:
            interaction_topology = "moderate"

        if speed_var >= 4.0:
            speed_variation = "high"
        elif speed_var >= 1.5:
            speed_variation = "medium"
        else:
            speed_variation = "low"

        # Risk score emphasises proximity + complexity + crosswalk presence.
        risk_score = (
            0.4 * _clamp(complexity, 0.0, 1.0)
            + 0.3 * _clamp(1.0 - min_dist / 20.0, 0.0, 1.0)
            + 0.2 * _clamp(neighbors / 6.0, 0.0, 1.0)
            + 0.1 * (1.0 if crosswalks > 0 else 0.0)
        )
        risk_score = _clamp(risk_score, 0.0, 1.0)

        recommended_model = "lstm"
        if risk_score >= 0.5:
            recommended_model = "gameformer"
        if (
            risk_score >= 0.6
            or (scene_type in {"intersection", "congestion", "occlusion"} and risk_score >= 0.5)
        ):
            recommended_model = "scene_conditioned"
        reasoning = (
            f"Rule-based heuristic (offline mode). "
            f"scene_type={scene_type}, neighbors={neighbors}, "
            f"min_dist={min_dist:.1f}, complexity={complexity:.2f}"
        )

        return SemanticFeatures(
            scene_type=scene_type,
            scene_subtype="heuristic",
            risk_level=_estimate_risk_level(risk_score),
            risk_score=risk_score,
            conflict_potential=_clamp(neighbors / 8.0 + complexity * 0.3, 0.0, 1.0),
            n_active_agents=int(neighbors),
            has_occlusion=has_occlusion,
            has_yielding=has_yielding,
            has_pedestrian=bool(scene_context.get("has_pedestrian", False)),
            has_sudden_maneuver=min_dist < 4.0,
            semantic_keywords=keywords,
            interaction_topology=interaction_topology,
            speed_variation=speed_variation,
            extraction_confidence=0.6 if keywords else 0.5,
            recommended_model=recommended_model,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Remote model integration
    # ------------------------------------------------------------------

    def _call_remote_model(self, scene_context: Dict[str, Any]) -> SemanticFeatures:
        """
        Query the configured LLM for semantic annotations.

        The model is prompted to respond with a JSON payload compatible
        with :class:`SemanticFeatures`. When parsing fails we fall back to
        the heuristic implementation, ensuring deterministic behaviour.
        """
        if genai is None or not self.api_key:  # pragma: no cover - guarded earlier.
            return self._heuristic_features(scene_context)

        prompt = self._build_prompt(scene_context)
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(prompt)
        text = response.text if hasattr(response, "text") else str(response)

        try:
            payload = json.loads(text)
            features = SemanticFeatures(**payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            features = self._heuristic_features(scene_context)

        if not features.reasoning:
            features.reasoning = "LLM generated result"
        if features.recommended_model is None:
            features.recommended_model = (
                "gameformer" if features.risk_score >= 0.5 else "lstm"
            )
        return features

    @staticmethod
    def _build_prompt(scene_context: Dict[str, Any]) -> str:
        """
        Build a prompt requesting a strict JSON response.
        """
        return (
            "You are assisting with autonomous driving trajectory prediction.\n"
            "Analyse the scenario described below and respond with a JSON object "
            "containing the following keys:\n"
            + json.dumps(list(SemanticFeatures().__dict__.keys()))
            + "\n\nScenario context:\n"
            + json.dumps(scene_context, indent=2, ensure_ascii=False)
            + "\n\nRespond with JSON only."
        )
