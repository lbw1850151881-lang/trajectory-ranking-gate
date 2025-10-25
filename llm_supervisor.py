"""
LLM Supervisor for dynamic gating.

This module orchestrates semantic analysis, override recommendation,
and optional trajectory verification. It relies on ``LLMFeatureExtractor``
for the heavy lifting (either remote LLM calls or deterministic heuristics)
and adds project-specific decision logic.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from llm_feature_extractor import LLMFeatureExtractor, SemanticFeatures


@dataclass
class LLMDecision:
    """Container for the supervisor's recommendation."""

    recommended_model: str
    risk_level: str
    risk_score: float
    confidence: float
    semantic_intent: str
    reasoning: str
    semantic_features: Dict[str, Any]
    override: bool
    trigger_reason: str
    hybrid_score: float
    advantage_estimate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """Results of the optional predict-verify-correct step."""

    risk_level: str
    risk_score: float
    messages: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMSupervisor:
    """
    High level supervisor coordinating semantic overrides.

    Parameters
    ----------
    extractor:
        Optional pre-built ``LLMFeatureExtractor`` instance. When omitted,
        we create a default one that honours the environment variables.
    risk_override_threshold:
        Minimum risk score required for the supervisor to consider an
        override suggestion.
    confidence_threshold:
        Minimum extraction confidence before trusting the recommendation.
    trigger_config:
        Dict defining additional trigger conditions beyond ranking score.
        Keys supported: ``min_neighbors``, ``max_min_distance``,
        ``require_crosswalk``.
    """

    def __init__(
        self,
        extractor: Optional[LLMFeatureExtractor] = None,
        *,
        risk_override_threshold: float = 0.7,
        confidence_threshold: float = 0.7,
        trigger_config: Optional[Dict[str, Any]] = None,
        sc_hybrid_threshold: float = 0.45,
    ) -> None:
        self.extractor = extractor or LLMFeatureExtractor()
        self.risk_override_threshold = risk_override_threshold
        self.confidence_threshold = confidence_threshold
        self.trigger_config = trigger_config or {}
        self.sc_hybrid_threshold = sc_hybrid_threshold
        self.scene_bonus = {
            "cut_in": 0.1,
            "high_speed": 0.05,
        }

    # ------------------------------------------------------------------
    # Semantic analysis and override recommendation
    # ------------------------------------------------------------------

    def analyze_scene(
        self,
        scene_context: Dict[str, Any],
        *,
        base_decision: str,
        trigger_reason: str = "confidence",
        semantic_features: Optional[SemanticFeatures] = None,
    ) -> LLMDecision:
        """
        Analyse a scene and return an override recommendation.
        """
        semantic = semantic_features or self.extractor.extract_features(scene_context)
        semantic_intent = semantic.scene_type or "other"

        recommended = semantic.recommended_model or (
            "gameformer" if semantic.risk_score >= 0.5 else "lstm"
        )

        advantage_estimate = float(scene_context.get("sc_advantage_estimate", 0.0))
        hybrid_score = self._compute_hybrid_score(
            semantic,
            scene_context,
            advantage_estimate,
        )
        override = self._should_override(
            base_decision,
            recommended,
            semantic,
            scene_context,
            hybrid_score,
        )

        return LLMDecision(
            recommended_model=recommended,
            risk_level=semantic.risk_level,
            risk_score=semantic.risk_score,
            confidence=semantic.extraction_confidence,
            semantic_intent=semantic_intent,
            reasoning=semantic.reasoning,
            semantic_features=semantic.to_dict(),
            override=override,
            trigger_reason=trigger_reason,
            hybrid_score=hybrid_score,
            advantage_estimate=advantage_estimate,
        )

    def _should_override(
        self,
        base_decision: str,
        recommended: str,
        semantic: SemanticFeatures,
        scene_context: Dict[str, Any],
        hybrid_score: float,
    ) -> bool:
        """
        Decide whether to override the base gate decision.

        Current policy: override when the LLM clearly prefers GameFormer in
        high-risk semantic contexts or when it strongly favours LSTM in
        simple scenes with high confidence.
        """
        if recommended == base_decision:
            return False

        if semantic.extraction_confidence < self.confidence_threshold:
            if recommended != "scene_conditioned":
                return False
            if semantic.risk_score < max(self.risk_override_threshold, 0.65):
                return False

        if recommended == "scene_conditioned":
            high_complexity_scene = semantic.scene_type in {"intersection", "congestion", "occlusion"}
            # Allow overrides for baseline GameFormer plus high-risk or targeted long-tail scenes
            if base_decision != "gameformer" and semantic.scene_type not in {"cut_in", "high_speed"}:
                return False

            bonus = self.scene_bonus.get(semantic.scene_type, 0.0)
            threshold = self.sc_hybrid_threshold - bonus
            if semantic.scene_type in {"cut_in", "high_speed"}:
                advantage = float(scene_context.get("sc_advantage_estimate", 0.0))
                advantage_lstm = float(scene_context.get("sc_vs_lstm_advantage", 0.0))
                if advantage > 0.1 and advantage_lstm > 0.1 and hybrid_score >= threshold:
                    return True
                return False

            if not high_complexity_scene:
                return False

            if float(scene_context.get("sc_advantage_estimate", 0.0)) <= 0.0:
                return False
            if float(scene_context.get("sc_vs_lstm_advantage", 0.0)) <= 0.0:
                return False

            return hybrid_score >= threshold

        if recommended == "gameformer":
            return semantic.risk_score >= self.risk_override_threshold

        # For switching to LSTM we require both low risk and explicit signal.
        return (
            semantic.risk_score <= 0.35
            and semantic.scene_type in {"other", "high_speed"}
        )

    # ------------------------------------------------------------------
    # Trigger helpers
    # ------------------------------------------------------------------

    def should_trigger_llm(
        self,
        ranking_confidence: float,
        scene_context: Dict[str, Any],
        trigger_threshold: float,
    ) -> bool:
        """
        Decide whether to invoke the LLM.
        """
        if ranking_confidence < trigger_threshold:
            return True

        if scene_context.get("force_llm"):
            return True

        neighbors = scene_context.get("n_neighbors", 0)
        min_distance = scene_context.get("min_neighbor_dist", 1e9) or 1e9
        crosswalks = scene_context.get("n_crosswalks", 0)

        if (
            self.trigger_config.get("min_neighbors", 0)
            and neighbors >= self.trigger_config["min_neighbors"]
        ):
            if min_distance <= self.trigger_config.get("max_min_distance", 5.0):
                return True

        if (
            self.trigger_config.get("require_crosswalk")
            and crosswalks >= 1
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # Verification step
    # ------------------------------------------------------------------

    def verify_prediction(
        self,
        prediction_stats: Dict[str, Any],
    ) -> VerificationResult:
        """
        Lightweight risk checks on produced trajectories.

        ``prediction_stats`` should summarise the trajectory (e.g. max
        acceleration). The method is intentionally simple to keep it cheap
        during evaluation runs.
        """
        messages = []
        risk_score = 0.0

        accel = prediction_stats.get("max_acceleration")
        if accel is not None:
            if accel > 8.0:
                messages.append(f"Acceleration {accel:.2f} m/s^2 exceeds safe limit.")
                risk_score += 0.4
            elif accel > 6.0:
                messages.append(
                    f"Acceleration {accel:.2f} m/s^2 nearing critical threshold."
                )
                risk_score += 0.2

        min_dist = prediction_stats.get("min_distance")
        if min_dist is not None:
            if min_dist < 2.0:
                messages.append("Predicted path gets closer than 2m to another agent.")
                risk_score += 0.4
            elif min_dist < 4.0:
                risk_score += 0.2

        curvature = prediction_stats.get("max_curvature")
        if curvature is not None and curvature > 0.5:
            messages.append("Sharp curvature detected in planned trajectory.")
            risk_score += 0.2

        risk_score = min(1.0, risk_score)
        risk_level = self._risk_level_from_score(risk_score)

        if not messages:
            messages.append("All verification checks passed.")

        return VerificationResult(
            risk_level=risk_level,
            risk_score=risk_score,
            messages=messages,
        )

    def _compute_hybrid_score(
        self,
        semantic: SemanticFeatures,
        scene_context: Dict[str, Any],
        advantage_estimate: float,
    ) -> float:
        risk = semantic.risk_score
        uncertainty = float(scene_context.get("gmf_uncertainty_total", 0.0))
        uncertainty_norm = max(0.0, min(1.0, uncertainty / 5.0))
        advantage_norm = max(0.0, min(1.0, advantage_estimate / 5.0))
        hybrid = 0.5 * risk + 0.3 * uncertainty_norm + 0.2 * advantage_norm
        return hybrid

    @staticmethod
    def _risk_level_from_score(score: float) -> str:
        if score >= 0.75:
            return "critical"
        if score >= 0.55:
            return "high"
        if score >= 0.35:
            return "medium"
        if score >= 0.15:
            return "low"
        return "none"


# ---------------------------------------------------------------------------
# Utility helpers used by tests
# ---------------------------------------------------------------------------


def build_dummy_scene(idx: int = 0) -> Dict[str, Any]:
    """
    Generate a deterministic pseudo-scene dictionary used in tests.
    """
    neighbors = (idx % 6) + 1
    crosswalks = 1 if idx % 3 == 0 else 0
    min_dist = max(2.0, 10.0 - idx)
    complexity = 0.3 + 0.05 * idx
    speed_var = 1.0 + 0.4 * (idx % 5)

    return {
        "sample_idx": idx,
        "complexity_score": complexity,
        "n_neighbors": neighbors,
        "speed_variance": speed_var,
        "min_neighbor_dist": min_dist,
        "n_crosswalks": crosswalks,
    }


def batch_build_dummy_scenes(count: int) -> list[Dict[str, Any]]:
    return [build_dummy_scene(i) for i in range(count)]


__all__ = [
    "LLMSupervisor",
    "LLMDecision",
    "VerificationResult",
    "build_dummy_scene",
    "batch_build_dummy_scenes",
]
