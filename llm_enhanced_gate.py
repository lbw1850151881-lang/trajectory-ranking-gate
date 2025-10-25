"""
LLM-Enhanced Gating pipeline.

This module augments the learned ranking gate with semantic supervision
provided by :class:`LLMSupervisor`. It keeps the implementation lightweight
so that evaluations can run offline (no LLM) or online (with API access).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from tri_expert_model import TriExpertModel, EXPERT_LABELS
from llm_supervisor import LLMDecision, LLMSupervisor, VerificationResult


@dataclass
class SampleDecision:
    """Detailed record for a single scene."""

    sample_idx: int
    ranking_score: float
    base_decision: str
    final_decision: str
    ranking_confidence: float
    llm_triggered: bool
    llm_decision: Optional[Dict[str, Any]]
    verification: Optional[Dict[str, Any]]
    base_probabilities: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Ensure nested dataclasses are serialised.
        if self.llm_decision and isinstance(self.llm_decision, LLMDecision):
            payload["llm_decision"] = self.llm_decision.to_dict()
        if self.verification and isinstance(self.verification, VerificationResult):
            payload["verification"] = self.verification.to_dict()
        return payload


class LLMEnhancedGate:
    """
    Combines ranking-gate scores with LLM supervision according to the
    decision rules documented in the experimental log.
    """

    def __init__(
        self,
        checkpoint: Dict[str, Any],
        *,
        supervisor: Optional[LLMSupervisor] = None,
        llm_enabled: bool = True,
        base_threshold: float = -0.25,
        trigger_threshold: float = 0.3,
        llm_override: bool = True,
        verification_enabled: bool = False,
        device: str = "cpu",
        min_lstm_prob: float = 0.8,
        min_scene_prob: float = 0.4,
    ) -> None:
        self.device = torch.device(device)
        self.min_lstm_prob = min_lstm_prob
        self.min_scene_prob = min_scene_prob
        self.trigger_threshold = trigger_threshold
        self.llm_enabled = llm_enabled
        self.llm_override = llm_override
        self.verification_enabled = verification_enabled

        input_dim = checkpoint.get("input_dim")
        hidden_dims = checkpoint.get("hidden_dims", [128, 64, 32])
        dropout = checkpoint.get("dropout", 0.3)

        if input_dim is None:
            raise ValueError("Tri-expert checkpoint missing 'input_dim'.")

        self.model = TriExpertModel(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        normalization = checkpoint.get("normalization", {})
        self.feature_mean = torch.tensor(
            normalization.get("mean", np.zeros(input_dim)),
            dtype=torch.float32,
            device=self.device,
        )
        self.feature_std = torch.tensor(
            normalization.get("std", np.ones(input_dim)),
            dtype=torch.float32,
            device=self.device,
        )
        self.feature_keys = normalization.get("feature_keys", [])

        self.supervisor = supervisor or LLMSupervisor()

    # ------------------------------------------------------------------
    # Core inference helpers
    # ------------------------------------------------------------------

    def predict_probabilities(self, feature_vector: np.ndarray) -> np.ndarray:
        feature_tensor = torch.tensor(feature_vector, dtype=torch.float32, device=self.device)
        feature_tensor = (feature_tensor - self.feature_mean) / (self.feature_std + 1e-8)
        with torch.no_grad():
            logits = self.model(feature_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs

    def decide(
        self,
        sample_idx: int,
        feature_vector: np.ndarray,
        scene_context: Dict[str, Any],
        *,
        prediction_stats: Optional[Dict[str, Any]] = None,
    ) -> SampleDecision:
        """
        Combine ranking gate and supervisor decisions for a single sample.
        """
        probs = self.predict_probabilities(feature_vector)
        base_idx = int(np.argmax(probs))
        base_decision = EXPERT_LABELS[base_idx]
        base_prob = float(probs[base_idx])

        if base_decision == "lstm" and base_prob < self.min_lstm_prob:
            base_decision = "gameformer"
            base_prob = float(probs[EXPERT_LABELS.index("gameformer")])
        if base_decision == "scene_conditioned" and base_prob < self.min_scene_prob:
            base_decision = "gameformer"
            base_prob = float(probs[EXPERT_LABELS.index("gameformer")])

        ranking_conf = 1.0 - base_prob

        scene_advantage = float(
            scene_context.get("gameformer_fde", 0.0)
            - scene_context.get("scene_conditioned_fde", scene_context.get("gameformer_fde", 0.0))
        )
        scene_context = dict(scene_context)
        scene_context["sc_advantage_estimate"] = scene_advantage
        scene_context["sc_vs_lstm_advantage"] = float(
            scene_context.get("lstm_fde", scene_context.get("scene_conditioned_fde", 0.0))
            - scene_context.get("scene_conditioned_fde", scene_context.get("lstm_fde", 0.0))
        )

        scene_type = scene_context.get("scene_conditioned_scene_type")
        if scene_type in {"cut_in", "high_speed"}:
            scene_context["force_llm"] = True

        llm_triggered = False
        if self.llm_enabled:
            llm_triggered = self.supervisor.should_trigger_llm(
                ranking_confidence=ranking_conf,
                scene_context=scene_context,
                trigger_threshold=self.trigger_threshold,
            )

        llm_decision: Optional[LLMDecision] = None
        verification: Optional[VerificationResult] = None
        final_decision = base_decision

        if llm_triggered:
            trigger_reason = "confidence" if ranking_conf < self.trigger_threshold else "semantic"
            llm_decision = self.supervisor.analyze_scene(
                scene_context,
                base_decision=base_decision,
                trigger_reason=trigger_reason,
            )
            if self.llm_override and llm_decision.override:
                final_decision = llm_decision.recommended_model

        if self.verification_enabled:
            verification = self.supervisor.verify_prediction(prediction_stats or {})
            if (
                verification.risk_level in {"critical", "high"}
                and final_decision != "lstm"
            ):
                final_decision = "lstm"

        return SampleDecision(
            sample_idx=sample_idx,
            ranking_score=base_prob,
            base_decision=base_decision,
            final_decision=final_decision,
            ranking_confidence=ranking_conf,
            llm_triggered=llm_triggered,
            llm_decision=llm_decision.to_dict() if llm_decision else None,
            verification=verification.to_dict() if verification else None,
            base_probabilities={label: float(prob) for label, prob in zip(EXPERT_LABELS, probs)},
        )

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        feature_matrix: np.ndarray,
        scene_contexts: List[Dict[str, Any]],
        *,
        prediction_stats: Optional[List[Optional[Dict[str, Any]]]] = None,
        sample_indices: Optional[np.ndarray] = None,
    ) -> List[SampleDecision]:
        """
        Evaluate multiple samples at once.
        """
        decisions: List[SampleDecision] = []
        prediction_stats = prediction_stats or [None] * len(scene_contexts)

        for idx, (features, context, stats) in enumerate(
            zip(feature_matrix, scene_contexts, prediction_stats)
        ):
            sample_id = int(sample_indices[idx]) if sample_indices is not None else idx
            decisions.append(
                self.decide(
                    sample_id,
                    features,
                    context,
                    prediction_stats=stats,
                )
            )
        return decisions


def load_tri_checkpoint(path: str, device: str = "cpu") -> Dict[str, Any]:
    load_kwargs = {"map_location": device}
    try:
        checkpoint = torch.load(path, weights_only=False, **load_kwargs)
    except TypeError:
        # Fallback for older torch versions that do not accept weights_only.
        checkpoint = torch.load(path, **load_kwargs)
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Tri-expert checkpoint missing 'model_state_dict'. "
            "Run gate_tri_classifier.py to generate it."
        )
    return checkpoint


__all__ = ["LLMEnhancedGate", "SampleDecision", "load_tri_checkpoint"]
