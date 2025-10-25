from __future__ import annotations

from typing import Dict


BASE_FEATURE_KEYS = [
    "complexity_score",
    "n_neighbors",
    "neighbor_score",
    "speed_variance",
    "velocity_score",
    "min_neighbor_dist",
    "proximity_score",
    "n_crosswalks",
    "crosswalk_score",
    "lstm_uncertainty_total",
    "lstm_uncertainty_final",
    "lstm_stability",
    "lstm_physics_violation",
    "gmf_uncertainty_total",
    "gmf_uncertainty_final",
    "gmf_stability",
    "gmf_physics_violation",
    "uncertainty_ratio",
    "stability_ratio",
    "violation_diff",
    "stability_diff",
    "scene_conditioned_uncertainty_total",
    "scene_conditioned_uncertainty_final",
    "scene_conditioned_stability",
    "scene_conditioned_physics_violation",
    "scene_conditioned_uncertainty_ratio_lstm",
    "scene_conditioned_uncertainty_ratio_gmf",
    "scene_conditioned_stability_ratio_lstm",
    "scene_conditioned_stability_ratio_gmf",
    "scene_conditioned_violation_diff_lstm",
    "scene_conditioned_violation_diff_gmf",
]

DERIVED_FEATURE_KEYS = [
    "fde_lstm_minus_gmf",
    "fde_scene_minus_gmf",
    "fde_scene_minus_lstm",
    "ade_scene_minus_gmf",
    "ade_scene_minus_lstm",
]


def compute_additional_features(sample: Dict[str, float]) -> Dict[str, float]:
    lstm_fde = float(sample.get("lstm_fde", 0.0))
    gmf_fde = float(sample.get("gameformer_fde", 0.0))
    sc_fde = float(sample.get("scene_conditioned_fde", 0.0))

    lstm_ade = float(sample.get("lstm_ade", lstm_fde))
    gmf_ade = float(sample.get("gameformer_ade", gmf_fde))
    sc_ade = float(sample.get("scene_conditioned_ade", sc_fde))

    return {
        "fde_lstm_minus_gmf": lstm_fde - gmf_fde,
        "fde_scene_minus_gmf": sc_fde - gmf_fde,
        "fde_scene_minus_lstm": sc_fde - lstm_fde,
        "ade_scene_minus_gmf": sc_ade - gmf_ade,
        "ade_scene_minus_lstm": sc_ade - lstm_ade,
    }


def build_feature_vector(sample: Dict[str, float], feature_keys) -> Dict[str, float]:
    base_map = {k: float(sample.get(k, 0.0)) for k in BASE_FEATURE_KEYS}
    base_map.update(compute_additional_features(sample))
    return {k: base_map.get(k, 0.0) for k in feature_keys}
