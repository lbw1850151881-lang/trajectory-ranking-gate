"""
Evaluation entry-point for the LLM-enhanced gate.

The script mirrors the workflow described in the experimental log:
1. Load pre-computed meta features (NPZ) and per-sample statistics (JSON).
2. Score each sample with the ranking gate.
3. Optionally trigger the LLM supervisor for semantic overrides.
4. Aggregate metrics and emit both summary statistics and a detailed trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from llm_enhanced_gate import LLMEnhancedGate, load_tri_checkpoint
from llm_supervisor import LLMSupervisor
from llm_feature_extractor import LLMFeatureExtractor


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM-enhanced gating.")
    parser.add_argument(
        "--ranking_model",
        type=str,
        default="eval_out/ranking_gate.pth",
        help="Path to ranking gate checkpoint.",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default="eval_out/meta_features_dataset.npz",
        help="NPZ file containing feature matrix and FDEs.",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="eval_out/fusion_stats_with_meta_features.json",
        help="JSON with per-sample ADE/FDE and context features.",
    )
    parser.add_argument("--output_dir", type=str, default="eval_out", help="Output folder.")
    parser.add_argument("--device", type=str, default="cpu", help="torch device for ranking model.")

    parser.add_argument("--llm_enabled", action="store_true", help="Enable LLM supervisor.")
    parser.add_argument("--llm_override", action="store_true", help="Allow LLM to override.")
    parser.add_argument(
        "--verification_enabled",
        action="store_true",
        help="Enable trajectory verification safeguard.",
    )
    parser.add_argument(
        "--llm_trigger_threshold",
        type=float,
        default=0.3,
        help="Ranking confidence threshold that triggers LLM calls.",
    )
    parser.add_argument(
        "--base_threshold",
        type=float,
        default=-0.25,
        help="Threshold applied to ranking scores.",
    )
    parser.add_argument(
        "--min_lstm_prob",
        type=float,
        default=0.8,
        help="Minimum probability to keep LSTM selection.",
    )
    parser.add_argument(
        "--min_scene_prob",
        type=float,
        default=0.4,
        help="Minimum probability to keep SceneConditioned selection.",
    )
    parser.add_argument(
        "--sc_hybrid_threshold",
        type=float,
        default=0.45,
        help="Hybrid score threshold for SceneConditioned overrides.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples (useful for debugging).",
    )
    parser.add_argument(
        "--offline_llm",
        action="store_true",
        help="Force supervisor to operate in heuristic/offline mode.",
    )
    parser.add_argument(
        "--save_details",
        action="store_true",
        help="Persist per-sample decision trace to JSON.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate a lightweight matplotlib summary figure.",
    )
    parser.add_argument("--debug", action="store_true", help="Print additional diagnostics.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_npz_features(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return (
        data["features"],
        data["sample_indices"],
        data["lstm_fde"],
        data["gmf_fde"],
    )


def load_stats_json(path: str) -> Dict[int, Dict[str, Any]]:
    with open(path, "r") as f:
        samples = json.load(f)
    return {int(sample["sample_idx"]): sample for sample in samples}


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(
    decisions,
    stats_map,
) -> Dict[str, Any]:
    ade_values = []
    fde_values = []
    oracle_fdes = []
    gmf_fdes = []
    base_lstm = 0
    base_gmf = 0
    base_scene = 0
    final_lstm = 0
    final_gmf = 0
    final_scene = 0
    llm_triggered = 0
    llm_override = 0

    for decision in decisions:
        sample = stats_map.get(decision.sample_idx, {})
        lstm_fde = sample.get("lstm_fde", 0.0)
        gmf_fde = sample.get("gameformer_fde", 0.0)
        lstm_ade = sample.get("lstm_ade", lstm_fde)
        gmf_ade = sample.get("gameformer_ade", gmf_fde)
        scene_fde = sample.get("scene_conditioned_fde", gmf_fde)
        scene_ade = sample.get("scene_conditioned_ade", gmf_ade)

        base_decision = decision.base_decision
        final_decision = decision.final_decision

        if base_decision == "lstm":
            base_lstm += 1
        elif base_decision == "gameformer":
            base_gmf += 1
        else:
            base_scene += 1

        if final_decision == "lstm":
            final_lstm += 1
            ade_values.append(lstm_ade)
            fde_values.append(lstm_fde)
        elif final_decision == "scene_conditioned":
            final_scene += 1
            ade_values.append(scene_ade)
            fde_values.append(scene_fde)
        else:
            final_gmf += 1
            ade_values.append(gmf_ade)
            fde_values.append(gmf_fde)

        oracle_fdes.append(min(lstm_fde, gmf_fde, scene_fde))
        gmf_fdes.append(gmf_fde)

        if decision.llm_triggered:
            llm_triggered += 1
            if decision.llm_decision and decision.llm_decision.get("override"):
                llm_override += 1

    ade_values = np.array(ade_values)
    fde_values = np.array(fde_values)
    oracle_fdes = np.array(oracle_fdes)
    gmf_fdes = np.array(gmf_fdes)

    ade = float(np.mean(ade_values))
    fde = float(np.mean(fde_values))
    ade_std = float(np.std(ade_values))
    fde_std = float(np.std(fde_values))

    oracle = float(np.mean(oracle_fdes))
    gmf_mean = float(np.mean(gmf_fdes))
    oracle_realization = 0.0
    if gmf_mean > oracle:
        oracle_realization = (gmf_mean - fde) / (gmf_mean - oracle)

    return {
        "metrics": {
            "ade": f"{ade:.6f}",
            "fde": f"{fde:.6f}",
            "ade_std": f"{ade_std:.6f}",
            "fde_std": f"{fde_std:.6f}",
            "oracle_fde": f"{oracle:.6f}",
            "oracle_realization_pct": f"{oracle_realization * 100:.1f}",
        },
        "statistics": {
            "total_samples": len(decisions),
            "ranking_lstm": base_lstm,
            "ranking_gmf": base_gmf,
            "ranking_scene_conditioned": base_scene,
            "final_lstm": final_lstm,
            "final_gmf": final_gmf,
            "final_scene_conditioned": final_scene,
            "llm_triggered": llm_triggered,
            "llm_override": llm_override,
            "ranking_lstm_rate": f"{(base_lstm / max(1, len(decisions))) * 100:.1f}%",
            "ranking_gmf_rate": f"{(base_gmf / max(1, len(decisions))) * 100:.1f}%",
            "ranking_scene_conditioned_rate": f"{(base_scene / max(1, len(decisions))) * 100:.1f}%",
            "llm_trigger_rate": f"{(llm_triggered / max(1, len(decisions))) * 100:.1f}%",
            "llm_override_rate": f"{(llm_override / max(1, llm_triggered)) * 100:.1f}%" if llm_triggered else "0.0%",
            "final_lstm_rate": f"{(final_lstm / max(1, len(decisions))) * 100:.1f}%",
            "final_gmf_rate": f"{(final_gmf / max(1, len(decisions))) * 100:.1f}%",
            "final_scene_conditioned_rate": f"{(final_scene / max(1, len(decisions))) * 100:.1f}%",
        },
    }


# ---------------------------------------------------------------------------
# Visualization (lightweight)
# ---------------------------------------------------------------------------


def maybe_visualize(decisions, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return

    scores = [d.ranking_score for d in decisions]
    triggers = [d.llm_triggered for d in decisions]
    overrides = [
        d.llm_decision.get("override", False) if d.llm_decision else False for d in decisions
    ]

    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=40, alpha=0.7, label="Ranking scores")
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="Zero boundary")
    plt.xlabel("Ranking score")
    plt.ylabel("Count")
    plt.title("Score distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "llm_gate_score_hist.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 3))
    plt.bar(["Triggered", "Overrides"], [sum(triggers), sum(overrides)])
    plt.ylabel("Count")
    plt.title("LLM actions")
    plt.tight_layout()
    plt.savefig(output_dir / "llm_gate_actions.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, sample_indices, lstm_fde, gmf_fde = load_npz_features(args.features_path)
    stats_map = load_stats_json(args.stats_path)

    if args.max_samples is not None:
        features = features[: args.max_samples]
        sample_indices = sample_indices[: args.max_samples]

    checkpoint = load_tri_checkpoint(args.ranking_model, device=args.device)

    extractor = LLMFeatureExtractor(offline=args.offline_llm or not args.llm_enabled)
    supervisor = LLMSupervisor(
        extractor=extractor,
        sc_hybrid_threshold=args.sc_hybrid_threshold,
    )

    gate = LLMEnhancedGate(
        checkpoint,
        supervisor=supervisor,
        llm_enabled=args.llm_enabled,
        base_threshold=args.base_threshold,
        trigger_threshold=args.llm_trigger_threshold,
        llm_override=args.llm_override and args.llm_enabled,
        verification_enabled=args.verification_enabled,
        device=args.device,
        min_lstm_prob=args.min_lstm_prob,
        min_scene_prob=args.min_scene_prob,
    )

    scene_contexts: List[Dict[str, Any]] = [
        stats_map[int(idx)] for idx in sample_indices
    ]

    decisions = gate.evaluate(
        features,
        scene_contexts,
        sample_indices=sample_indices,
    )

    summary = compute_metrics(decisions, stats_map)
    summary["statistics"]["llm_cache_size"] = len(extractor.cache)
    summary["statistics"]["llm_offline_mode"] = extractor.offline
    summary["args"] = vars(args)

    output_path = output_dir / "llm_gate_results.json"
    output_path.write_text(json.dumps(summary, indent=2))

    if args.save_details:
        details_path = output_dir / "llm_gate_details.json"
        details_payload = [decision.to_dict() for decision in decisions]
        details_path.write_text(json.dumps(details_payload, indent=2))

    if args.visualize:
        maybe_visualize(decisions, output_dir)

    if args.debug:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
