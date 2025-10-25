#!/usr/bin/env python3
"""
Generate per-sample metrics for the Scene-Conditioned GameFormer.

This script evaluates the trained scene-conditioned expert on the full validation
set, using LLM-provided semantic labels when available and falling back to the
heuristic (offline) LLM feature extractor otherwise. The resulting ADE/FDE
statistics can be merged into the gating datasets so that the LLM gate can
choose amongst three experts (LSTM, GameFormer, SceneConditioned).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from GameFormer.train_utils import DrivingData
from llm_feature_extractor import LLMFeatureExtractor
from scene_conditioned_gameformer import SceneConditionedGameFormer, get_scene_type_mapping


def load_vocab(vocab_path: Path) -> Tuple[Dict[str, int], int]:
    data = json.loads(vocab_path.read_text())
    keyword_to_id = data.get("keyword_to_id")
    if not keyword_to_id:
        vocab = data.get("vocab", [])
        keyword_to_id = {kw: idx for idx, kw in enumerate(vocab)}
    return keyword_to_id, len(keyword_to_id)


def normalise_scene_type(raw_scene: str, mapping: Dict[str, int]) -> str:
    if not raw_scene:
        return "other"
    tokens = [token.strip().lower() for token in raw_scene.replace("-", "_").split("|")]
    for token in tokens:
        if token in mapping:
            return token
    # fall back to best effort heuristics
    heuristics = {
        "intersection": ["intersection"],
        "cut_in": ["cut_in", "lane_change"],
        "congestion": ["congestion", "traffic"],
        "yielding": ["yield"],
        "merging": ["merge"],
        "occlusion": ["occlusion", "occluded"],
        "high_speed": ["high_speed", "speed"],
    }
    lower = raw_scene.lower()
    for target, hints in heuristics.items():
        if any(h in lower for h in hints):
            return target
    return "other"


def build_scene_context(stats: Dict[str, float], default_complexity: float = 0.5) -> Dict[str, float]:
    return {
        "complexity_score": float(stats.get("complexity_score", default_complexity)),
        "n_neighbors": float(stats.get("n_neighbors", stats.get("num_neighbors", 0))),
        "speed_variance": float(stats.get("speed_variance", 0.0)),
        "min_neighbor_dist": float(stats.get("min_neighbor_dist", 10.0)),
        "n_crosswalks": float(stats.get("n_crosswalks", 0)),
    }


def compute_ego_metrics(plan_trajectory: torch.Tensor, ego_future: torch.Tensor) -> Tuple[float, float]:
    distance = torch.norm(plan_trajectory[:, :, :2] - ego_future[:, :, :2], dim=-1)
    ade = torch.mean(distance).item()
    fde = torch.mean(distance[:, -1]).item()
    return ade, fde


def convert_keywords(keywords: List[str], keyword_to_id: Dict[str, int], vocab_size: int) -> torch.Tensor:
    vec = torch.zeros(vocab_size, dtype=torch.float32)
    for kw in keywords:
        key = kw.strip().lower()
        if key in keyword_to_id:
            vec[keyword_to_id[key]] = 1.0
    return vec


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-sample SceneConditioned metrics.")
    parser.add_argument("--conditioned_model", type=str, required=True, help="Path to SceneConditioned checkpoint.")
    parser.add_argument("--data_dir", type=str, default="/home/hamster/nuplan/processed_data/valid", help="Processed data directory.")
    parser.add_argument("--stats_path", type=str, default="eval_out/fusion_stats_with_meta_features.json", help="Existing stats JSON (for context).")
    parser.add_argument("--labels_path", type=str, default="eval_out/llm_longtail_labels.json", help="LLM label file (optional).")
    parser.add_argument("--vocab_path", type=str, default="eval_out/clusters/semantic_vocab.json", help="Semantic vocabulary JSON.")
    parser.add_argument("--output", type=str, default="eval_out/scene_conditioned_metrics.json", help="Output JSON path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on.")
    parser.add_argument("--encoder_layers", type=int, default=3, help="SceneConditioned encoder layers.")
    parser.add_argument("--decoder_levels", type=int, default=2, help="SceneConditioned decoder levels.")
    parser.add_argument("--num_neighbors", type=int, default=10, help="Number of neighbour agents.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    keyword_to_id, vocab_size = load_vocab(Path(args.vocab_path))
    scene_type_mapping = get_scene_type_mapping()

    stats_map = {}
    stats_path = Path(args.stats_path)
    if stats_path.exists():
        with stats_path.open("r") as f:
            stats_data = json.load(f)
        stats_map = {int(item["sample_idx"]): item for item in stats_data}

    labels_map: Dict[int, Dict[str, Any]] = {}
    labels_path = Path(args.labels_path)
    if labels_path.exists():
        with labels_path.open("r") as f:
            labels = json.load(f)
        if isinstance(labels, dict):
            labels = labels.get("labels", [])
        for item in labels or []:
            labels_map[int(item.get("sample_idx", -1))] = item

    print(f"Loaded {len(stats_map)} stats entries, {len(labels_map)} semantic labels, vocab size {vocab_size}.")

    dataset = DrivingData(str(Path(args.data_dir) / "*.npz"), n_neighbors=args.num_neighbors)
    print(f"Dataset size: {len(dataset)} samples.")

    extractor = LLMFeatureExtractor(offline=True)

    model = SceneConditionedGameFormer(
        vocab_size=vocab_size,
        encoder_layers=args.encoder_layers,
        decoder_levels=args.decoder_levels,
        neighbors=args.num_neighbors,
    )
    checkpoint = torch.load(args.conditioned_model, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    results = []

    for idx in tqdm(range(len(dataset)), desc="SceneConditioned inference"):
        ego, neighbors, map_lanes, map_crosswalks, route_lanes, ego_future_gt, _ = dataset[idx]

        stats = stats_map.get(idx, {})
        scene_context = build_scene_context(stats)

        if idx in labels_map:
            label_info = labels_map[idx]
            scene_type_raw = label_info.get("scene_type", "other")
            keywords = label_info.get("semantic_keywords", [])
        else:
            semantic = extractor.extract_features(scene_context)
            scene_type_raw = semantic.scene_type
            keywords = semantic.semantic_keywords

        scene_type = normalise_scene_type(scene_type_raw, scene_type_mapping)
        scene_type_id = scene_type_mapping.get(scene_type, scene_type_mapping["other"])
        keyword_vec = convert_keywords(keywords, keyword_to_id, vocab_size)

        inputs = {
            "ego_agent_past": torch.tensor(ego, dtype=torch.float32, device=device).unsqueeze(0),
            "neighbor_agents_past": torch.tensor(neighbors, dtype=torch.float32, device=device).unsqueeze(0),
            "map_lanes": torch.tensor(map_lanes, dtype=torch.float32, device=device).unsqueeze(0),
            "map_crosswalks": torch.tensor(map_crosswalks, dtype=torch.float32, device=device).unsqueeze(0),
            "route_lanes": torch.tensor(route_lanes, dtype=torch.float32, device=device).unsqueeze(0),
        }

        scene_type_tensor = torch.tensor([scene_type_id], dtype=torch.long, device=device)
        keyword_tensor = keyword_vec.unsqueeze(0).to(device)

        with torch.no_grad():
            _, ego_plan = model(
                inputs,
                scene_type_ids=scene_type_tensor,
                keyword_vectors=keyword_tensor,
            )

        ego_future = torch.tensor(ego_future_gt, dtype=torch.float32, device=device).unsqueeze(0)
        ade, fde = compute_ego_metrics(ego_plan, ego_future)

        results.append(
            {
                "sample_idx": idx,
                "scene_conditioned_ade": ade,
                "scene_conditioned_fde": fde,
                "scene_type": scene_type,
                "keywords": [kw for kw in keywords if kw in keyword_to_id],
                "semantic_source": "label" if idx in labels_map else "heuristic",
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} entries to {output_path}")


if __name__ == "__main__":
    main()
