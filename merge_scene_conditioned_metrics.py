#!/usr/bin/env python3
"""
Merge SceneConditioned metrics into the fusion stats JSON used by gating.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge scene-conditioned metrics into fusion stats.")
    parser.add_argument("--stats", type=str, default="eval_out/fusion_stats_with_meta_features.json", help="Base stats JSON.")
    parser.add_argument("--scene_metrics", type=str, default="eval_out/scene_conditioned_metrics.json", help="SceneConditioned metrics JSON.")
    parser.add_argument("--output", type=str, default="eval_out/fusion_stats_with_scene_conditioned.json", help="Output path (can overwrite stats).")
    args = parser.parse_args()

    stats_path = Path(args.stats)
    scene_path = Path(args.scene_metrics)
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene metrics file not found: {scene_path}")

    stats = json.loads(stats_path.read_text())
    scene_metrics = json.loads(scene_path.read_text())
    metrics_map = {int(entry["sample_idx"]): entry for entry in scene_metrics}

    merged = []
    missing = 0
    for entry in stats:
        idx = int(entry.get("sample_idx", -1))
        metric = metrics_map.get(idx)
        if metric:
            entry["scene_conditioned_ade"] = metric.get("scene_conditioned_ade")
            entry["scene_conditioned_fde"] = metric.get("scene_conditioned_fde")
            entry["scene_conditioned_source"] = metric.get("semantic_source")
            entry["scene_conditioned_scene_type"] = metric.get("scene_type")
        else:
            missing += 1
        merged.append(entry)

    Path(args.output).write_text(json.dumps(merged, indent=2))
    print(f"Merged {len(stats) - missing} scene-conditioned entries (missing {missing}).")
    if args.output == args.stats:
        print("⚠️ Stats file overwritten with merged data.")


if __name__ == "__main__":
    main()

