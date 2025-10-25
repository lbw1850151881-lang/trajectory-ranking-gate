"""
Utility to compute open-loop planning metrics (ADE/FDE) from NuPlan simulation logs.

The script expects a path to a `.msgpack.xz` simulation log produced by
`nuplan/planning/script/run_simulation.py`. It loads the serialized
`SimulationLog`, extracts the planned trajectories from the history, and
aligns them with the expert (ground-truth) future trajectory for the same
scenario iteration. We focus on the predicted planning trajectories rather
than the realized executed path.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# nuPlan modules live outside the repo root; the caller must ensure PYTHONPATH
# already includes both the GameFormer-Planner repo and nuplan-devkit.
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.simulation_log import SimulationLog


def _collect_plan_states(sample) -> List[EgoState]:
    """
    Return the list of ego states composing the planner's predicted trajectory
    for the given simulation sample.
    """
    trajectory = sample.trajectory
    return trajectory.get_sampled_trajectory()


def _collect_future_states(scenario, iteration: int, horizon_s: float, num_requested: int) -> List[EgoState]:
    """
    Fetch the expert (ground truth) future trajectory starting at the provided
    scenario iteration. The generator returned by nuPlan omits the current
    state, so the length typically matches the requested horizon.
    """
    future_gen: Iterable[EgoState] = scenario.get_ego_future_trajectory(
        iteration, horizon_s, num_samples=num_requested
    )
    return list(future_gen)


def _compute_errors(
    plan_states: List[EgoState],
    future_states: List[EgoState],
) -> Tuple[np.ndarray, float]:
    """
    Compute the per-timestep displacement errors (meters) and the final-step
    displacement (FDE) between planner predictions and expert future states.

    The first plan state coincides with the current ego state, so we ignore it
    and compare plan[1:] to the expert sequence.
    """
    if len(plan_states) <= 1 or not future_states:
        return np.empty((0,), dtype=np.float32), float("nan")

    plan_use = plan_states[1:]
    min_len = min(len(plan_use), len(future_states))
    if min_len == 0:
        return np.empty((0,), dtype=np.float32), float("nan")

    plan_use = plan_use[:min_len]
    future_use = future_states[:min_len]

    diffs = []
    for pred, gt in zip(plan_use, future_use):
        dx = pred.center.x - gt.center.x
        dy = pred.center.y - gt.center.y
        diffs.append(np.hypot(dx, dy))
    errors = np.asarray(diffs, dtype=np.float32)
    fde = float(errors[-1])
    return errors, fde


def evaluate_log(log_path: Path) -> dict:
    """
    Load the simulation log and compute ADE/FDE metrics aggregated across all
    simulation iterations.
    """
    log = SimulationLog.load_data(log_path)
    scenario = log.scenario

    all_errors: List[np.ndarray] = []
    final_errors: List[float] = []

    # Iterate over each simulation step.
    for sample in log.simulation_history.data:
        plan_states = _collect_plan_states(sample)
        if len(plan_states) <= 1:
            continue

        horizon_s = float(plan_states[-1].time_point.time_s - plan_states[0].time_point.time_s)
        future_states = _collect_future_states(
            scenario,
            iteration=sample.iteration.index,
            horizon_s=horizon_s,
            num_requested=len(plan_states),
        )

        errors, fde = _compute_errors(plan_states, future_states)
        if errors.size == 0 or not np.isfinite(fde):
            continue

        all_errors.append(errors)
        final_errors.append(fde)

    if not all_errors:
        raise RuntimeError(f"No valid trajectories found in {log_path}.")

    concatenated = np.concatenate(all_errors, axis=0)
    ade = float(np.mean(concatenated))
    fde = float(np.mean(final_errors))

    return {
        "scenario_name": scenario.scenario_name,
        "scenario_type": scenario.scenario_type,
        "iterations_evaluated": len(all_errors),
        "ade": ade,
        "fde": fde,
        "ade_std": float(np.std(concatenated)),
        "fde_std": float(np.std(final_errors)),
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ADE/FDE metrics from a NuPlan simulation log.")
    parser.add_argument("log_path", type=Path, help="Path to .msgpack.xz simulation log.")
    args = parser.parse_args()

    if not args.log_path.exists():
        raise SystemExit(f"Log file not found: {args.log_path}")

    metrics = evaluate_log(args.log_path)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
