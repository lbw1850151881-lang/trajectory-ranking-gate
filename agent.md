## Project Snapshot
- **Repo root:** `/home/hamster/GameFormer-Planner`
- **NuPlan assets:** `/mnt/d/paper/nuplan` (mini DB, maps, devkit). Use `/home/hamster/miniconda3/bin/conda run -n nuplan …` for every run.
- **Goal:** Deliver an open-loop NuPlan planner where an LLM-enhanced gate (GameFormer + SceneConditioned + LSTM) reliably beats the single GameFormer baseline on ADE/FDE, especially on long-tail left-turn scenarios.

## Key Artifacts (current as of Oct 16 2025)
| Purpose | Path |
| --- | --- |
| LLM gate entry point | `Planner/llm_gate_planner.py` |
| State-lattice planner (short-path fix) | `Planner/state_lattice_path_planner.py` |
| Scene-conditioned model (v4, ADE 3.356 m) | `training_log/SceneConditioned_v4/model_best_ADE_3.356.pth` |
| LSTM expert (80-step) | `training_log/LSTM_v2/best_seq2seq_lstm.pth` |
| Tri-expert gate logits (latest) | `eval_out/tri_gate_v4.pth` |
| Ranking gate summary | `eval_out/ranking_gate_results.json` |
| Override calibrator (v4) | `eval_out/trained_calibrator_v4.pth` |
| Meta-feature dataset | `eval_out/fusion_stats_with_scene_conditioned.json` |
| Scene-conditioned per-sample metrics | `eval_out/scene_conditioned_metrics.json` |
| Baseline metrics (GameFormer) | `testing_log/exp/open_loop_boxes/gameformer_metrics.txt` |
| Latest LLM gate metrics | `testing_log/exp/open_loop_boxes/llm_gate_metrics.txt` |
| Analysis runs (forced experts) | `analysis_runs/*.msgpack.xz` (see Notes) |

## Recent Progress
1. **Expert refresh** – retrained SceneConditioned GameFormer (v4) and Seq2Seq LSTM (LSTM v2); validated on processed NuPlan mini dataset.
2. **Meta-feature pipeline** – `meta_features_extractor.py` now samples GameFormer + SceneConditioned via MC Dropout; `extract_meta_features_dataset.py` produces `fusion_stats_with_scene_conditioned.json`.
3. **Gate training** – tri-expert classifier updated (`tri_gate_v4.pth`) and override calibrator retrained to the new `SemanticFeatures` schema.
4. **Planner stability** – state lattice post-processing handles short candidate paths, preventing cubic-spline crashes (fix in `state_lattice_path_planner.py`).
5. **Scenario deep-dive** – isolated two problematic left-turn scenarios and ran forced-expert simulations to quantify each expert’s contribution.

## Current Metrics (NuPlan mini – open-loop)
| Scenario | GameFormer ADE/FDE | LLM Gate ADE/FDE (v4) | Notes |
| --- | --- | --- | --- |
| `c1bf7374695a5c47` (starting_left_turn) | 12.28 / 18.17 | **14.77 / 30.97** | Gate and GF both clip the trajectory to ~8 s; high FDE due to truncated plan |
| `0580d87acd0e52b5` (starting_left_turn) | 7.92 / 12.61 | **11.31 / 26.00** | Gate selects SceneConditioned/LSTM frequently; similar truncation issue |
| `7790a109c37c530f` (starting_right_turn) | 6.73 / 7.90 | 6.93 / 8.43 | Near parity |

**Forced expert experiments (analysis_runs/)**  
- `scene_only_left1.msgpack.xz`: SceneConditioned forced on `c1bf737…` ⇒ ADE 12.54 / FDE 33.96  
- `scene_only_left2.msgpack.xz`: SceneConditioned forced on `0580d8…` ⇒ ADE 8.09 / FDE 18.70  
- `lstm_left1.msgpack.xz`: LSTM forced on `c1bf737…` ⇒ ADE 21.48 / FDE 43.53  
- `lstm_left2.msgpack.xz`: LSTM forced on `0580d8…` ⇒ ADE 14.76 / FDE 30.40  
👉 SceneConditioned mildly helps the second left-turn but still trails GameFormer; LSTM degrades both scenarios.

## Findings from the Left-turn Investigation
- **Gating decisions:** `~149` iterations per scenario. Gate chooses `{GameFormer:55, SceneConditioned:53, LSTM:41}` on `c1bf737…`; `{SceneConditioned:82, LSTM:47, GameFormer:20}` on `0580d8…`.
- **Trajectory horizon mismatch:** State lattice truncates generated paths to ~8 s, whereas ground-truth future extends to 80 steps; FDE becomes the distance between the short path end-point and the real trajectory at 8 s vs 80 s. Baseline GameFormer exhibits the same issue (ADE matches forced runs).
- **SceneConditioned regression:** Despite better validation ADE, forced runs show minimal improvement and even worsen FDE for the first left-turn.
- **LSTM regression:** LSTM-only runs catastrophically fail, indicating the v2 weights are not reliable under current preprocessing.

## How to Reproduce Key Runs
```bash
# Baseline GameFormer (already logged in gameformer_metrics.txt)
PYTHONPATH=/home/hamster/GameFormer-Planner:/mnt/d/paper/nuplan/nuplan-devkit \
GAMEFORMER_PLANNER_CKPT=/home/hamster/GameFormer-Planner/pretrained_models/GameFormer-Planner/training_log/Exp1/model_epoch_18_valADE_1.7272.pth \
NUPLAN_DATA_ROOT=/mnt/d/paper/nuplan \
NUPLAN_MAPS_ROOT=/mnt/d/paper/nuplan/nuplan-maps-v1.0/maps \
NUPLAN_EXP_ROOT=/home/hamster/GameFormer-Planner/testing_log \
/home/hamster/miniconda3/bin/conda run -n nuplan python \
  /mnt/d/paper/nuplan/nuplan-devkit/nuplan/planning/script/run_simulation.py \
  experiment=open_loop_boxes planner=gameformer_planner worker=sequential \
  scenario_builder=nuplan_mini \
  scenario_filter.log_names="['2021.05.25.14.16.10_veh-35_01690_02183','2021.05.12.23.36.44_veh-35_01133_01535','2021.05.12.23.36.44_veh-35_00152_00504']" \
  scenario_filter.scenario_types="['changing_lane','starting_left_turn','starting_right_turn']" \
  scenario_filter.limit_total_scenarios=3 scenario_filter.shuffle=false

# LLM gate (tri_gate_v4, SceneConditioned_v4, LSTM_v2)
PYTHONPATH=/home/hamster/GameFormer-Planner:/mnt/d/paper/nuplan/nuplan-devkit \
LLM_GATE_GAMEFORMER=/home/hamster/GameFormer-Planner/pretrained_models/GameFormer-Planner/training_log/Exp1/model_epoch_18_valADE_1.7272.pth \
LLM_GATE_SCENE=/home/hamster/GameFormer-Planner/training_log/SceneConditioned_v4/model_best_ADE_3.356.pth \
LLM_GATE_LSTM=/home/hamster/GameFormer-Planner/training_log/LSTM_v2/best_seq2seq_lstm.pth \
LLM_GATE_RANKING=/home/hamster/GameFormer-Planner/eval_out/tri_gate_v4.pth \
LLM_GATE_VOCAB=/home/hamster/GameFormer-Planner/clusters_200/semantic_vocab.json \
LLM_GATE_CACHE=/home/hamster/GameFormer-Planner/eval_out/llm_semantic_cache.json \
NUPLAN_DATA_ROOT=/mnt/d/paper/nuplan \
NUPLAN_MAPS_ROOT=/mnt/d/paper/nuplan/nuplan-maps-v1.0/maps \
NUPLAN_EXP_ROOT=/home/hamster/GameFormer-Planner/testing_log \
/home/hamster/miniconda3/bin/conda run -n nuplan python \
  /mnt/d/paper/nuplan/nuplan-devkit/nuplan/planning/script/run_simulation.py \
  experiment=open_loop_boxes planner=llm_gate_planner worker=sequential \
  job_name=open_loop_llm_gate_scene_v4 \
  scenario_builder=nuplan_mini \
  scenario_filter.log_names="['2021.05.25.14.16.10_veh-35_01690_02183','2021.05.12.23.36.44_veh-35_01133_01535','2021.05.12.23.36.44_veh-35_00152_00504']" \
  scenario_filter.scenario_types="['changing_lane','starting_left_turn','starting_right_turn']" \
  scenario_filter.limit_total_scenarios=3 scenario_filter.shuffle=false scenario_filter.remove_invalid_goals=false

# Force a specific expert (set LLM_GATE_FORCE_EXPERT to scene_conditioned | lstm | gameformer)
LLM_GATE_FORCE_EXPERT=scene_conditioned ... run_simulation.py ...

# Post-process metrics
PYTHONPATH=/home/hamster/GameFormer-Planner:/mnt/d/paper/nuplan/nuplan-devkit \
/home/hamster/miniconda3/bin/conda run -n nuplan python compute_open_loop_metrics.py <path_to_msgpack.xz>
```

## Outstanding Next Steps
1. **Fix trajectory truncation**
   - Investigate why the lattice planner trims to ~80 samples (8 s). Confirm `_plan_to_states` vs NuPlan horizon (80 steps) and ensure full-length trajectories reach the goal. Consider extending `_plan_to_states` resampling.
2. **SceneConditioned model diagnostics**
   - Review training logs in `training_log/SceneConditioned_v4/train.log` and `training_history.json`.
   - Check data pipeline (semantic vocab, weighting) to ensure v4 generalizes beyond validation.
   - Consider fine-tuning on the problematic left-turn slice with longer temporal context.
3. **Gate retraining adjustments**
   - Tri-gate rarely selects SceneConditioned despite high weight. Explore class weighting or thresholding (increase penalties for large FDE deltas).
   - Evaluate using `LLM_GATE_FORCE_EXPERT` to collect per-expert metrics for additional scenarios and feed into gate retraining.
4. **Override calibrator integration**
   - Hook `trained_calibrator_v4.pth` back into `llm_gate_planner._select_expert` to override low-confidence decisions, especially for left turns.
5. **Logging improvements**
   - `~/llm_gate_debug.log` already records scenario, type, iteration, decision, ADE/FDE. Consider persisting per-iteration ground-truth deltas to ease analysis.

## Operational Notes
- Sequential worker remains the default (Ray + CUDA on WSL causes worker crashes).
- NuPlan map JSON must come from `/mnt/d/paper/nuplan/nuplan-maps-v1.0/maps`. Mini DB symlink lives at `/mnt/d/paper/nuplan/nuplan-v1.1/splits/mini`.
- LLM feature extractor cache: `eval_out/llm_semantic_cache.json`. Delete only if fresh semantic labels are needed.
- Analysis artifacts (forced runs) reside in `analysis_runs/`. Keep these for future regression comparisons.

Keep this document updated whenever new checkpoints or investigations modify the plan so the next agent can resume with full context.***
