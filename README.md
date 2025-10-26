# Dynamic Model Selection for Trajectory Prediction

> Open-source reference implementation of the paper **“Dynamic Model Selection for Trajectory Prediction via Pairwise Ranking and Meta-Features”** . The repository exposes the full tri-expert gating stack—experts, meta-features, learning-to-rank gate, and LLM supervision—to reproduce the reported gains on the 1,287-scene **nuPlan-mini** benchmark.

## Project Overview
- **Motivation**: Single-model predictors collapse in long-tail scenes (dense intersections, occlusions, cut-ins). We instead select the most reliable expert per sample while preserving physical feasibility.
- **Approach**: Combine an LSTM–Kalman Filter physics expert, a scene-conditioned transformer, and a fine-tuned GameFormer. Internal signals (uncertainty, stability, physics violations) form meta-features that drive a RankNet-style gate, further audited by an LLM supervisor.
- **Headline results**: Validation FDE improves from 2.835 m (GameFormer) to 2.567 m (−9.5%) with **57.8% Oracle Realization Rate**, and left-turn open-loop FDE drops by roughly 10%.
- **Key entry points**: `meta_features_extractor.py`, `gate_ranking.py`, `llm_enhanced_gate.py`, `Planner/llm_gate_planner.py`, and `scripts/run_llm_gate.sh`.

## Core Contributions
1. **Meta-feature driven gating** – first systematic use of internal model signals (MC Dropout uncertainty, input perturbation stability, physics violation rates) for trajectory expert selection.
2. **Pairwise ranking formulation** – RankNet loss learns relative superiority between experts, avoiding calibration issues typical of regression/classification gates.
3. **LLM semantic oversight** – a lightweight supervisor triggers on 34% of scenes based on confidence and semantic risk, overriding to safer experts when necessary.
4. **Oracle-gap quantification & open-loop validation** – Oracle Realization Rate (ORR) tracks how much of the theoretical gain is achieved, and nuPlan open-loop simulations confirm offline improvements transfer to planning.

## Methodology
### Tri-Expert Ensemble
- `train_predictor_lstm.py` – trains a Seq2Seq LSTM + Kalman Filter that strictly respects dynamics (mean FDE 8.12 m) for low-uncertainty scenes.
- `train_scene_conditioned.py` – scene-conditioned GameFormer variant specialized for long-tail interactions (mean FDE 7.07 m).
- Pretrained GameFormer – strong generalist (FDE 2.84 m) handling ~68.7% of scenarios.

### Meta-Feature Pipeline
- `meta_features_extractor.py` generates >20 features via MC Dropout, input perturbations, and physics-violation tests.
- `extract_meta_features_dataset.py` merges geometric descriptors with meta-features into `eval_out/fusion_stats_with_scene_conditioned.json` for gate training.

### Pairwise Ranking Gate
- `gate_ranking.py` fits a RankNet-style scorer (labels ±1 denote which expert wins) with automatic feature normalization and threshold search.
- Outputs: `eval_out/ranking_gate.pth`, `ranking_gate_results.json`, and diagnostics such as `ranking_gate_analysis.png`.

### LLM-Enhanced Inference
- `llm_enhanced_gate.py` fuses TriExpert logits with LLM guidance; low confidence (<0.4) or semantic risk triggers an override, optionally verified by physics checks.
- `Planner/llm_gate_planner.py` loads expert checkpoints, ranking parameters, and the semantic vocabulary (`clusters_200/semantic_vocab.json`) to deploy the gate inside nuPlan.

## Experimental Results
| Model | FDE↓ (m) | ADE↓ (m) | Oracle Realization |
| --- | --- | --- | --- |
| LSTM–KF (Physics) | 8.117 ± 0.15 | 2.820 ± 0.09 | – |
| Transformer (Long-tail) | 7.066 ± 0.12 | 2.574 ± 0.08 | – |
| GameFormer (Baseline) | 2.835 ± 0.04 | 1.469 ± 0.02 | 0% |
| **LLM-Enhanced Gate (Ours)** | **2.567 ± 0.03** | **1.255 ± 0.02** | **57.8%** |

- **Open-loop simulation**: left-turn scenario `c1bf7374695a5c47` FDE drops from 18.17 m to 16.37 m; LLM interventions occur in ~34% of scenes.
- **Expert allocation**: GameFormer 68.7%, scene-conditioned 25.3%, LSTM 6.0% (physics safeguard).

## Repository Layout
```
multi_agent_package/
├── Planner/llm_gate_planner.py        # nuPlan planner entry point with LLM-enhanced gating
├── GameFormer/                        # GameFormer + scene-conditioned training code
├── scripts/run_llm_gate.sh            # Example script to reproduce open-loop experiments
├── meta_features_extractor.py         # Meta-feature extraction utilities
├── extract_meta_features_dataset.py   # Dataset builder for gate training
├── gate_ranking.py                    # Pairwise ranking gate training
├── llm_enhanced_gate.py               # Inference-time fusion of logits and LLM overrides
├── training_log/, pretrained_models/  # Checkpoints and logs
└── my_paper.tex                       # Manuscript source
```

## Quick Start
### Environment
1. Python >= 3.10, PyTorch >= 2.1 with CUDA, plus nuPlan-devkit dependencies.
2. Use a Conda env (e.g., `nuplan`) and append `GameFormer` + `nuplan-devkit` to `PYTHONPATH`.
3. Install additional requirements:
   ```bash
   pip install -r requirements.txt  # includes theseus-ai==0.1.3
   ```

### Data & Resources
```bash
export NUPLAN_DATA_ROOT=/mnt/d/paper/nuplan
export NUPLAN_MAPS_ROOT=/mnt/d/paper/nuplan/nuplan-maps-v1.0/maps
export NUPLAN_EXP_ROOT=/home/hamster/GameFormer-Planner/testing_log
```
Place the LSTM, scene-conditioned, and GameFormer checkpoints under `training_log/` or `pretrained_models/` (paths documented in `agent.md`).

### Training Pipeline
1. **Train experts**
   ```bash
   python train_predictor_lstm.py
   python train_scene_conditioned.py
   # Reuse the provided GameFormer checkpoint
   ```
2. **Extract meta-features**
   ```bash
   python extract_meta_features_dataset.py \
     --gameformer_ckpt <path> \
     --scene_ckpt <path> \
     --lstm_ckpt <path> \
     --output ./eval_out/fusion_stats_with_scene_conditioned.json
   ```
3. **Train ranking gate**
   ```bash
   python gate_ranking.py \
     --data ./eval_out/fusion_stats_with_scene_conditioned.json \
     --epochs 50 --batch_size 64 --output_dir ./eval_out
   ```
4. **LLM-enhanced inference**
   - Provide `LLM_GATE_VOCAB` (semantic vocabulary) and `LLM_GATE_CACHE` (LLM label cache).
   - `llm_enhanced_gate.py` can run purely offline (LLM disabled) or call an API for semantic analysis.

## NuPlan Open-Loop Simulation
Use `scripts/run_llm_gate.sh` as a template and set:
```
export LLM_GATE_GAMEFORMER=.../model_epoch_18_valADE_1.7272.pth
export LLM_GATE_SCENE=.../SceneConditioned_v4/model_best_ADE_3.356.pth
export LLM_GATE_LSTM=.../LSTM_v2/best_seq2seq_lstm.pth
export LLM_GATE_RANKING=.../eval_out/tri_gate_v7.pth
export LLM_GATE_VOCAB=.../semantic_vocab.json
export LLM_GATE_CACHE=.../llm_semantic_cache.json
```
Then launch `nuplan.planning.script.run_simulation` with `planner=llm_gate_planner` and configure the scenario filters described in the paper.

### Evaluation & Analysis
- `compute_open_loop_metrics.py`, `compute_scene_conditioned_metrics.py` parse nuPlan msgpack logs into ADE/FDE summaries.
- `gate_ranking.py` reports ORR, expert ratios, and threshold sweeps to align with the manuscript tables.
- `agent.md` documents forced-expert runs and diagnostic scripts for the left-turn case study.

## Reproduction Tips
- **Oracle Realization**: inspect `eval_out/ranking_gate_results.json` (contains baseline, oracle, and gate FDE).
- **LLM triggers**: tune `trigger_threshold`, `min_lstm_prob`, `min_scene_prob`, and the `LLM_GATE_FORCE_EXPERT` env variable to match the study’s override strategy.
- **Long-tail slices**: use `analysis_runs/*.msgpack.xz` with `compute_open_loop_metrics.py` to evaluate specific scenario types.

## Citation
```bibtex
@article{lu2026dynamic,
  title   = {Dynamic Model Selection for Trajectory Prediction via Pairwise Ranking and Meta-Features},
  author  = {Lu, Bowen},
  journal = {AAAI Conference on Artificial Intelligence},
  year    = {2026}
}
```

## Contact
- Please open a GitHub Issue for bugs or feature requests.
- Direct questions to the author at `bluu0021@student.monash.edu`.

The documentation will continue to evolve as new experiments land—contributions that improve gating, semantic supervision, or planner integration are highly welcome.
