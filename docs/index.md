---
title: Dynamic Model Selection for Trajectory Prediction
description: LLM-enhanced tri-expert gating for reliable autonomous driving forecasts.
---

# Dynamic Model Selection for Trajectory Prediction

> A research-grade implementation of the AAAI 2026 submission “Dynamic Model Selection for Trajectory Prediction via Pairwise Ranking and Meta-Features”. We release the complete pipeline—experts, meta-features, ranking gate, and LLM supervision—to close the oracle gap on **nuPlan-mini**.

## Motivation
Autonomous driving predictors excel on average yet remain brittle in long-tail scenarios (dense intersections, occlusions, cut-ins). Our analysis of 1,287 nuPlan-mini scenes reveals a **0.464 m FDE oracle gap (16.4%)** between the best single expert (GameFormer) and a hypothetical tri-expert oracle. Closing this gap demands:

- **Scene-adaptive expert selection** instead of monolithic models.
- **Error-correlated signals** beyond handcrafted geometry.
- **Semantic awareness** to reason about ambiguous scenes.

## Method At A Glance
1. **Tri-Expert Ensemble**  
   - *Physics expert*: Seq2Seq LSTM + Kalman Filter for dynamically feasible plans.  
   - *Interaction expert*: Scene-conditioned Transformer specialising in long-tail traffic.  
   - *Workhorse*: Fine-tuned GameFormer covering 68.7% of scenes.
2. **Meta-Feature Extraction**  
   - MC Dropout uncertainty (`total`, `final`), input perturbation stability, physics violation scores.  
   - Ratios and differences across experts amplify discriminative power.
3. **Pairwise Ranking Gate**  
   - RankNet loss learns relative superiority (`lstm` vs `gameformer` vs `scene_conditioned`).  
   - Threshold search maximises oracle realisation while maintaining stability.
4. **LLM Supervisor**  
   - Confidence- and semantics-triggered overrides (34% scenes) inject long-horizon reasoning.  
   - Optional verification layer downgrades risky outputs to physics expert.

## Pipeline
```
nuPlan data → expert training → meta-feature extraction
            → ranking gate training → LLM-enhanced deployment
                                     ↘ open-loop nuPlan evaluation
```

Key components:
- `meta_features_extractor.py` and `extract_meta_features_dataset.py` build the feature corpus.
- `gate_ranking.py` trains the RankNet-style gate (`ranking_gate.pth`, `ranking_gate_results.json`).
- `llm_enhanced_gate.py` fuses logits with semantic cues and powers `Planner/llm_gate_planner.py`.
- `scripts/run_llm_gate.sh` reproduces the reported nuPlan open-loop runs.

## Experimental Highlights
| Model | ADE↓ (m) | FDE↓ (m) | Oracle Realization |
| --- | --- | --- | --- |
| GameFormer (best single expert) | 1.469 ± 0.02 | 2.835 ± 0.04 | 0% |
| **LLM-enhanced tri-expert gate** | **1.255 ± 0.02** | **2.567 ± 0.03** | **57.8%** |

- Open-loop left-turn (`c1bf7374695a5c47`): FDE reduced from 18.17 m to 16.37 m.
- Scene-conditioned expert receives 25.3% of traffic-heavy scenes; LSTM safeguards low-risk 6.0%.
- ORR metric quantifies closed oracle gap and is logged for every training run.

## Reproduction Checklist
1. **Environment**: Python ≥ 3.10, PyTorch ≥ 2.1, CUDA GPU, nuPlan-devkit; install `theseus-ai==0.1.3`.
2. **Data**: Download nuPlan-mini DB + maps, set `NUPLAN_DATA_ROOT`, `NUPLAN_MAPS_ROOT`, `NUPLAN_EXP_ROOT`.
3. **Train / Import Experts**:
   ```bash
   python train_predictor_lstm.py
   python train_scene_conditioned.py
   # GameFormer checkpoint from pretrained_models/
   ```
4. **Meta-Features & Gate**:
   ```bash
   python extract_meta_features_dataset.py --output ./eval_out/fusion_stats_with_scene_conditioned.json
   python gate_ranking.py --data ./eval_out/fusion_stats_with_scene_conditioned.json
   ```
5. **LLM-Enhanced Planner**: set `LLM_GATE_*` env vars (experts, ranking ckpt, semantic vocab/cache) and run `scripts/run_llm_gate.sh` to launch `planner=llm_gate_planner`.
6. **Evaluation**: parse nuPlan logs with `compute_open_loop_metrics.py` / `compute_scene_conditioned_metrics.py`; inspect `eval_out/ranking_gate_results.json` for ORR.

## Ongoing Work
- **Trajectory horizon alignment** for the state-lattice planner to remove residual truncation artifacts.
- **Scene-conditioned fine-tuning** on problematic left-turn slices with extended observation windows.
- **Gate calibration** via class-weighting and semantic-aware thresholds to favour specialists when justified.
- **Enhanced logging** (`analysis_runs/*.msgpack.xz`) for rapid regression tracking.

## Citation
```bibtex
@article{lu2026dynamic,
  title   = {Dynamic Model Selection for Trajectory Prediction via Pairwise Ranking and Meta-Features},
  author  = {Lu, Bowen},
  journal = {AAAI Conference on Artificial Intelligence},
  year    = {2026}
}
```

For questions, please open an issue or reach out to `bluu0021@student.monash.edu`.
