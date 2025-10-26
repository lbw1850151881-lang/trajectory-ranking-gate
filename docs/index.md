---
title: Dynamic Model Selection for Trajectory Prediction
description: Pairwise-ranking tri-expert gating with LLM supervision for nuPlan-mini.
---

# Dynamic Model Selection for Trajectory Prediction

> Reference site for the AAAI 2026 manuscript **“Dynamic Model Selection for Trajectory Prediction via Pairwise Ranking and Meta-Features.”** The repo exposes the full stack needed to reproduce the reported improvements on nuPlan-mini: expert models, meta-feature extraction, RankNet gate training, and LLM-supervised deployment.

## Why This Project
Recent trajectory predictors excel on average yet fail in safety-critical long-tail scenes. Analysis on 1,287 nuPlan-mini samples shows a **0.464 m FDE oracle gap (16.4%)** between the best single expert (GameFormer) and a tri-expert oracle. Bridging that gap requires:
- Adaptive expert selection per scene.
- High-signal meta-features derived from model internals instead of coarse geometry.
- Semantic reasoning to catch ambiguous or risky situations.

## Method Snapshot
1. **Tri-Expert Ensemble**
   - *Physics expert*: Seq2Seq LSTM + Kalman Filter for dynamically consistent plans.
   - *Scene-conditioned transformer*: excels on interaction-heavy, long-tail cases.
   - *GameFormer*: fine-tuned generalist covering 68.7% of scenes.
2. **Meta-Features**
   - MC Dropout uncertainty, input perturbation stability, physics-violation rates, plus ratios/differences across experts.
3. **Pairwise Ranking Gate**
   - RankNet loss decides which expert should win; threshold search maximizes Oracle Realization Rate (ORR).
4. **LLM Supervisor**
   - Confidence and semantic triggers (34% of scenes) request reasoning overrides; optional verification backs off to physics expert when risk is high.

## Pipeline Overview
```
nuPlan data ──> train experts ──> extract meta-features
             └─> gate training ──> LLM-enhanced deployment ──> open-loop evaluation
```

**Key modules**
- `meta_features_extractor.py`, `extract_meta_features_dataset.py` – feature corpus.
- `gate_ranking.py` – RankNet training (`ranking_gate.pth`, `ranking_gate_results.json`).
- `llm_enhanced_gate.py`, `Planner/llm_gate_planner.py` – inference/runtime integration.
- `scripts/run_llm_gate.sh` – template command for nuPlan open-loop simulations.

## Experimental Highlights
| Model | ADE↓ (m) | FDE↓ (m) | ORR |
| --- | --- | --- | --- |
| GameFormer (best single expert) | 1.469 ± 0.02 | 2.835 ± 0.04 | 0% |
| **LLM-enhanced tri-expert gate** | **1.255 ± 0.02** | **2.567 ± 0.03** | **57.8%** |

- Left-turn scenario `c1bf7374695a5c47`: FDE drops from 18.17 m to 16.37 m.
- Scene-conditioned expert selected 25.3% of the time; LSTM covers 6.0% as a physics fail-safe.

## Reproduce the Study
1. **Environment** – Python ≥3.10, PyTorch ≥2.1 (CUDA), nuPlan-devkit, `pip install -r requirements.txt` (includes `theseus-ai==0.1.3`).
2. **Data** – download nuPlan-mini DB/maps and set `NUPLAN_DATA_ROOT`, `NUPLAN_MAPS_ROOT`, `NUPLAN_EXP_ROOT`.
3. **Experts** – run `train_predictor_lstm.py`, `train_scene_conditioned.py`, and reuse the provided GameFormer checkpoint.
4. **Meta-features + Gate** – execute `extract_meta_features_dataset.py` then `gate_ranking.py`.
5. **Planner** – configure `LLM_GATE_*` env vars (expert ckpts, ranking gate, semantic vocab/cache) and run `scripts/run_llm_gate.sh` with `planner=llm_gate_planner`.
6. **Evaluation** – parse open-loop logs using `compute_open_loop_metrics.py`; inspect `eval_out/ranking_gate_results.json` for ORR.

## Roadmap
- Fix state-lattice horizon truncation so 8 s clipping no longer inflates FDE.
- Fine-tune scene-conditioned expert on difficult left-turn slices with longer histories.
- Apply semantic-aware class weighting in the gate to emphasize specialist usage.
- Expand logging (`analysis_runs/*.msgpack.xz`) for quicker regression tracking.

## Citation
```bibtex
@article{lu2026dynamic,
  title   = {Dynamic Model Selection for Trajectory Prediction via Pairwise Ranking and Meta-Features},
  author  = {Lu, Bowen},
  journal = {AAAI Conference on Artificial Intelligence},
  year    = {2026}
}
```

Questions? Open an issue or email `bluu0021@student.monash.edu`.
