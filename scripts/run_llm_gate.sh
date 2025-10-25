#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/hamster/GameFormer-Planner:/mnt/d/paper/nuplan/nuplan-devkit"
export LLM_GATE_GAMEFORMER="/home/hamster/GameFormer-Planner/pretrained_models/GameFormer-Planner/training_log/Exp1/model_epoch_18_valADE_1.7272.pth"
export LLM_GATE_SCENE="/home/hamster/GameFormer-Planner/training_log/SceneConditioned_v4/model_best_ADE_3.356.pth"
export LLM_GATE_LSTM="/home/hamster/GameFormer-Planner/training_log/LSTM_v2/best_seq2seq_lstm.pth"
export LLM_GATE_RANKING="/home/hamster/GameFormer-Planner/eval_out/tri_gate_v7.pth"
export LLM_GATE_VOCAB="/home/hamster/GameFormer-Planner/clusters_200/semantic_vocab.json"
export LLM_GATE_CACHE="/home/hamster/GameFormer-Planner/eval_out/llm_semantic_cache.json"
export NUPLAN_DATA_ROOT="/mnt/d/paper/nuplan"
export NUPLAN_MAPS_ROOT="/mnt/d/paper/nuplan/nuplan-maps-v1.0/maps"
export NUPLAN_EXP_ROOT="/home/hamster/GameFormer-Planner/testing_log"

/home/hamster/miniconda3/bin/conda run -n nuplan python \
  /mnt/d/paper/nuplan/nuplan-devkit/nuplan/planning/script/run_simulation.py \
  experiment=open_loop_boxes \
  planner=llm_gate_planner \
  ego_controller=perfect_tracking_controller \
  observation=box_observation \
  worker=sequential \
  job_name=open_loop_llm_gate_scene_v4_resample \
  scenario_builder=nuplan_mini \
  scenario_filter.log_names="['2021.05.25.14.16.10_veh-35_01690_02183','2021.05.12.23.36.44_veh-35_01133_01535','2021.05.12.23.36.44_veh-35_00152_00504']" \
  scenario_filter.scenario_types="['changing_lane','starting_left_turn','starting_right_turn']" \
  scenario_filter.limit_total_scenarios=3 \
  scenario_filter.shuffle=false \
  scenario_filter.remove_invalid_goals=false
