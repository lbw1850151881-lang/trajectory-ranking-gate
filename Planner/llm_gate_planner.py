import copy
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from shapely import Point

from .observation import observation_adapter
from .planner_utils import (
    TrajectoryPlanner,
    annotate_occupancy,
    annotate_speed,
    transform_to_ego_frame,
    wrap_to_pi,
)
from GameFormer.predictor import GameFormer
from train_predictor_lstm import Seq2SeqLSTM
from scene_conditioned_gameformer import SceneConditionedGameFormer, get_scene_type_mapping
from .state_lattice_path_planner import LatticePlanner

from common_utils import DT, MAX_LEN, T
from complexity_metrics import calculate_complexity
from meta_features_extractor import (
    detect_physics_violations,
    input_perturbation_stability,
    mc_forward_gameformer,
    mc_forward_lstm,
)
from tri_expert_model import EXPERT_LABELS, TriExpertModel
from tri_feature_utils import BASE_FEATURE_KEYS, DERIVED_FEATURE_KEYS, compute_additional_features
from llm_feature_extractor import LLMFeatureExtractor
from llm_supervisor import LLMSupervisor, SemanticFeatures

from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states


@dataclass
class PlanCandidate:
    """Container for a single expert plan."""

    plan_xyh: np.ndarray
    states: list
    ade: float
    fde: float
    path_length: float


class LLMGatePlanner(AbstractPlanner):
    """
    NuPlan-compatible planner that combines GameFormer, Scene-Conditioned GameFormer,
    and (optionally) an LSTM predictor via the learned LLM-enhanced gate.
    """

    requires_scenario = True

    def __init__(
        self,
        gameformer_path: str,
        scene_model_path: str,
        ranking_checkpoint: str,
        vocab_path: str,
        lstm_path: Optional[str] = None,
        device: Optional[str] = None,
        llm_cache: Optional[str] = "eval_out/llm_semantic_cache.json",
        mc_dropout_samples: int = 4,
        scenario: Optional[AbstractScenario] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self._device = torch.device(device)

        self._gameformer_path = gameformer_path
        self._transformer_path = scene_model_path
        self._lstm_path = lstm_path
        self._ranking_checkpoint_path = ranking_checkpoint
        self._vocab_path = vocab_path
        self._llm_cache = llm_cache
        self._mc_samples = mc_dropout_samples
        self._scene_type_mapping = get_scene_type_mapping()
        self._forced_expert = os.getenv("LLM_GATE_FORCE_EXPERT")

        self._map_api = None
        self._route_roadblocks = []
        self._route_roadblock_ids: List[str] = []
        self._candidate_lane_edge_ids = []
        self._trajectory_planner: Optional[TrajectoryPlanner] = None
        self._path_planner: Optional[LatticePlanner] = None
        self._scenario: Optional[AbstractScenario] = scenario
        self._scene_model: Optional[SceneConditionedGameFormer] = None

        self._load_gate_components()

    def _log_debug(self, message: str) -> None:
        try:
            timestamp = datetime.utcnow().isoformat()
            log_path = Path.home() / "llm_gate_debug.log"
            log_path.open("a").write(f"{timestamp} {message}\n")
        except OSError:
            pass

    # ------------------------------------------------------------------
    # NuPlan planner interface
    # ------------------------------------------------------------------

    def name(self) -> str:
        return "LLM Gate Planner"

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        if self._scenario is None:
            raise RuntimeError("LLM Gate Planner requires scenario context during construction.")
        self._map_api = initialization.map_api
        self._route_roadblock_ids = list(initialization.route_roadblock_ids)
        self._initialize_route_plan(self._route_roadblock_ids)
        self._initialize_models()
        self._trajectory_planner = TrajectoryPlanner(device=self._device)
        self._path_planner = LatticePlanner(self._candidate_lane_edge_ids, MAX_LEN)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        if self._trajectory_planner is None:
            raise RuntimeError("Planner not initialised.")

        iteration = current_input.iteration.index
        history = current_input.history
        ego_state, observation = history.current_state
        traffic_light_data = list(current_input.traffic_light_data)

        features = observation_adapter(
            history,
            traffic_light_data,
            self._map_api,
            self._route_roadblock_ids,
            self._device,
        )

        ref_path = self._get_reference_path(ego_state, traffic_light_data, observation)

        candidates: Dict[str, PlanCandidate] = {}
        feature_pool: Dict[str, float] = {}

        complexity_score, breakdown = calculate_complexity(
            features['ego_agent_past'].detach().cpu(),
            features['neighbor_agents_past'].detach().cpu(),
            features['map_lanes'].detach().cpu(),
            features['map_crosswalks'].detach().cpu(),
        )
        feature_pool['complexity_score'] = float(complexity_score)
        for key, value in breakdown.items():
            if key == 'weights':
                continue
            feature_pool[key] = float(value)

        gmf_candidate, gmf_meta = self._run_gameformer(features, ref_path, history)
        candidates["gameformer"] = gmf_candidate
        feature_pool.update(gmf_meta)

        semantic_features = None
        scene_type_tensor = None
        keyword_tensor = None
        if self._scene_model is not None:
            partial_candidates = {"gameformer": gmf_candidate}
            partial_map = self._assemble_feature_map(dict(feature_pool), partial_candidates)
            partial_context = self._build_scene_context(partial_map)
            semantic_features = self._llm_supervisor.extractor.extract_features(partial_context)
            scene_type_tensor, keyword_tensor = self._prepare_semantic_tensors(semantic_features)

        sc_candidate = None
        if self._scene_model is not None:
            sc_candidate, sc_meta = self._run_scene_conditioned(
                features,
                ref_path,
                history,
                scene_type_tensor,
                keyword_tensor,
                semantic_features,
            )
            if sc_candidate is not None:
                candidates["scene_conditioned"] = sc_candidate
                feature_pool.update(sc_meta)

        lstm_candidate, lstm_meta = self._run_lstm(features, history)
        if lstm_candidate is not None:
            candidates["lstm"] = lstm_candidate
            feature_pool.update(lstm_meta)

        gt_xy = self._get_ground_truth_xy(iteration)
        if gt_xy is not None:
            for cand in candidates.values():
                cand.ade, cand.fde = self._compute_metrics(cand.plan_xyh, gt_xy)

        feature_map = self._assemble_feature_map(feature_pool, candidates)
        scene_context = self._build_scene_context(feature_map)
        decision = self._select_expert(feature_map, scene_context, semantic_features=semantic_features)
        if self._forced_expert:
            forced = self._forced_expert.strip().lower()
            if forced in candidates and np.all(np.isfinite(candidates[forced].plan_xyh)):
                decision = forced
        final_decision = decision
        chosen = candidates.get(decision)
        if chosen is None:
            chosen = candidates.get("scene_conditioned") or candidates.get("gameformer")
            final_decision = "gameformer"
        if chosen is None:
            raise RuntimeError("No valid expert candidate available")
        scenario_name = self._scenario.scenario_name if self._scenario else "unknown"
        scenario_type = self._scenario.scenario_type if self._scenario else "unknown"
        iteration_index = getattr(iteration, "index", iteration)
        if not np.all(np.isfinite(chosen.plan_xyh)):
            self._log_debug(
                f"scenario={scenario_name} type={scenario_type} iteration={iteration_index} "
                f"non_finite_candidate decision={decision} fallback=gameformer"
            )
            fallback = candidates.get("gameformer")
            if fallback and np.all(np.isfinite(fallback.plan_xyh)):
                chosen = fallback
                final_decision = "gameformer"
            else:
                chosen.plan_xyh = np.nan_to_num(chosen.plan_xyh, nan=0.0, posinf=0.0, neginf=0.0)
                self._log_debug("sanitised_candidate_plan")
        else:
            self._log_debug(
                f"scenario={scenario_name} type={scenario_type} iteration={iteration_index} "
                f"decision={decision} ade={chosen.ade:.3f} fde={chosen.fde:.3f} "
                f"min_xy=({chosen.plan_xyh[:,0].min():.3f},{chosen.plan_xyh[:,1].min():.3f}) "
                f"max_xy=({chosen.plan_xyh[:,0].max():.3f},{chosen.plan_xyh[:,1].max():.3f})"
            )
        # Default path-quality safeguard: if the selected expert produces an implausibly short trajectory,
        # fall back to GameFormer which is the most stable expert online.
        gmf_candidate = candidates.get("gameformer")
        if chosen and gmf_candidate and final_decision != "gameformer":
            min_path = 5.0
            short_path = chosen.path_length < min_path or chosen.path_length < 0.6 * gmf_candidate.path_length
            final_delta = float(
                np.hypot(
                    chosen.plan_xyh[-1, 0] - gmf_candidate.plan_xyh[-1, 0],
                    chosen.plan_xyh[-1, 1] - gmf_candidate.plan_xyh[-1, 1],
                )
            )
            divergent_goal = final_delta > 12.0 and chosen.path_length < gmf_candidate.path_length * 1.1
            if short_path or divergent_goal:
                self._log_debug(
                    f"scenario={scenario_name} type={scenario_type} iteration={iteration_index} "
                    f"fallback_gameformer decision={final_decision} "
                    f"path_len={chosen.path_length:.2f} gmf_len={gmf_candidate.path_length:.2f} "
                    f"final_delta={final_delta:.2f}"
                )
                final_decision = "gameformer"
                chosen = gmf_candidate
        if chosen and gmf_candidate and final_decision == "scene_conditioned":
            blended_plan = self._blend_with_gameformer(gmf_candidate.plan_xyh, chosen.plan_xyh)
            states = self._plan_to_states(blended_plan, history)
            chosen = PlanCandidate(
                plan_xyh=blended_plan,
                states=states,
                ade=float("nan"),
                fde=float("nan"),
                path_length=self._compute_path_length(states),
            )
        return InterpolatedTrajectory(chosen.states)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_gate_components(self) -> None:
        checkpoint = torch.load(self._ranking_checkpoint_path, map_location="cpu")
        input_dim = checkpoint["input_dim"]
        hidden_dims = checkpoint.get("hidden_dims", [128, 64, 32])
        dropout = checkpoint.get("dropout", 0.3)
        self._gate_model = TriExpertModel(input_dim, hidden_dims, dropout).to(self._device)
        self._gate_model.load_state_dict(checkpoint["model_state_dict"])
        self._gate_model.eval()

        normalization = checkpoint.get("normalization", {})
        mean = normalization.get("mean", np.zeros(input_dim))
        std = normalization.get("std", np.ones(input_dim))
        self._feature_keys = normalization.get("feature_keys", BASE_FEATURE_KEYS + DERIVED_FEATURE_KEYS)
        self._feature_mean = torch.tensor(mean, dtype=torch.float32, device=self._device)
        self._feature_std = torch.tensor(std, dtype=torch.float32, device=self._device)

        self._llm_extractor = LLMFeatureExtractor(cache_path=self._llm_cache, offline=True)
        self._llm_supervisor = LLMSupervisor(
            extractor=self._llm_extractor,
            sc_hybrid_threshold=checkpoint.get("sc_hybrid_threshold", 0.45),
        )

        self._keyword_to_id: Dict[str, int] = {}
        vocab_path = Path(self._vocab_path)
        if vocab_path.exists():
            vocab_data = json.loads(vocab_path.read_text())
            self._keyword_to_id = vocab_data.get("keyword_to_id", {})
            self._vocab_size = int(vocab_data.get("vocab_size", len(self._keyword_to_id)))
        else:
            self._vocab_size = len(self._keyword_to_id)

    def _initialize_models(self) -> None:
        self._gameformer_model = GameFormer(encoder_layers=3, decoder_levels=2)
        gm_state = torch.load(self._gameformer_path, map_location=self._device)
        self._gameformer_model.load_state_dict(gm_state)
        self._gameformer_model.to(self._device).eval()

        self._scene_model = None
        if self._transformer_path and os.path.exists(self._transformer_path):
            try:
                vocab_size = max(len(self._keyword_to_id), 1)
                base_ckpt = self._gameformer_path
                # Initialise with base GameFormer weights to ensure architectural match.
                self._scene_model = SceneConditionedGameFormer.from_pretrained(
                    base_ckpt,
                    vocab_size=vocab_size,
                )
                scene_state = torch.load(self._transformer_path, map_location="cpu")
                if isinstance(scene_state, dict):
                    for key in ["model_state_dict", "state_dict", "model"]:
                        if key in scene_state and isinstance(scene_state[key], dict):
                            scene_state = scene_state[key]
                            break
                missing, unexpected = self._scene_model.load_state_dict(scene_state, strict=False)
                if missing or unexpected:
                    self._log_debug(
                        f"scene_conditioned_state_mismatch missing={len(missing)} unexpected={len(unexpected)}"
                    )
                self._scene_model.to(self._device).eval()
            except Exception as exc:  # pragma: no cover - protection for deployment
                self._scene_model = None
                self._log_debug(f"scene_conditioned_model_load_failed: {exc}")

        self._lstm_model = None
        if self._lstm_path and os.path.exists(self._lstm_path):
            self._lstm_model = Seq2SeqLSTM(
                input_dim=4,
                hidden_dim=256,
                num_layers=2,
                future_steps=int(T / DT),
                dropout=0.2,
            )
            lstm_state = torch.load(self._lstm_path, map_location="cpu")
            if isinstance(lstm_state, dict):
                if "state_dict" in lstm_state:
                    lstm_state = lstm_state["state_dict"]
                elif "model_state_dict" in lstm_state:
                    lstm_state = lstm_state["model_state_dict"]
                elif "model" in lstm_state and isinstance(lstm_state["model"], dict):
                    lstm_state = lstm_state["model"]
            self._lstm_model.load_state_dict(lstm_state)
            self._lstm_model.to(self._device).eval()

    def _initialize_route_plan(self, route_roadblock_ids):
        self._route_roadblocks = []
        for block_id in route_roadblock_ids:
            block = self._map_api.get_map_object(block_id, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(block_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            if block is not None:
                self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = [
            edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        ]

    # ------------------------------------------------------------------
    # Expert inference
    # ------------------------------------------------------------------

    def _run_gameformer(
        self,
        features: Dict[str, torch.Tensor],
        ref_path: Optional[np.ndarray],
        history,
    ) -> Tuple[PlanCandidate, Dict[str, float]]:
        with torch.no_grad():
            predictions, ego_plan = self._gameformer_model(features)
        ego_plan_np = ego_plan.detach().cpu().numpy()[0]
        if not np.all(np.isfinite(ego_plan_np)):
            ego_plan_np = np.nan_to_num(ego_plan_np, nan=0.0, posinf=0.0, neginf=0.0)
        # Extract final level interactions
        levels = len(predictions) // 2 - 1
        final_predictions = predictions[f"level_{levels}_interactions"][:, 1:]
        final_scores = predictions[f"level_{levels}_scores"]
        ego_state_transformed = features["ego_agent_past"][:, -1]
        neighbors_state_transformed = features["neighbor_agents_past"][:, :, -1]

        try:
            plan_xyh = self._trajectory_planner.plan(
                history.current_state[0],
                ego_state_transformed,
                neighbors_state_transformed,
                final_predictions,
                ego_plan,
                final_scores,
                ref_path,
                history.current_state[1],
            )
        except Exception:
            plan_xyh = ego_plan_np.copy()
        else:
            if not np.all(np.isfinite(plan_xyh)):
                plan_xyh = ego_plan_np.copy()
        plan_xyh = np.nan_to_num(plan_xyh, nan=0.0, posinf=0.0, neginf=0.0)
        states = self._plan_to_states(plan_xyh, history)
        candidate = PlanCandidate(
            plan_xyh=plan_xyh,
            states=states,
            ade=float("nan"),
            fde=float("nan"),
            path_length=self._compute_path_length(states),
        )

        meta = self._compute_gameformer_meta(features, plan_xyh)
        return candidate, meta

    def _run_scene_conditioned(
        self,
        features: Dict[str, torch.Tensor],
        ref_path: Optional[np.ndarray],
        history,
        scene_type_tensor: Optional[torch.Tensor],
        keyword_tensor: Optional[torch.Tensor],
        semantic_features: Optional[SemanticFeatures],
    ) -> Tuple[Optional[PlanCandidate], Dict[str, float]]:
        if self._scene_model is None:
            return None, {}

        inputs = {k: v.to(self._device) for k, v in features.items()}
        batch_size = next(iter(inputs.values())).shape[0]
        if scene_type_tensor is None:
            scene_type_tensor = torch.full(
                (batch_size,),
                self._scene_type_mapping.get("other", 0),
                dtype=torch.long,
                device=self._device,
            )
        if keyword_tensor is None:
            vocab = self._scene_model.encoder.semantic_embedding.vocab_size
            keyword_tensor = torch.zeros((batch_size, vocab), dtype=torch.float32, device=self._device)

        with torch.no_grad():
            decoder_outputs, ego_plan = self._scene_model(
                inputs,
                scene_type_ids=scene_type_tensor,
                keyword_vectors=keyword_tensor,
            )

        levels = len(decoder_outputs) // 2 - 1
        final_predictions = decoder_outputs[f"level_{levels}_interactions"][:, 1:]
        final_scores = decoder_outputs[f"level_{levels}_scores"]
        ego_state_transformed = inputs["ego_agent_past"][:, -1]
        neighbors_state_transformed = inputs["neighbor_agents_past"][:, :, -1]

        with torch.no_grad():
            plan = self._trajectory_planner.plan(
                history.current_state[0],
                ego_state_transformed,
                neighbors_state_transformed,
                final_predictions,
                ego_plan,
                final_scores,
                ref_path,
                history.current_state[1],
            )

        plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
        plan_xyh = plan.astype(np.float32)
        states = self._plan_to_states(plan_xyh, history)
        candidate = PlanCandidate(
            plan_xyh=plan_xyh,
            states=states,
            ade=float("nan"),
            fde=float("nan"),
            path_length=self._compute_path_length(states),
        )
        meta = self._compute_scene_conditioned_meta(inputs, plan_xyh)
        meta["scene_conditioned_scene_type"] = (
            semantic_features.scene_type if semantic_features is not None else "other"
        )
        return candidate, meta

    def _run_lstm(self, features: Dict[str, torch.Tensor], history) -> Tuple[Optional[PlanCandidate], Dict[str, float]]:
        if self._lstm_model is None:
            return None, {}

        ego_input = features["ego_agent_past"][..., :4]
        with torch.no_grad():
            pred_xy = self._lstm_model(ego_input).cpu().numpy()[0]

        if pred_xy.shape[0] < int(T / DT):
            pred_xy = np.pad(pred_xy, ((0, int(T / DT) - pred_xy.shape[0]), (0, 0)), mode="edge")
        headings = self._estimate_headings(pred_xy)
        plan_xyh = np.column_stack([pred_xy[:, 0], pred_xy[:, 1], headings]).astype(np.float32)

        states = self._plan_to_states(plan_xyh, history)
        candidate = PlanCandidate(
            plan_xyh=plan_xyh,
            states=states,
            ade=float("nan"),
            fde=float("nan"),
            path_length=self._compute_path_length(states),
        )
        meta = self._compute_lstm_meta(ego_input, plan_xyh)
        return candidate, meta

    # ------------------------------------------------------------------
    # Trajectory post-processing
    # ------------------------------------------------------------------

    def _plan_to_states(self, plan_xyh: np.ndarray, history) -> List:
        """
        Convert absolute planner output into a list of EgoStates with timestamps.
        """
        ego_state = history.current_state[0]
        relative_xy = plan_xyh[:, :2]
        # Headings produced by the lattice planner are already ego-centric.
        relative_heading = plan_xyh[:, 2]
        relative_plan = np.stack(
            [relative_xy[:, 0], relative_xy[:, 1], relative_heading],
            axis=-1,
        ).astype(np.float32)
        if len(plan_xyh) > 0:
            self._log_debug(
                f"plan_to_states abs_head={plan_xyh[0][:3]} rel0={relative_plan[0][:3]} "
                f"abs_max=({plan_xyh[:, 0].max():.3f},{plan_xyh[:, 1].max():.3f}) "
                f"rel_max=({relative_plan[:, 0].max():.3f},{relative_plan[:, 1].max():.3f})"
            )
        history_dt = getattr(history, "sample_interval", None)
        if history_dt is None and self._scenario is not None:
            history_dt = getattr(self._scenario, "database_interval", None)
        target_dt = float(history_dt) if history_dt else DT
        target_steps = max(int(round(T / target_dt)), 1)
        if len(relative_plan) != target_steps:
            relative_plan = self._resample_plan(relative_plan, target_steps)

        clamp_horizon = getattr(self, "_evaluation_horizon", None)
        if clamp_horizon is None:
            clamp_horizon = 4.0
        clamp_horizon = max(min(float(clamp_horizon), float(T)), target_dt)
        clamp_steps = max(int(round(clamp_horizon / target_dt)), 1)
        if len(relative_plan) > clamp_steps:
            relative_plan = relative_plan[:clamp_steps]

        states = transform_predictions_to_states(
            relative_plan,
            history.ego_states,
            clamp_horizon,
            target_dt,
            include_ego_state=False,
        )
        states.insert(0, ego_state)
        return states

    @staticmethod
    def _resample_plan(plan_xyh: np.ndarray, target_steps: int) -> np.ndarray:
        """
        Resample a [N, 3] relative plan to match the number of steps expected by NuPlan logs.
        """
        if plan_xyh.shape[0] == 0:
            return plan_xyh
        if plan_xyh.shape[0] == target_steps:
            return plan_xyh.astype(np.float32, copy=False)

        original_steps = plan_xyh.shape[0]
        original_times = np.linspace(T / original_steps, T, original_steps, dtype=np.float32)
        target_times = np.arange(1, target_steps + 1, dtype=np.float32) * (T / target_steps)
        # guard against potential floating round-off so we never exceed horizon
        target_times = np.clip(target_times, 0.0, T)

        x_interp = np.interp(target_times, original_times, plan_xyh[:, 0])
        y_interp = np.interp(target_times, original_times, plan_xyh[:, 1])

        sin_heading = np.sin(plan_xyh[:, 2])
        cos_heading = np.cos(plan_xyh[:, 2])
        sin_interp = np.interp(target_times, original_times, sin_heading)
        cos_interp = np.interp(target_times, original_times, cos_heading)
        headings = np.arctan2(sin_interp, cos_interp).astype(np.float32)

        resampled = np.column_stack((x_interp.astype(np.float32), y_interp.astype(np.float32), headings))
        return resampled

    @staticmethod
    def _compute_path_length(states: List) -> float:
        total = 0.0
        for prev, curr in zip(states[:-1], states[1:]):
            total += float(
                np.hypot(curr.center.x - prev.center.x, curr.center.y - prev.center.y)
            )
        return total

    def _prepare_semantic_tensors(
        self, semantic: Optional[SemanticFeatures]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = 1
        scene_idx = self._scene_type_mapping.get("other", 0)
        if semantic is not None:
            scene_idx = self._scene_type_mapping.get(semantic.scene_type, scene_idx)
        scene_tensor = torch.tensor([scene_idx], dtype=torch.long, device=self._device)

        vocab = self._scene_model.encoder.semantic_embedding.vocab_size if self._scene_model is not None else len(self._keyword_to_id)
        keyword_tensor = torch.zeros((batch_size, vocab), dtype=torch.float32, device=self._device)
        if semantic is not None:
            for kw in semantic.semantic_keywords:
                idx = self._keyword_to_id.get(kw)
                if idx is not None and idx < vocab:
                    keyword_tensor[0, idx] = 1.0
        return scene_tensor, keyword_tensor

    @staticmethod
    def _blend_with_gameformer(base_plan: np.ndarray, scene_plan: np.ndarray) -> np.ndarray:
        length = min(len(base_plan), len(scene_plan))
        if length == 0:
            return base_plan
        tail = 1
        weights = np.zeros(length, dtype=np.float32)
        if tail == 1:
            weights[-1] = 1.0
        else:
            weights[-tail:] = np.linspace(0.0, 1.0, tail, dtype=np.float32) ** 1.5
        blended = base_plan[:length].copy()
        delta_xy = scene_plan[:length, :2] - base_plan[:length, :2]
        blended[:length, :2] = base_plan[:length, :2] + weights[:, None] * delta_xy

        delta_heading = wrap_to_pi(scene_plan[:length, 2] - base_plan[:length, 2])
        blended[:length, 2] = wrap_to_pi(base_plan[:length, 2] + weights * delta_heading)
        return blended

    # ------------------------------------------------------------------
    # Meta feature extraction
    # ------------------------------------------------------------------

    def _compute_gameformer_meta(self, features: Dict[str, torch.Tensor], plan_xyh: np.ndarray) -> Dict[str, float]:
        meta: Dict[str, float] = {}
        gmf_inputs = {k: v.clone() for k, v in features.items()}

        with torch.no_grad():
            _, unc_total, unc_final = mc_forward_gameformer(self._gameformer_model, gmf_inputs, K=self._mc_samples)
        stability = input_perturbation_stability(self._gameformer_model, gmf_inputs, model_type="gameformer")

        plan_tensor = torch.from_numpy(plan_xyh[:, :2]).unsqueeze(0)
        violations = detect_physics_violations(plan_tensor)

        meta.update(
            {
                "gmf_uncertainty_total": float(unc_total[0]),
                "gmf_uncertainty_final": float(unc_final[0]),
                "gmf_stability": float(stability[0]),
                "gmf_physics_violation": float(violations[0]),
            }
        )
        return meta

    def _compute_scene_conditioned_meta(
        self, model_inputs: Dict[str, torch.Tensor], plan_xyh: np.ndarray
    ) -> Dict[str, float]:
        meta: Dict[str, float] = {}
        if self._scene_model is None:
            zero = 0.0
            meta.update(
                {
                    "scene_conditioned_uncertainty_total": zero,
                    "scene_conditioned_uncertainty_final": zero,
                    "scene_conditioned_stability": 1.0,
                    "scene_conditioned_physics_violation": zero,
                }
            )
            return meta

        with torch.no_grad():
            _, unc_total, unc_final = mc_forward_gameformer(self._scene_model, model_inputs, K=self._mc_samples)
        stability = input_perturbation_stability(self._scene_model, model_inputs, model_type="gameformer")
        plan_tensor = torch.from_numpy(plan_xyh[:, :2]).unsqueeze(0)
        violations = detect_physics_violations(plan_tensor)

        meta.update(
            {
                "scene_conditioned_uncertainty_total": float(unc_total[0]),
                "scene_conditioned_uncertainty_final": float(unc_final[0]),
                "scene_conditioned_stability": float(stability[0]),
                "scene_conditioned_physics_violation": float(violations[0]),
            }
        )
        return meta

    def _compute_lstm_meta(self, lstm_input: torch.Tensor, plan_xyh: np.ndarray) -> Dict[str, float]:
        meta: Dict[str, float] = {}
        if self._lstm_model is None:
            zero = 0.0
            meta.update(
                {
                    "lstm_uncertainty_total": zero,
                    "lstm_uncertainty_final": zero,
                    "lstm_stability": 1.0,
                    "lstm_physics_violation": zero,
                }
            )
            return meta

        with torch.no_grad():
            _, unc_total, unc_final = mc_forward_lstm(self._lstm_model, lstm_input, K=self._mc_samples)
        stability = input_perturbation_stability(self._lstm_model, lstm_input, model_type="lstm")
        plan_tensor = torch.from_numpy(plan_xyh[:, :2]).unsqueeze(0)
        violations = detect_physics_violations(plan_tensor)

        meta.update(
            {
                "lstm_uncertainty_total": float(unc_total[0]),
                "lstm_uncertainty_final": float(unc_final[0]),
                "lstm_stability": float(stability[0]),
                "lstm_physics_violation": float(violations[0]),
            }
        )
        return meta

    # ------------------------------------------------------------------
    # Feature assembly and gating
    # ------------------------------------------------------------------

    def _assemble_feature_map(self, meta: Dict[str, float], candidates: Dict[str, PlanCandidate]) -> Dict[str, float]:
        # Complexity and environment metrics
        feature_map = {k: v for k, v in meta.items() if k in BASE_FEATURE_KEYS or k in DERIVED_FEATURE_KEYS}
        if "scene_conditioned_scene_type" in meta:
            feature_map["scene_conditioned_scene_type"] = meta["scene_conditioned_scene_type"]

        # Complexity metrics might already be present; ensure defaults
        for key in BASE_FEATURE_KEYS:
            feature_map.setdefault(key, 0.0)

        # Add ADE/FDE for experts if computed
        for expert in ["lstm", "gameformer", "scene_conditioned"]:
            cand = candidates.get(expert)
            if cand is None:
                continue
            feature_map[f"{expert}_ade"] = float(cand.ade if not math.isnan(cand.ade) else 0.0)
            feature_map[f"{expert}_fde"] = float(cand.fde if not math.isnan(cand.fde) else 0.0)

        # Relative meta features
        lstm_unc = feature_map.get("lstm_uncertainty_total", 0.0)
        gmf_unc = feature_map.get("gmf_uncertainty_total", 0.0)
        if lstm_unc < 1e-8:
            feature_map["uncertainty_ratio"] = float(gmf_unc)
        else:
            feature_map["uncertainty_ratio"] = float(gmf_unc / (lstm_unc + 1e-8))

        lstm_stab = feature_map.get("lstm_stability", 1.0)
        gmf_stab = feature_map.get("gmf_stability", 1.0)
        feature_map["stability_ratio"] = float(gmf_stab / (lstm_stab + 1e-6))
        feature_map["stability_diff"] = float(gmf_stab - lstm_stab)

        lstm_violation = feature_map.get("lstm_physics_violation", 0.0)
        gmf_violation = feature_map.get("gmf_physics_violation", 0.0)
        feature_map["violation_diff"] = float(gmf_violation - lstm_violation)

        sc_unc = feature_map.get("scene_conditioned_uncertainty_total", None)
        if sc_unc is not None:
            lstm_unc = feature_map.get("lstm_uncertainty_total", 0.0)
            gmf_unc = feature_map.get("gmf_uncertainty_total", 0.0)
            feature_map["scene_conditioned_uncertainty_ratio_lstm"] = float(sc_unc / (lstm_unc + 1e-6)) if lstm_unc > 0 else float(sc_unc)
            feature_map["scene_conditioned_uncertainty_ratio_gmf"] = float(sc_unc / (gmf_unc + 1e-6)) if gmf_unc > 0 else float(sc_unc)
            sc_stability = feature_map.get("scene_conditioned_stability", 1.0)
            feature_map["scene_conditioned_stability_ratio_lstm"] = float(sc_stability / (lstm_stab + 1e-6))
            feature_map["scene_conditioned_stability_ratio_gmf"] = float(sc_stability / (gmf_stab + 1e-6))
            sc_violation = feature_map.get("scene_conditioned_physics_violation", 0.0)
            feature_map["scene_conditioned_violation_diff_lstm"] = float(sc_violation - lstm_violation)
            feature_map["scene_conditioned_violation_diff_gmf"] = float(sc_violation - gmf_violation)

        feature_map.update(compute_additional_features(feature_map))
        return feature_map

    def _build_scene_context(self, feature_map: Dict[str, float]) -> Dict[str, float]:
        context = {
            "complexity_score": feature_map.get("complexity_score", 0.0),
            "n_neighbors": feature_map.get("n_neighbors", 0.0),
            "neighbor_score": feature_map.get("neighbor_score", 0.0),
            "speed_variance": feature_map.get("speed_variance", 0.0),
            "velocity_score": feature_map.get("velocity_score", 0.0),
            "min_neighbor_dist": feature_map.get("min_neighbor_dist", 100.0),
            "n_crosswalks": feature_map.get("n_crosswalks", 0.0),
            "crosswalk_score": feature_map.get("crosswalk_score", 0.0),
            "gmf_uncertainty_total": feature_map.get("gmf_uncertainty_total", 0.0),
            "lstm_uncertainty_total": feature_map.get("lstm_uncertainty_total", 0.0),
            "scene_conditioned_fde": feature_map.get("scene_conditioned_fde", 0.0),
            "gameformer_fde": feature_map.get("gameformer_fde", 0.0),
            "lstm_fde": feature_map.get("lstm_fde", 0.0),
        }
        if "scene_conditioned_scene_type" in feature_map:
            context["scene_conditioned_scene_type"] = feature_map["scene_conditioned_scene_type"]
        return context

    def _select_expert(
        self,
        feature_map: Dict[str, float],
        scene_context: Dict[str, float],
        *,
        semantic_features: Optional[SemanticFeatures] = None,
    ) -> str:
        feature_vector = [float(feature_map.get(key, 0.0)) for key in self._feature_keys]
        feature_tensor = torch.tensor(feature_vector, dtype=torch.float32, device=self._device)
        feature_tensor = (feature_tensor - self._feature_mean) / (self._feature_std + 1e-8)

        with torch.no_grad():
            logits = self._gate_model(feature_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        base_idx = int(np.argmax(probs))
        base_decision = EXPERT_LABELS[base_idx]
        base_prob = float(probs[base_idx])
        gmf_prob = float(probs[EXPERT_LABELS.index("gameformer")])
        scene_prob = float(probs[EXPERT_LABELS.index("scene_conditioned")])

        # Confidence adjustments
        if base_decision == "lstm" and base_prob < 0.95:
            base_decision = "gameformer"
        scene_margin = 0.15
        if base_decision == "scene_conditioned" and (
            base_prob < 0.6 or scene_prob < gmf_prob + scene_margin
        ):
            base_decision = "gameformer"
            base_prob = gmf_prob

        supervisor_decision = base_decision
        ranking_confidence = 1.0 - base_prob

        llm_decision = self._llm_supervisor.analyze_scene(
            scene_context,
            base_decision=base_decision,
            semantic_features=semantic_features,
        )
        if llm_decision.override:
            supervisor_decision = llm_decision.recommended_model

        if supervisor_decision not in {"gameformer", "scene_conditioned", "lstm"}:
            supervisor_decision = base_decision

        return supervisor_decision

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _get_reference_path(self, ego_state, traffic_light_data, observation):
        starting_block = None
        closest_distance = math.inf
        ego_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)

        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(ego_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance
            if math.isclose(closest_distance, 0.0):
                break

        if closest_distance > 5 or starting_block is None:
            return None

        ref_path = self._path_planner.plan(ego_state, starting_block, observation, traffic_light_data)
        if ref_path is None:
            return None

        occupancy = np.zeros((ref_path.shape[0], 1))
        for tl in traffic_light_data:
            lane_id = str(tl.lane_connector_id)
            if tl.status == TrafficLightStatusType.RED and lane_id in self._candidate_lane_edge_ids:
                lane = self._map_api.get_map_object(lane_id, SemanticMapLayer.LANE_CONNECTOR)
                if lane:
                    conn_path = np.array([[p.x, p.y] for p in lane.baseline_path.discrete_path])
                    red_path = transform_to_ego_frame(conn_path, ego_state)
                    occupancy = annotate_occupancy(occupancy, ref_path, red_path)

        target_speed = starting_block.interior_edges[0].speed_limit_mps or 13.0
        target_speed = np.clip(target_speed, 3.0, 15.0)
        max_speed = annotate_speed(ref_path, target_speed)
        ref_path = np.concatenate([ref_path, max_speed, occupancy], axis=-1)

        if len(ref_path) < MAX_LEN * 10:
            ref_path = np.append(ref_path, np.repeat(ref_path[np.newaxis, -1], MAX_LEN * 10 - len(ref_path), axis=0), axis=0)

        return ref_path.astype(np.float32)

    def _get_ground_truth_xy(self, iteration: int) -> Optional[np.ndarray]:
        if self._scenario is None:
            return None
        future = list(self._scenario.get_ego_future_trajectory(iteration, T))
        if not future:
            return None
        xy = np.array([[state.rear_axle.x, state.rear_axle.y] for state in future], dtype=np.float32)
        desired_len = int(T / DT)
        if len(xy) < desired_len:
            xy = np.pad(xy, ((0, desired_len - len(xy)), (0, 0)), mode="edge")
        else:
            xy = xy[:desired_len]
        return xy

    def _compute_metrics(self, plan_xyh: np.ndarray, gt_xy: np.ndarray) -> Tuple[float, float]:
        length = min(len(plan_xyh), len(gt_xy))
        if length == 0:
            return float("nan"), float("nan")
        diff = plan_xyh[:length, :2] - gt_xy[:length]
        dists = np.linalg.norm(diff, axis=1)
        ade = float(np.mean(dists))
        fde = float(dists[-1])
        return ade, fde

    @staticmethod
    def _estimate_headings(xy: np.ndarray) -> np.ndarray:
        velocities = np.diff(xy, axis=0, prepend=xy[:1])
        headings = np.arctan2(velocities[:, 1], velocities[:, 0])
        return headings.astype(np.float32)
