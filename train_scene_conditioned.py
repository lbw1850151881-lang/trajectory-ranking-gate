"""
场景条件专家训练脚本 (Scene-Conditioned Expert Training)
====================================================

目标：在LLM标注的数据上训练场景条件GameFormer

训练策略：
1. 从预训练GameFormer初始化
2. 加载LLM标注的语义标签
3. 在困难样本上过采样训练
4. 联合优化预测精度和语义对齐

数据流：
原始数据 + LLM标注 → 场景条件embedding → 改进的预测
"""

import os
import re
import glob
import json
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pathlib import Path
import logging
import argparse
from collections import defaultdict, Counter
from typing import Dict, Optional, List, Tuple

from scene_conditioned_gameformer import (
    SceneConditionedGameFormer,
    encode_scene_semantic,
    get_scene_type_mapping
)
from GameFormer.train_utils import DrivingData, motion_metrics

# 确保torch已导入（用于数据转换）
import torch.nn.functional as F


# ============================================================================
# 简化的metric计算（仅计算ego指标）
# ============================================================================

def compute_ego_metrics(plan_trajectory, ego_future):
    """
    计算ego车辆的预测指标
    
    Args:
        plan_trajectory: [B, T, 3] ego预测轨迹 (x, y, heading)
        ego_future: [B, T, 3] ego真实轨迹
    
    Returns:
        (ade, fde) - 平均位移误差和最终位移误差
    """
    # 计算位移误差
    distance = torch.norm(plan_trajectory[:, :, :2] - ego_future[:, :, :2], dim=-1)
    
    # ADE: 平均位移误差
    ade = torch.mean(distance).item()
    
    # FDE: 最终位移误差
    fde = torch.mean(distance[:, -1]).item()
    
    return ade, fde


def parse_data_patterns(data_root_arg: str) -> List[str]:
    """Parse data root argument into a list of glob patterns or paths."""
    if not data_root_arg:
        return []
    normalized = data_root_arg.replace(';', ' ').replace(',', ' ')
    patterns = [token.strip() for token in normalized.split() if token.strip()]
    return patterns


def parse_orig_aug_ratio(arg: Optional[str]) -> Optional[Tuple[float, float]]:
    """Parse orig:aug ratio string like '4:1' into floats."""
    if not arg:
        return None
    cleaned = arg.replace(',', ':').strip()
    if ':' not in cleaned:
        print(f"Warning: invalid orig:aug ratio '{arg}', expected format like 4:1")
        return None
    left, right = cleaned.split(':', 1)
    try:
        orig = float(left)
        aug = float(right)
    except ValueError:
        print(f"Warning: invalid orig:aug ratio '{arg}', could not parse numbers")
        return None
    if orig < 0 or aug < 0:
        print(f"Warning: orig:aug ratio must be non-negative, got {orig}:{aug}")
        return None
    return (orig, aug)


def parse_unfreeze_modules(arg: Optional[str]) -> List[str]:
    if not arg:
        return []
    modules = []
    for token in arg.split(','):
        name = token.strip()
        if name:
            modules.append(name)
    return modules


def set_batchnorm_eval(module: nn.Module) -> None:
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


# ============================================================================
# 自定义collate函数（处理batch中不同大小的数据）
# ============================================================================

def custom_collate_fn(batch):
    """
    自定义collate函数，将batch数据整理成正确的格式
    
    Args:
        batch: list of dicts, 每个dict包含 'sample', 'scene_type_id', 'keyword_vector', 'label'
    
    Returns:
        collated batch
    """
    # 分离各个部分
    samples = [item['sample'] for item in batch]
    scene_type_ids = torch.stack([item['scene_type_id'] for item in batch])
    keyword_vectors = torch.stack([item['keyword_vector'] for item in batch])
    labels = [item['label'] for item in batch]
    variant_types = [item.get('variant_type', 'base') for item in batch]
    
    # 将sample中的各个字段stack起来
    batch_sample = {
        'ego_agent_past': torch.stack([s['ego_agent_past'] for s in samples]),
        'neighbor_agents_past': torch.stack([s['neighbor_agents_past'] for s in samples]),
        'map_lanes': torch.stack([s['map_lanes'] for s in samples]),
        'map_crosswalks': torch.stack([s['map_crosswalks'] for s in samples]),
        'route_lanes': torch.stack([s['route_lanes'] for s in samples]),
        'ego_agent_future': torch.stack([s['ego_agent_future'] for s in samples]),
        'neighbor_agents_future': torch.stack([s['neighbor_agents_future'] for s in samples]),
    }
    
    return {
        'sample': batch_sample,
        'scene_type_id': scene_type_ids,
        'keyword_vector': keyword_vectors,
        'label': labels,
        'variant_type': variant_types
    }


# ============================================================================
# 场景条件数据集
# ============================================================================

class SceneConditionedDataset(Dataset):
    """
    场景条件数据集
    
    在原始DrivingData基础上添加语义标签
    """
    
    def __init__(
        self,
        data_patterns: List[str],
        labels_path: str,
        vocab_path: str,
        num_neighbors: int = 10,
        scene_weights: Optional[Dict[str, float]] = None,
        orig_aug_ratio: Optional[Tuple[float, float]] = None,
    ):
        """
        Args:
            data_patterns: 数据路径或glob模式列表（第一个为主数据，后续为增强数据）
            labels_path: LLM标注文件路径
            vocab_path: 语义词表路径
            num_neighbors: 邻居车辆数量
            orig_aug_ratio: 原始与增强样本的采样比例 (orig, aug)
        """
        if not data_patterns:
            raise ValueError("data_patterns must contain at least one path or glob pattern")
        self.data_patterns = data_patterns
        self.base_pattern = data_patterns[0]
        self.additional_patterns = data_patterns[1:]
        self.num_neighbors = num_neighbors
        self.orig_aug_ratio = orig_aug_ratio

        # 加载原始数据
        self.base_dataset = DrivingData(self.base_pattern, n_neighbors=num_neighbors)
        self.base_dataset_len = len(self.base_dataset)
        
        # 加载LLM标注
        print(f"Loading LLM labels from {labels_path}...")
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
            # 兼容两种格式：列表或字典
            self.labels = labels_data if isinstance(labels_data, list) else labels_data.get('labels', labels_data)
        
        # 构建sample_idx到标签的映射（不使用token，因为可能不可用）
        self.idx_to_label = {}
        for label in self.labels:
            sample_idx = label.get('sample_idx', -1)
            if sample_idx >= 0:
                self.idx_to_label[sample_idx] = label
        
        print(f"Loaded {len(self.idx_to_label)} labeled samples")
        
        # 加载语义词表
        print(f"Loading vocabulary from {vocab_path}...")
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
            self.vocab_dict = vocab_data['keyword_to_id']
            self.vocab_size = vocab_data['vocab_size']
        
        print(f"Loaded vocabulary: {self.vocab_size} keywords")
        
        # 场景类型映射
        self.scene_type_mapping = get_scene_type_mapping()
        
        # 统计有标签的样本（直接使用idx_to_label中的索引）
        self.labeled_indices = []
        self.sample_weights = []  # 用于重采样
        
        self.scene_weights = scene_weights or {}
        if self.scene_weights:
            print(f"Scene weight multipliers: {self.scene_weights}")

        # 直接使用有标签的sample_idx
        for sample_idx in sorted(self.idx_to_label.keys()):
            # 确保sample_idx在数据集范围内
            if sample_idx < len(self.base_dataset):
                self.labeled_indices.append(sample_idx)

                # 计算采样权重（困难样本权重更高）
                label = self.idx_to_label[sample_idx]
                severity = label.get('failure_severity', 'low')
                severity_weight = {'critical': 4.0, 'high': 2.0, 'medium': 1.0, 'low': 0.5}
                weight = severity_weight.get(severity, 1.0)

                scene_type = label.get('scene_type', 'other')
                weight *= self.scene_weights.get(scene_type, 1.0)
                self.sample_weights.append(weight)

        print(f"Found {len(self.labeled_indices)}/{len(self.base_dataset)} samples with labels")

        # 归一化权重
        total_weight = sum(self.sample_weights)
        if total_weight > 0:
            self.sample_weights = [w / total_weight for w in self.sample_weights]
        else:
            uniform = 1.0 / max(len(self.sample_weights), 1)
            self.sample_weights = [uniform for _ in self.sample_weights]

        # 构建样本变体（原始 + 增强）
        self.rng = np.random.default_rng()
        self.sample_variants: Dict[int, List[Tuple[str, Optional[int], Optional[Path]]]] = defaultdict(list)

        missing_base_indices = 0
        for sample_idx in self.labeled_indices:
            if sample_idx < self.base_dataset_len:
                self.sample_variants[sample_idx].append(('base', sample_idx, None))
            else:
                missing_base_indices += 1

        if missing_base_indices:
            print(f"⚠️  {missing_base_indices} labeled samples exceed base dataset length "
                  f"({self.base_dataset_len}); they will be skipped.")

        self.augmented_files = self._collect_augmented_files(self.additional_patterns)
        augmented_variant_count = 0
        for sample_idx, paths in self.augmented_files.items():
            if sample_idx not in self.sample_variants:
                continue
            for path in paths:
                self.sample_variants[sample_idx].append(('aug', None, path))
                augmented_variant_count += 1

        if augmented_variant_count:
            augmented_sample_count = sum(1 for paths in self.augmented_files.values() if paths)
            print(f"Augmented variants attached: {augmented_variant_count} files "
                  f"covering {augmented_sample_count} labeled samples")
        elif self.additional_patterns:
            print("⚠️  No augmented files matched labeled samples; only original data will be used")
            if self.orig_aug_ratio:
                print("⚠️  keep_orig_to_aug_ratio specified but no augmented variants found; ratio ignored")
    
    def __len__(self):
        return len(self.labeled_indices)
 
    def __getitem__(self, idx):
        """
        Returns:
            sample: 原始数据（重新打包成字典）
            scene_type_id: 场景类型索引
            keyword_vector: 关键词多热向量
            label: 完整的LLM标注（用于分析）
        """
        # 获取原始数据
        base_idx = self.labeled_indices[idx]
        variants = self.sample_variants.get(base_idx, [('base', base_idx, None)])
        variant_type, variant_idx, variant_path = self._select_variant(variants)

        if variant_type == 'base':
            ego, neighbors, map_lanes, map_crosswalks, route_lanes, ego_future_gt, neighbors_future_gt = \
                self.base_dataset[variant_idx]
        else:
            (ego, neighbors, map_lanes, map_crosswalks,
             route_lanes, ego_future_gt, neighbors_future_gt) = self._load_augmented_sample(variant_path)

        neighbors = self._truncate_neighbors(neighbors)
        neighbors_future_gt = self._truncate_neighbors_future(neighbors_future_gt)
        
        # 重新打包成字典格式
        sample = {
            'ego_agent_past': torch.from_numpy(ego).float(),
            'neighbor_agents_past': torch.from_numpy(neighbors).float(),
            'map_lanes': torch.from_numpy(map_lanes).float(),
            'map_crosswalks': torch.from_numpy(map_crosswalks).float(),
            'route_lanes': torch.from_numpy(route_lanes).float(),
            'ego_agent_future': torch.from_numpy(ego_future_gt).float(),
            'neighbor_agents_future': torch.from_numpy(neighbors_future_gt).float(),
        }
        
        # 获取标签（直接使用sample_idx）
        label = self.idx_to_label.get(base_idx, {})
        
        # 编码场景类型
        scene_type = label.get('scene_type', 'other')
        scene_type_id = self.scene_type_mapping.get(scene_type, 7)  # 默认'other'=7
        
        # 编码关键词
        keywords = label.get('semantic_keywords', [])
        keyword_vector = torch.zeros(self.vocab_size, dtype=torch.float32)
        for kw in keywords:
            if kw in self.vocab_dict:
                keyword_vector[self.vocab_dict[kw]] = 1.0
        
        return {
            'sample': sample,
            'scene_type_id': torch.tensor(scene_type_id, dtype=torch.long),
            'keyword_vector': keyword_vector,
            'label': label,
            'variant_type': variant_type
        }
    
    def get_sampler(self):
        """获取加权采样器（过采样困难样本）"""
        weight_tensor = torch.tensor(self.sample_weights, dtype=torch.double)
        return WeightedRandomSampler(
            weights=weight_tensor,
            num_samples=len(self),
            replacement=True
        )

    def _select_variant(self, variants: List[Tuple[str, Optional[int], Optional[Path]]]) -> Tuple[str, Optional[int], Optional[Path]]:
        if not variants:
            raise RuntimeError("No variants available for sampling")

        if self.orig_aug_ratio:
            orig_weight, aug_weight = self.orig_aug_ratio
            base_variants = [v for v in variants if v[0] == 'base']
            aug_variants = [v for v in variants if v[0] == 'aug']
            if base_variants and aug_variants and orig_weight >= 0 and aug_weight >= 0:
                total = orig_weight + aug_weight
                if total > 0:
                    prob_aug = aug_weight / total
                    use_aug = self.rng.random() < prob_aug
                    pool = aug_variants if use_aug else base_variants
                    return pool[self.rng.integers(len(pool))]

        # fallback: uniform choice over available variants
        return variants[self.rng.integers(len(variants))]

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _collect_augmented_files(self, patterns: List[str]) -> Dict[int, List[Path]]:
        augmented: Dict[int, List[Path]] = defaultdict(list)
        for pattern in patterns:
            expanded = self._expand_pattern(pattern)
            if not expanded:
                print(f"⚠️  Augmented pattern '{pattern}' produced no matches")
                continue
            for path in expanded:
                match = re.search(r'_idx(\d+)_', path.stem)
                if not match:
                    continue
                sample_idx = int(match.group(1))
                augmented[sample_idx].append(path)
        # 保证路径有序
        for sample_idx, paths in augmented.items():
            augmented[sample_idx] = sorted(paths)
        return augmented

    @staticmethod
    def _expand_pattern(pattern: str) -> List[Path]:
        path = Path(pattern)
        has_wildcard = any(ch in pattern for ch in ('*', '?', '['))
        if has_wildcard:
            matches = [Path(p) for p in glob.glob(pattern)]
        elif path.is_dir():
            matches = list(path.glob('*.npz'))
        elif path.suffix == '.npz' and path.exists():
            matches = [path]
        else:
            matches = []
        # 去除重复并排序
        unique_matches = sorted({m.resolve() for m in matches})
        return unique_matches

    def _load_augmented_sample(self, file_path: Optional[Path]):
        if file_path is None:
            raise ValueError("Augmented sample path is None")
        with np.load(file_path, allow_pickle=False) as data:
            ego = np.asarray(data['ego_agent_past'], dtype=np.float32)
            neighbors = np.asarray(data['neighbor_agents_past'], dtype=np.float32)
            ego_future = np.asarray(data['ego_agent_future'], dtype=np.float32)
            neighbors_future = np.asarray(data['neighbor_agents_future'], dtype=np.float32)

            if 'lanes' in data.files:
                map_lanes = np.asarray(data['lanes'], dtype=np.float32)
            else:
                map_lanes = np.asarray(data['map_lanes'], dtype=np.float32)

            if 'crosswalks' in data.files:
                map_crosswalks = np.asarray(data['crosswalks'], dtype=np.float32)
            else:
                map_crosswalks = np.asarray(data['map_crosswalks'], dtype=np.float32)

            map_route = np.asarray(data['route_lanes'], dtype=np.float32)

        neighbors = neighbors[:self.num_neighbors]
        neighbors_future = neighbors_future[:self.num_neighbors]

        return (
            ego,
            neighbors,
            map_lanes,
            map_crosswalks,
            map_route,
            ego_future,
            neighbors_future
        )

    def _truncate_neighbors(self, neighbors: np.ndarray) -> np.ndarray:
        if neighbors.shape[0] >= self.num_neighbors:
            return neighbors[:self.num_neighbors]
        # pad with zeros if neighbors fewer than required
        pad_count = self.num_neighbors - neighbors.shape[0]
        if pad_count <= 0:
            return neighbors
        pad_shape = (pad_count,) + neighbors.shape[1:]
        padding = np.zeros(pad_shape, dtype=neighbors.dtype)
        return np.concatenate([neighbors, padding], axis=0)

    def _truncate_neighbors_future(self, neighbors_future: np.ndarray) -> np.ndarray:
        if neighbors_future.shape[0] >= self.num_neighbors:
            return neighbors_future[:self.num_neighbors]
        pad_count = self.num_neighbors - neighbors_future.shape[0]
        if pad_count <= 0:
            return neighbors_future
        pad_shape = (pad_count,) + neighbors_future.shape[1:]
        padding = np.zeros(pad_shape, dtype=neighbors_future.dtype)
        return np.concatenate([neighbors_future, padding], axis=0)


# ============================================================================
# 训练函数
# ============================================================================ 

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    log_interval: int = 10,
    teacher_model: Optional[SceneConditionedGameFormer] = None,
    kd_lambda: float = 0.0,
    freeze_bn: bool = False,
    clip_grad_norm: float = 0.0,
    variant_counter: Optional[Counter] = None,
):
    """训练一个epoch"""
    model.train()
    if freeze_bn:
        model.apply(set_batchnorm_eval)
    if teacher_model is not None:
        teacher_model.eval()

    losses: List[float] = []
    ade_list: List[float] = []
    fde_list: List[float] = []
    kd_losses: List[float] = []
    loss_fn = nn.MSELoss()
    kd_fn = nn.MSELoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch_data in enumerate(pbar):
        samples = batch_data['sample']
        scene_type_ids = batch_data['scene_type_id'].to(device)
        keyword_vectors = batch_data['keyword_vector'].to(device)
        variant_types = batch_data.get('variant_type', [])
        if variant_counter is not None:
            for vt in variant_types:
                variant_counter[vt] += 1

        inputs = {
            'ego_agent_past': samples['ego_agent_past'].to(device),
            'neighbor_agents_past': samples['neighbor_agents_past'].to(device),
            'map_lanes': samples['map_lanes'].to(device),
            'map_crosswalks': samples['map_crosswalks'].to(device),
            'route_lanes': samples['route_lanes'].to(device)
        }
        ego_future = samples['ego_agent_future'].to(device)

        _, ego_plan = model(
            inputs,
            scene_type_ids=scene_type_ids,
            keyword_vectors=keyword_vectors
        )

        loss = loss_fn(ego_plan[..., :2], ego_future[..., :2])
        if teacher_model is not None and kd_lambda > 0:
            with torch.no_grad():
                _, teacher_plan = teacher_model(
                    inputs,
                    scene_type_ids=scene_type_ids,
                    keyword_vectors=keyword_vectors
                )
            kd_loss_value = kd_fn(ego_plan[..., :2], teacher_plan[..., :2])
            kd_losses.append(float(kd_loss_value.detach().cpu()))
            loss = loss + kd_lambda * kd_loss_value

        optimizer.zero_grad()
        loss.backward()
        if clip_grad_norm and clip_grad_norm > 0 and trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, clip_grad_norm)
        optimizer.step()
        if freeze_bn:
            model.apply(set_batchnorm_eval)

        losses.append(float(loss.item()))

        with torch.no_grad():
            ade, fde = compute_ego_metrics(ego_plan, ego_future)
            ade_list.append(ade)
            fde_list.append(fde)

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = np.mean(losses[-log_interval:]) if losses else 0.0
            avg_ade = np.mean(ade_list[-log_interval:]) if ade_list else 0.0
            avg_fde = np.mean(fde_list[-log_interval:]) if fde_list else 0.0
            postfix = {
                'loss': f'{avg_loss:.4f}',
                'ADE': f'{avg_ade:.3f}',
                'FDE': f'{avg_fde:.3f}'
            }
            if kd_losses:
                avg_kd = np.mean(kd_losses[-min(log_interval, len(kd_losses)):])
                postfix['KD'] = f'{avg_kd:.4f}'
            pbar.set_postfix(postfix)

    epoch_loss = np.mean(losses) if losses else 0.0
    epoch_ade = np.mean(ade_list) if ade_list else 0.0
    epoch_fde = np.mean(fde_list) if fde_list else 0.0
    epoch_kd = np.mean(kd_losses) if kd_losses else 0.0

    return {
        'loss': epoch_loss,
        'ade': epoch_ade,
        'fde': epoch_fde,
        'kd_loss': epoch_kd
    }

def validate(model, dataloader, device):
    """验证"""
    model.eval()
    
    ade_values: List[float] = []
    fde_values: List[float] = []
    scene_metrics: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"ade": [], "fde": []})
    scene_id_to_name = {v: k for k, v in get_scene_type_mapping().items()}

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating"):
            # 解包batch
            samples = batch_data['sample']
            scene_type_ids = batch_data['scene_type_id'].to(device)
            keyword_vectors = batch_data['keyword_vector'].to(device)
            
            # 构建输入
            inputs = {
                'ego_agent_past': samples['ego_agent_past'].to(device),
                'neighbor_agents_past': samples['neighbor_agents_past'].to(device),
                'map_lanes': samples['map_lanes'].to(device),
                'map_crosswalks': samples['map_crosswalks'].to(device),
                'route_lanes': samples['route_lanes'].to(device)
            }
            
            # Ground truth
            ego_future = samples['ego_agent_future'].to(device)
            
            # 前向传播
            decoder_outputs, ego_plan = model(
                inputs,
                scene_type_ids=scene_type_ids,
                keyword_vectors=keyword_vectors
            )

            # 计算ADE/FDE并记录场景表现
            diffs = torch.norm(ego_plan[:, :, :2] - ego_future[:, :, :2], dim=-1)
            batch_ade = diffs.mean(dim=1)
            batch_fde = diffs[:, -1]

            ade_values.extend(batch_ade.cpu().float().tolist())
            fde_values.extend(batch_fde.cpu().float().tolist())

            for idx, scene_id in enumerate(scene_type_ids.cpu().tolist()):
                scene_entry = scene_metrics[scene_id]
                scene_entry["ade"].append(float(batch_ade[idx].item()))
                scene_entry["fde"].append(float(batch_fde[idx].item()))

    val_ade = float(np.mean(ade_values)) if ade_values else float("nan")
    val_fde = float(np.mean(fde_values)) if fde_values else float("nan")

    per_scene_summary: Dict[str, Dict[str, float]] = {}
    for scene_id, metrics in scene_metrics.items():
        if not metrics["ade"]:
            continue
        name = scene_id_to_name.get(scene_id, f"id_{scene_id}")
        per_scene_summary[name] = {
            "ade": float(np.mean(metrics["ade"])),
            "fde": float(np.mean(metrics["fde"])),
            "count": len(metrics["ade"]),
        }

    return {'ade': val_ade, 'fde': val_fde, 'per_scene': per_scene_summary}


# ============================================================================
# 主训练流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Scene-Conditioned GameFormer')
    
    # 数据路径
    parser.add_argument('--data_root', type=str,
                       default='/home/hamster/nuplan/processed_data/valid/*.npz',
                       help='Training data root')
    parser.add_argument('--labels_path', type=str,
                       default='./eval_out/llm_longtail_labels.json',
                       help='LLM labels file')
    parser.add_argument('--vocab_path', type=str,
                       default='./eval_out/clusters/semantic_vocab.json',
                       help='Semantic vocabulary file')
    parser.add_argument('--pretrained_path', type=str,
                       default='./training_log/Exp1/model_epoch_17_valADE_1.97.pth',
                       help='Pretrained GameFormer checkpoint')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--clip_grad_norm', type=float, default=0.0,
                       help='Gradient clipping max norm (0 disables)')
    parser.add_argument('--num_neighbors', type=int, default=10,
                       help='Number of neighbor vehicles')
    parser.add_argument('--encoder_layers', type=int, default=6,
                       help='Number of encoder layers')
    parser.add_argument('--decoder_levels', type=int, default=3,
                       help='Number of decoder levels')
    
    # 训练策略
    parser.add_argument('--freeze_base', action='store_true',
                       help='Freeze base encoder (only train semantic parts)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone modules; use --unfreeze_modules to selectively enable training')
    parser.add_argument('--freeze_bn_running_stats', action='store_true',
                        help='Keep BatchNorm layers in eval mode to avoid updating running statistics')
    parser.add_argument('--unfreeze_modules', type=str, default=None,
                        help='Comma separated module name prefixes to keep trainable when freezing backbone')
    parser.add_argument('--keep_orig_to_aug_ratio', type=str, default=None,
                        help='Sampling ratio between original and augmented variants, e.g. 4:1')
    parser.add_argument('--kd_teacher_ckpt', type=str, default=None,
                        help='Optional teacher checkpoint for knowledge distillation')
    parser.add_argument('--kd_lambda', type=float, default=0.0,
                        help='Weight of KD loss (0 disables)')
    parser.add_argument('--weighted_sampling', action='store_true',
                       help='Use weighted sampling (oversample difficult samples)')
    
    # 输出
    parser.add_argument('--output_dir', type=str,
                       default='./training_log/SceneConditioned',
                       help='Output directory for checkpoints')
    parser.add_argument('--scene_weights', type=str, default=None,
                       help='Scene-specific sampling weights, e.g. cut_in=5,high_speed=3')
    
    args = parser.parse_args()
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*80}")
    print(f"🚀 SCENE-CONDITIONED GAMEFORMER TRAINING")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Pretrained: {args.pretrained_path}")
    print(f"Labels: {args.labels_path}")
    print(f"Vocab: {args.vocab_path}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )
    
    # ========================================================================
    # 1. 加载数据
    # ========================================================================
    logging.info("Loading dataset...")
    def parse_scene_weights(arg: Optional[str]) -> Dict[str, float]:
        if not arg:
            return {}
        result: Dict[str, float] = {}
        for item in arg.split(','):
            if not item.strip():
                continue
            if '=' not in item:
                continue
            key, value = item.split('=', 1)
            try:
                result[key.strip()] = float(value.strip())
            except ValueError:
                print(f"Warning: invalid scene weight '{item}', skipping")
        return result

    scene_weights = parse_scene_weights(args.scene_weights)

    data_patterns = parse_data_patterns(args.data_root)
    if not data_patterns:
        raise ValueError("No data patterns provided for --data_root")
    logging.info(f"Data patterns: {data_patterns}")

    orig_aug_ratio = parse_orig_aug_ratio(args.keep_orig_to_aug_ratio)
    if orig_aug_ratio:
        logging.info(f"Original to augmented sampling ratio set to {orig_aug_ratio[0]}:{orig_aug_ratio[1]}")

    unfreeze_modules = parse_unfreeze_modules(args.unfreeze_modules)
    if unfreeze_modules:
        logging.info(f"Modules to unfreeze when freezing backbone: {unfreeze_modules}")

    dataset = SceneConditionedDataset(
        data_patterns=data_patterns,
        labels_path=args.labels_path,
        vocab_path=args.vocab_path,
        num_neighbors=args.num_neighbors,
        scene_weights=scene_weights,
        orig_aug_ratio=orig_aug_ratio,
    )
    
    # 数据加载器
    if args.weighted_sampling:
        sampler = dataset.get_sampler()
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,  # 使用0避免多进程问题
            collate_fn=custom_collate_fn
        )
        logging.info("Using weighted sampling (oversample difficult samples)")
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # 使用0避免多进程问题
            collate_fn=custom_collate_fn
        )
    
    logging.info(f"Dataset size: {len(dataset)}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Batches per epoch: {len(dataloader)}")
    
    # ========================================================================
    # 2. 创建模型
    # ========================================================================
    logging.info("Creating model...")
    model = SceneConditionedGameFormer.from_pretrained(
        args.pretrained_path,
        vocab_size=dataset.vocab_size,
        encoder_layers=args.encoder_layers,
        decoder_levels=args.decoder_levels,
        modalities=6,
        neighbors=args.num_neighbors
    )
    
    freeze_backbone_flag = args.freeze_backbone or args.freeze_base
    if args.freeze_base and not args.freeze_backbone:
        logging.info("`--freeze_base` is deprecated; treating as --freeze_backbone.")
    if freeze_backbone_flag:
        logging.info("Freezing backbone parameters; gradients disabled by default.")
        for name, param in model.named_parameters():
            param.requires_grad = False
        if unfreeze_modules:
            for prefix in unfreeze_modules:
                matched = False
                for name, param in model.named_parameters():
                    if name.startswith(prefix):
                        param.requires_grad = True
                        matched = True
                if matched:
                    logging.info(f"Unfroze parameters with prefix '{prefix}'")
                else:
                    logging.warning(f"No parameters matched prefix '{prefix}' while attempting to unfreeze")
        else:
            logging.warning("Backbone frozen but no modules specified to unfreeze; model may not update")
    
    model = model.to(device)
    
    teacher_model = None
    if args.kd_teacher_ckpt and args.kd_lambda > 0:
        logging.info(f"Loading teacher checkpoint from {args.kd_teacher_ckpt}")
        teacher_model = SceneConditionedGameFormer.from_pretrained(
            args.kd_teacher_ckpt,
            vocab_size=dataset.vocab_size,
            encoder_layers=args.encoder_layers,
            decoder_levels=args.decoder_levels,
            modalities=6,
            neighbors=args.num_neighbors
        ).to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
    elif args.kd_teacher_ckpt and args.kd_lambda <= 0:
        logging.warning("kd_teacher_ckpt provided but kd_lambda <= 0; skipping KD teacher load")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # ========================================================================
    # 3. 优化器
    # ========================================================================
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.1
    )
    
    # ========================================================================
    # 4. 训练循环
    # ========================================================================
    logging.info(f"\n{'='*80}")
    logging.info("Starting training...")
    logging.info(f"{'='*80}\n")
    
    best_val_ade = float('inf')
    training_history = []
    
    for epoch in range(1, args.epochs + 1):
        logging.info(f"\nEpoch {epoch}/{args.epochs}")
        logging.info(f"{'─'*80}")
        
        # 训练
        variant_counter = Counter()
        train_metrics = train_one_epoch(
            model,
            dataloader,
            optimizer,
            device,
            epoch,
            teacher_model=teacher_model,
            kd_lambda=args.kd_lambda,
            freeze_bn=args.freeze_bn_running_stats,
            clip_grad_norm=args.clip_grad_norm,
            variant_counter=variant_counter
        )
        
        train_log = (f"Train - Loss: {train_metrics['loss']:.4f}, "
                     f"ADE: {train_metrics['ade']:.3f}, "
                     f"FDE: {train_metrics['fde']:.3f}")
        if train_metrics.get('kd_loss', 0.0) > 0:
            train_log += f", KD: {train_metrics['kd_loss']:.4f} (λ={args.kd_lambda})"
        logging.info(train_log)
        total_variants = sum(variant_counter.values())
        if total_variants > 0:
            base_count = variant_counter.get('base', 0)
            aug_count = total_variants - base_count
            base_pct = base_count / total_variants * 100
            aug_pct = aug_count / total_variants * 100
            logging.info(f"Variant mix - base: {base_count} ({base_pct:.1f}%), "
                         f"aug: {aug_count} ({aug_pct:.1f}%)")
        else:
            logging.info("Variant mix - only base samples seen this epoch")
        
        # 验证（在训练集上，因为没有单独的验证集）
        # 实际应该有独立的验证集
        val_metrics = validate(model, dataloader, device)
        
        logging.info(
            f"Val   - ADE: {val_metrics['ade']:.3f}, "
            f"FDE: {val_metrics['fde']:.3f}"
        )
        scene_breakdown = val_metrics.get('per_scene') or {}
        if scene_breakdown:
            ranked = sorted(scene_breakdown.items(), key=lambda item: item[1]['fde'], reverse=True)
            highlights = []
            for name, metrics in ranked[:3]:
                highlights.append(f"{name}:FDE={metrics['fde']:.2f}")
            logging.info("Val scene spotlight: " + ", ".join(highlights))
            left_turn = scene_breakdown.get('starting_left_turn')
            if left_turn:
                logging.info(
                    "Val left-turn ADE/FDE: "
                    f"{left_turn['ade']:.3f}/{left_turn['fde']:.3f} (n={left_turn['count']})"
                )
        
        # 学习率调整
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Learning rate: {current_lr:.6f}")
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        # 保存最新
        torch.save(checkpoint, output_dir / 'model_latest.pth')
        
        # 保存最佳
        if val_metrics['ade'] < best_val_ade:
            best_val_ade = val_metrics['ade']
            torch.save(checkpoint, output_dir / f'model_best_ADE_{val_metrics["ade"]:.3f}.pth')
            logging.info(f"✅ New best model saved (ADE: {val_metrics['ade']:.3f})")
        
        # 定期保存
        if epoch % 5 == 0:
            torch.save(checkpoint, output_dir / f'model_epoch_{epoch}.pth')
        
        # 记录历史
        training_history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': current_lr,
            'variant_mix': dict(variant_counter)
        })
    
    # ========================================================================
    # 5. 保存训练历史
    # ========================================================================
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logging.info(f"\n{'='*80}")
    logging.info("✅ TRAINING COMPLETED!")
    logging.info(f"{'='*80}")
    logging.info(f"Best validation ADE: {best_val_ade:.3f}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"\n🎯 Next steps:")
    logging.info(f"   1. Evaluate on test set")
    logging.info(f"   2. Compare with baseline GameFormer")
    logging.info(f"   3. Analyze performance on different scene types")
    logging.info(f"{'='*80}")


if __name__ == "__main__":
    main()
