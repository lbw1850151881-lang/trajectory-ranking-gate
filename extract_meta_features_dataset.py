"""
批量提取数据集的元特征
为所有样本添加模型内部信号（不确定性、稳定性等）
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from GameFormer.predictor import GameFormer
from GameFormer.train_utils import DrivingData
from train_predictor_lstm import Seq2SeqLSTM
from scene_conditioned_gameformer import SceneConditionedGameFormer
from meta_features_extractor import extract_meta_features


def extract_all_meta_features(lstm_model, scene_model, gmf_model, dataset, device='cuda', K=8, max_samples=None):
    """
    为整个数据集提取元特征
    
    Args:
        lstm_model: LSTM 模型
        gmf_model: GameFormer 模型
        dataset: DrivingData 数据集
        device: 设备
        K: MC Dropout 采样次数
        max_samples: 最大处理样本数（用于快速测试）
    
    Returns:
        meta_features_list: list of dicts
    """
    lstm_model.eval()
    scene_model.eval()
    gmf_model.eval()
    
    meta_features_list = []
    
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    print(f"\n🔬 Extracting meta-features for {n_samples} samples...")
    print(f"   MC Dropout samples: K={K}")
    print(f"   Device: {device}\n")
    
    for idx in tqdm(range(n_samples), desc="Extracting meta-features", unit="sample"):
        try:
            batch = dataset[idx]
            
            # 提取元特征
            meta_feats = extract_meta_features(
                lstm_model, scene_model, gmf_model, batch, device=device, K=K
            )
            
            # 添加样本索引
            meta_feats['sample_idx'] = idx
            
            meta_features_list.append(meta_feats)
            
        except Exception as e:
            print(f"\n⚠️  Warning: Sample {idx} failed with error: {e}")
            continue
    
    print(f"\n✅ Successfully extracted meta-features for {len(meta_features_list)} samples")
    
    return meta_features_list


def merge_with_existing_features(meta_features_list, existing_stats_path):
    """
    将元特征与现有的复杂度特征合并
    
    Args:
        meta_features_list: 新提取的元特征
        existing_stats_path: 现有的 fusion_detailed_stats.json 路径
    
    Returns:
        merged_samples: 合并后的样本列表
    """
    print(f"\n🔗 Merging with existing features from {existing_stats_path}...")
    
    with open(existing_stats_path, 'r') as f:
        existing_samples = json.load(f)
    
    # 创建索引映射
    meta_dict = {mf['sample_idx']: mf for mf in meta_features_list}
    
    merged_samples = []
    
    for sample in existing_samples:
        sample_idx = sample['sample_idx']
        
        if sample_idx in meta_dict:
            # 合并元特征
            sample.update(meta_dict[sample_idx])
        
        merged_samples.append(sample)
    
    print(f"✅ Merged {len(merged_samples)} samples")
    
    return merged_samples


def compute_meta_feature_stats(meta_features_list):
    """计算元特征的统计信息"""
    
    print(f"\n📊 Meta-Features Statistics:")
    print("="*80)
    
    # 提取所有特征
    feature_keys = [k for k in meta_features_list[0].keys() if k != 'sample_idx']
    
    for key in feature_keys:
        values = [mf[key] for mf in meta_features_list]
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        print(f"{key:<35} Mean: {mean_val:>8.4f} | Std: {std_val:>8.4f} | "
              f"Range: [{min_val:>7.4f}, {max_val:>7.4f}]")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract meta-features from models')
    parser.add_argument('--data_root', type=str, 
                       default='/home/hamster/nuplan/processed_data/valid',
                       help='Path to processed data')
    parser.add_argument('--lstm_model', type=str, 
                       default='/mnt/d/paper/nuplan/checkpoints/best_seq2seq_lstm_xy_vxy.pth',
                       help='Path to LSTM model')
    parser.add_argument('--lstm_hidden_dim', type=int, default=256,
                       help='Hidden dimension of LSTM expert')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--lstm_dropout', type=float, default=0.25,
                       help='Dropout used in LSTM expert')
    parser.add_argument('--scene_model', type=str,
                       default='training_log/SceneConditioned/model_best_ADE_4.977.pth',
                       help='Path to Scene-Conditioned GameFormer checkpoint')
    parser.add_argument('--scene_encoder_layers', type=int, default=3,
                       help='Scene-conditioned encoder layers')
    parser.add_argument('--scene_decoder_levels', type=int, default=2,
                       help='Scene-conditioned decoder levels')
    parser.add_argument('--scene_neighbors', type=int, default=10,
                       help='Number of neighbour agents for scene-conditioned model')
    parser.add_argument('--scene_vocab', type=str, default='clusters_200/semantic_vocab.json',
                       help='Semantic vocabulary JSON used by Scene-Conditioned model')
    parser.add_argument('--gmf_model', type=str, 
                       default='pretrained_models/GameFormer-Planner/training_log/Exp1/model_epoch_18_valADE_1.7272.pth',
                       help='Path to GameFormer model')
    parser.add_argument('--existing_stats', type=str,
                       default='./eval_out/fusion_detailed_stats_th0.30.json',
                       help='Path to existing fusion stats')
    parser.add_argument('--K', type=int, default=8,
                       help='MC Dropout sampling times')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to process (for testing)')
    parser.add_argument('--output_dir', type=str, default='./eval_out',
                       help='Output directory')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}\n")
    
    # ===========================
    # 加载数据集
    # ===========================
    print(f"📂 Loading dataset from {args.data_root}...")
    dataset = DrivingData(args.data_root + "/*.npz", n_neighbors=10)
    print(f"✅ Loaded {len(dataset)} samples\n")
    
    # ===========================
    # 加载模型
    # ===========================
    print(f"🧠 Loading models...")
    
    # LSTM
    lstm_model = Seq2SeqLSTM(
        input_dim=4,
        hidden_dim=args.lstm_hidden_dim,
        num_layers=args.lstm_num_layers,
        future_steps=80,
        dropout=args.lstm_dropout,
    )
    lstm_model.load_state_dict(torch.load(args.lstm_model, map_location=device))
    lstm_model = lstm_model.to(device)
    print(f"   ✅ LSTM loaded from {args.lstm_model}")

    # Scene-Conditioned expert
    vocab_path = Path(args.scene_vocab)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Scene vocabulary not found: {vocab_path}")
    vocab_json = json.loads(vocab_path.read_text())
    if isinstance(vocab_json, dict):
        if 'keyword_to_id' in vocab_json:
            vocab_size = len(vocab_json['keyword_to_id'])
        elif 'vocab' in vocab_json:
            vocab_size = len(vocab_json['vocab'])
        else:
            raise ValueError(f"Unrecognised vocabulary schema in {vocab_path}")
    else:
        vocab_size = len(vocab_json)

    scene_model = SceneConditionedGameFormer(
        vocab_size=vocab_size,
        encoder_layers=args.scene_encoder_layers,
        decoder_levels=args.scene_decoder_levels,
        neighbors=args.scene_neighbors,
    )
    scene_ckpt = torch.load(args.scene_model, map_location=device)
    scene_state = scene_ckpt.get('model', scene_ckpt.get('model_state_dict', scene_ckpt))
    scene_model.load_state_dict(scene_state, strict=False)
    scene_model = scene_model.to(device)
    print(f"   ✅ Scene-Conditioned model loaded from {args.scene_model}")
    
    # GameFormer
    gmf_model = GameFormer(encoder_layers=3, decoder_levels=2, neighbors=10)
    gmf_model.load_state_dict(torch.load(args.gmf_model, map_location=device))
    gmf_model = gmf_model.to(device)
    print(f"   ✅ GameFormer loaded from {args.gmf_model}\n")
    
    # ===========================
    # 提取元特征
    # ===========================
    meta_features_list = extract_all_meta_features(
        lstm_model, scene_model, gmf_model, dataset,
        device=device, K=args.K, max_samples=args.max_samples
    )
    
    # ===========================
    # 统计分析
    # ===========================
    compute_meta_feature_stats(meta_features_list)
    
    # ===========================
    # 合并现有特征
    # ===========================
    if Path(args.existing_stats).exists():
        merged_samples = merge_with_existing_features(meta_features_list, args.existing_stats)
        
        # 保存合并后的数据
        output_path = f"{args.output_dir}/fusion_stats_with_meta_features.json"
        with open(output_path, 'w') as f:
            json.dump(merged_samples, f, indent=2)
        print(f"✅ Merged data saved to {output_path}")
    else:
        print(f"⚠️  Existing stats file not found: {args.existing_stats}")
        
        # 仅保存元特征
        output_path = f"{args.output_dir}/meta_features_only.json"
        with open(output_path, 'w') as f:
            json.dump(meta_features_list, f, indent=2)
        print(f"✅ Meta-features saved to {output_path}")
    
    print("\n" + "="*80)
    print("✅ META-FEATURES EXTRACTION COMPLETED!")
    print("="*80)
    print(f"\n💡 Next step:")
    print(f"   python gate_meta_mlp.py --data {output_path}")
    print("="*80 + "\n")
