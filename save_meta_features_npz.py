"""
将JSON格式的meta特征转换为NPZ格式
用于 eval_llm_gate.py
"""

import json
import numpy as np
from pathlib import Path

from tri_feature_utils import BASE_FEATURE_KEYS, DERIVED_FEATURE_KEYS, compute_additional_features

def convert_json_to_npz(json_path, output_path):
    """
    将JSON格式的meta特征转换为NPZ格式
    
    Args:
        json_path: 输入JSON文件路径
        output_path: 输出NPZ文件路径
    """
    print(f"📂 Loading meta-features from {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} samples\n")
    
    # 提取特征和元数据
    sample_indices = []
    features = []
    lstm_fde = []
    gmf_fde = []
    scene_fde = []
    scene_ade = []
    transformer_fde = []
    transformer_ade = []
    
    # Meta特征的键（21维）
    meta_keys = BASE_FEATURE_KEYS + DERIVED_FEATURE_KEYS
    
    print("🔍 Processing samples...")
    
    for sample in data:
        sample_indices.append(sample.get('sample_idx', -1))
        
        # 提取21维特征
        feature_map = {k: sample.get(k, 0.0) for k in BASE_FEATURE_KEYS}
        feature_map.update(compute_additional_features(sample))
        feature_vec = []
        for key in meta_keys:
            if key in feature_map:
                feature_vec.append(feature_map[key])
            else:
                print(f"⚠️  Missing key '{key}' in sample {sample.get('sample_idx', 'unknown')}, using 0.0")
                feature_vec.append(0.0)
        
        features.append(feature_vec)
        # 从JSON中提取FDE（键名可能是 'lstm_fde' 或 'gameformer_fde'）
        lstm_fde.append(sample.get('lstm_fde', 0.0))
        gmf_fde.append(sample.get('gameformer_fde', 0.0))
        sc_fde = sample.get('scene_conditioned_fde', sample.get('gameformer_fde', 0.0))
        sc_ade = sample.get('scene_conditioned_ade', sample.get('gameformer_ade', sc_fde))
        scene_fde.append(sc_fde)
        scene_ade.append(sc_ade)
        transformer_fde.append(sample.get('transformer_fde', sc_fde))
        transformer_ade.append(sample.get('transformer_ade', sc_ade))
    
    # 转换为numpy数组
    features = np.array(features, dtype=np.float32)
    sample_indices = np.array(sample_indices, dtype=np.int32)
    lstm_fde = np.array(lstm_fde, dtype=np.float32)
    gmf_fde = np.array(gmf_fde, dtype=np.float32)
    scene_fde = np.array(scene_fde, dtype=np.float32)
    scene_ade = np.array(scene_ade, dtype=np.float32)
    transformer_fde = np.array(transformer_fde, dtype=np.float32)
    transformer_ade = np.array(transformer_ade, dtype=np.float32)
    
    print(f"\n📊 Data shapes:")
    print(f"   Features: {features.shape}")
    print(f"   Sample Indices: {sample_indices.shape}")
    print(f"   LSTM FDE: {lstm_fde.shape}")
    print(f"   GMF FDE: {gmf_fde.shape}")
    print(f"   Scene FDE: {scene_fde.shape}")
    
    # 保存为NPZ
    print(f"\n💾 Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        features=features,
        sample_indices=sample_indices,
        lstm_fde=lstm_fde,
        gmf_fde=gmf_fde,
        scene_fde=scene_fde,
        scene_ade=scene_ade,
        transformer_fde=transformer_fde,
        transformer_ade=transformer_ade,
    )
    
    print(f"✅ Successfully saved NPZ file!\n")
    
    # 验证
    print("🔍 Verifying saved file...")
    loaded = np.load(output_path)
    print(f"   ✓ Features: {loaded['features'].shape}")
    print(f"   ✓ Sample Indices: {loaded['sample_indices'].shape}")
    print(f"   ✓ LSTM FDE: {loaded['lstm_fde'].shape}")
    print(f"   ✓ GMF FDE: {loaded['gmf_fde'].shape}")
    print(f"   ✓ Scene FDE: {loaded['scene_fde'].shape}")
    if 'transformer_fde' in loaded:
        print(f"   ✓ Transformer FDE: {loaded['transformer_fde'].shape}")
    
    print("\n" + "="*80)
    print("✅ CONVERSION COMPLETED!")
    print("="*80)
    print(f"\n💡 Next step:")
    print(f"   python eval_llm_gate.py --llm_enabled --llm_override --debug --visualize")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert JSON meta-features to NPZ')
    parser.add_argument('--input', type=str,
                       default='eval_out/fusion_stats_with_meta_features.json',
                       help='Input JSON file')
    parser.add_argument('--output', type=str,
                       default='eval_out/meta_features_dataset.npz',
                       help='Output NPZ file')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    convert_json_to_npz(args.input, args.output)
