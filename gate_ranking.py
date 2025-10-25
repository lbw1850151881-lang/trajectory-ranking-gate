"""
基于排序学习的门控决策
使用 Pairwise Ranking (RankNet) 学习"哪个专家更好"
直接优化排序关系，而非预测绝对 FDE 值
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


# ===========================
# 1. Pairwise Ranking Loss
# ===========================
class PairwiseRankingLoss(nn.Module):
    """
    RankNet 损失函数
    
    给定特征 x 和两个模型的 FDE (fde_lstm, fde_gmf)，
    学习一个打分函数 s(x)，使得：
    - 若 fde_gmf < fde_lstm (GMF 更好)，则 s(x) > 0
    - 若 fde_lstm < fde_gmf (LSTM 更好)，则 s(x) < 0
    
    Loss = -log(sigmoid(label * score))
    其中 label = +1 if GMF better else -1
    """
    
    def __init__(self):
        super(PairwiseRankingLoss, self).__init__()
    
    def forward(self, scores, labels):
        """
        Args:
            scores: [B] 模型输出的相对优势分数
            labels: [B] +1 if GMF better, -1 if LSTM better
        """
        # RankNet loss: -log(sigmoid(label * score))
        loss = -torch.log(torch.sigmoid(labels * scores) + 1e-8)
        return loss.mean()


# ===========================
# 2. 排序学习数据集
# ===========================
class RankingDataset(Dataset):
    """排序学习数据集"""
    
    def __init__(self, samples, feature_keys=None):
        """
        Args:
            samples: 数据样本
            feature_keys: 特征列表
        """
        
        # 特征
        if feature_keys is None:
            self.feature_keys = [
                # 几何特征
                'complexity_score',
                'n_neighbors',
                'neighbor_score',
                'speed_variance',
                'velocity_score',
                'min_neighbor_dist',
                'proximity_score',
                'n_crosswalks',
                'crosswalk_score',
                # 元特征
                'lstm_uncertainty_total',
                'lstm_uncertainty_final',
                'lstm_stability',
                'lstm_physics_violation',
                'gmf_uncertainty_total',
                'gmf_uncertainty_final',
                'gmf_stability',
                'gmf_physics_violation',
                'uncertainty_ratio',
                'stability_ratio',
                'violation_diff',
                'stability_diff'
            ]
        else:
            self.feature_keys = feature_keys
        
        # 提取特征和标签
        self.features = []
        self.labels = []  # +1 if GMF better, -1 if LSTM better
        self.fde_diff = []  # FDE_LSTM - FDE_GMF (用于分析)
        
        gmf_better_count = 0
        lstm_better_count = 0
        
        for s in samples:
            # 特征
            feat = [s.get(k, 0.0) for k in self.feature_keys]
            self.features.append(feat)
            
            # 标签：谁更好
            lstm_fde = s['lstm_fde']
            gmf_fde = s['gameformer_fde']
            
            if gmf_fde < lstm_fde:  # GMF 更好
                label = +1.0
                gmf_better_count += 1
            else:  # LSTM 更好
                label = -1.0
                lstm_better_count += 1
            
            self.labels.append(label)
            self.fde_diff.append(lstm_fde - gmf_fde)
        
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.fde_diff = np.array(self.fde_diff, dtype=np.float32)
        
        # 特征归一化
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0) + 1e-8
        self.features = (self.features - self.mean) / self.std
        
        # 打印统计
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples:   {len(samples)}")
        print(f"   GMF better:      {gmf_better_count} ({gmf_better_count/len(samples)*100:.1f}%)")
        print(f"   LSTM better:     {lstm_better_count} ({lstm_better_count/len(samples)*100:.1f}%)")
        print(f"\n📊 FDE Difference (LSTM - GMF) Distribution:")
        print(f"   Mean:   {self.fde_diff.mean():>8.4f} m")
        print(f"   Std:    {self.fde_diff.std():>8.4f} m")
        print(f"   Min:    {self.fde_diff.min():>8.4f} m")
        print(f"   Q25:    {np.percentile(self.fde_diff, 25):>8.4f} m")
        print(f"   Q50:    {np.percentile(self.fde_diff, 50):>8.4f} m")
        print(f"   Q75:    {np.percentile(self.fde_diff, 75):>8.4f} m")
        print(f"   Max:    {self.fde_diff.max():>8.4f} m")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )
    
    def get_normalization_stats(self):
        return {'mean': self.mean, 'std': self.std, 'feature_keys': self.feature_keys}


# ===========================
# 3. 排序学习模型
# ===========================
class RankingModel(nn.Module):
    """
    排序学习模型
    
    输出一个标量分数：
    - 分数 > 0 → 选择 GameFormer
    - 分数 < 0 → 选择 LSTM
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super(RankingModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层：单个标量分数
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, D] 特征
        
        Returns:
            scores: [B] 相对优势分数
                > 0: GMF 更好
                < 0: LSTM 更好
        """
        return self.network(x).squeeze()


# ===========================
# 4. 训练函数
# ===========================
def train_ranking_model(model, train_loader, val_loader, 
                       epochs=50, lr=1e-3, device='cuda'):
    """训练排序模型"""
    
    criterion = PairwiseRankingLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    print(f"\n🚀 Training Ranking Model...")
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            scores = model(features)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            
            # 计算准确率
            predictions = torch.sign(scores)  # +1 或 -1
            train_correct += (predictions == labels).sum().item()
            train_total += features.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                scores = model(features)
                loss = criterion(scores, labels)
                val_loss += loss.item() * features.size(0)
                
                predictions = torch.sign(scores)
                val_correct += (predictions == labels).sum().item()
                val_total += features.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        # 保存最佳模型（基于准确率）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    
    model.load_state_dict(best_model_state)
    print(f"\n✅ Best validation: Loss={best_val_loss:.4f}, Acc={best_val_acc:.4f}\n")
    
    return history


# ===========================
# 5. 评估函数
# ===========================
def extract_features(sample, feature_keys, norm_stats):
    """提取并归一化特征"""
    features = []
    for key in feature_keys:
        if key in sample:
            features.append(sample[key])
        else:
            features.append(0.0)
    
    features = np.array(features, dtype=np.float32)
    mean = norm_stats['mean']
    std = norm_stats['std']
    features = (features - mean) / std
    
    return features


def evaluate_ranking_gate(model, samples, norm_stats, threshold=0.0, device='cuda'):
    """
    评估排序门控
    
    决策规则：
    - score > threshold → 选择 GameFormer
    - score < threshold → 选择 LSTM
    """
    fusion_fdes = []
    fusion_ades = []
    model_choices = {'lstm': 0, 'gameformer': 0}
    
    all_scores = []
    all_true_labels = []
    correct_predictions = 0
    
    feature_keys = norm_stats['feature_keys']
    
    model.eval()
    
    with torch.no_grad():
        for sample in samples:
            # 提取特征
            features = extract_features(sample, feature_keys, norm_stats)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 预测分数
            score = model(features_tensor).item()
            all_scores.append(score)
            
            # 真实标签
            lstm_fde = sample['lstm_fde']
            gmf_fde = sample['gameformer_fde']
            true_label = +1.0 if gmf_fde < lstm_fde else -1.0
            all_true_labels.append(true_label)
            
            # 决策
            if score > threshold:  # GMF 更好
                model_choices['gameformer'] += 1
                fde = gmf_fde
                ade = sample['gameformer_ade']
                pred_label = +1.0
            else:  # LSTM 更好
                model_choices['lstm'] += 1
                fde = lstm_fde
                ade = sample['lstm_ade']
                pred_label = -1.0
            
            fusion_fdes.append(fde)
            fusion_ades.append(ade)
            
            if pred_label == true_label:
                correct_predictions += 1
    
    # Oracle
    oracle_fdes = [min(s['lstm_fde'], s['gameformer_fde']) for s in samples]
    oracle_fde = float(np.mean(oracle_fdes))
    
    # GameFormer baseline
    gmf_fdes = [s['gameformer_fde'] for s in samples]
    gmf_fde = float(np.mean(gmf_fdes))
    
    # 计算 Oracle 实现率
    fde_improvement = gmf_fde - float(np.mean(fusion_fdes))
    oracle_potential = gmf_fde - oracle_fde
    oracle_realization = (fde_improvement / oracle_potential * 100) if oracle_potential > 0 else 0
    
    results = {
        'ade': float(np.mean(fusion_ades)),
        'fde': float(np.mean(fusion_fdes)),
        'fde_std': float(np.std(fusion_fdes)),
        'regret': float(np.mean(fusion_fdes)) - oracle_fde,
        'oracle_fde': oracle_fde,
        'gameformer_baseline_fde': gmf_fde,
        'fde_improvement_vs_gmf': fde_improvement,
        'oracle_potential': oracle_potential,
        'oracle_realization_pct': oracle_realization,
        'accuracy': correct_predictions / len(samples),
        'lstm_selected': model_choices['lstm'],
        'gameformer_selected': model_choices['gameformer'],
        'lstm_ratio': model_choices['lstm'] / len(samples),
        'gameformer_ratio': model_choices['gameformer'] / len(samples),
        'n_samples': len(samples),
        'threshold': threshold
    }
    
    return results, all_scores, all_true_labels


# ===========================
# 6. 阈值搜索
# ===========================
def search_optimal_threshold(model, samples, norm_stats, device='cuda'):
    """搜索最优阈值"""
    
    # 先获取所有分数
    feature_keys = norm_stats['feature_keys']
    all_scores = []
    
    model.eval()
    with torch.no_grad():
        for sample in samples:
            features = extract_features(sample, feature_keys, norm_stats)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            score = model(features_tensor).item()
            all_scores.append(score)
    
    all_scores = np.array(all_scores)
    
    print(f"\n📊 Score Distribution:")
    print(f"   Mean:  {all_scores.mean():>8.4f}")
    print(f"   Std:   {all_scores.std():>8.4f}")
    print(f"   Min:   {all_scores.min():>8.4f}")
    print(f"   Q25:   {np.percentile(all_scores, 25):>8.4f}")
    print(f"   Q50:   {np.percentile(all_scores, 50):>8.4f}")
    print(f"   Q75:   {np.percentile(all_scores, 75):>8.4f}")
    print(f"   Max:   {all_scores.max():>8.4f}\n")
    
    # 搜索阈值
    min_score = all_scores.min()
    max_score = all_scores.max()
    thresholds = np.linspace(min_score, max_score, 100)
    
    print(f"🔍 Searching {len(thresholds)} thresholds in range [{min_score:.2f}, {max_score:.2f}]...\n")
    
    results_list = []
    for thresh in thresholds:
        result, _, _ = evaluate_ranking_gate(model, samples, norm_stats, thresh, device)
        results_list.append(result)
    
    # 找最小 FDE
    best_idx = np.argmin([r['fde'] for r in results_list])
    best_result = results_list[best_idx]
    
    print(f"🎯 Optimal Threshold Found:")
    print(f"   Threshold:          {best_result['threshold']:.4f}")
    print(f"   FDE:                {best_result['fde']:.4f} m")
    print(f"   Regret:             {best_result['regret']:.4f} m")
    print(f"   Accuracy:           {best_result['accuracy']:.4f}")
    print(f"   LSTM ratio:         {best_result['lstm_ratio']*100:.1f}%")
    print(f"   GMF ratio:          {best_result['gameformer_ratio']*100:.1f}%")
    print(f"   Oracle realization: {best_result['oracle_realization_pct']:.1f}%\n")
    
    return best_result, results_list


# ===========================
# 7. 可视化
# ===========================
def plot_ranking_results(history, threshold_results, all_scores, all_labels, 
                        best_result, output_dir='./eval_out'):
    """可视化排序学习结果"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 子图1: 训练历史 - Loss
    ax1 = axes[0, 0]
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training History - Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 子图2: 训练历史 - Accuracy
    ax2 = axes[0, 1]
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training History - Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 子图3: 分数分布
    ax3 = axes[0, 2]
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    scores_gmf_better = all_scores[all_labels > 0]
    scores_lstm_better = all_scores[all_labels < 0]
    
    ax3.hist(scores_gmf_better, bins=50, alpha=0.6, label='GMF Better (True)', 
            color='red', edgecolor='black')
    ax3.hist(scores_lstm_better, bins=50, alpha=0.6, label='LSTM Better (True)', 
            color='blue', edgecolor='black')
    ax3.axvline(best_result['threshold'], color='green', linestyle='--', linewidth=2,
               label=f'Threshold={best_result["threshold"]:.2f}')
    ax3.axvline(0, color='black', linestyle=':', linewidth=2, label='Zero')
    ax3.set_xlabel('Ranking Score', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Score Distribution by True Label', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 子图4: 阈值扫描 - FDE
    ax4 = axes[1, 0]
    thresholds = [r['threshold'] for r in threshold_results]
    fdes = [r['fde'] for r in threshold_results]
    best_idx = np.argmin(fdes)
    
    ax4.plot(thresholds, fdes, 'o-', linewidth=2, markersize=3, color='steelblue')
    ax4.plot(thresholds[best_idx], fdes[best_idx], 'r*', markersize=20,
            label=f'Best (t={thresholds[best_idx]:.2f}, FDE={fdes[best_idx]:.4f})')
    ax4.axhline(best_result['gameformer_baseline_fde'], color='orange', 
               linestyle='--', linewidth=2, label='GameFormer Baseline')
    ax4.set_xlabel('Threshold', fontsize=12)
    ax4.set_ylabel('FDE (m)', fontsize=12)
    ax4.set_title('FDE vs Threshold', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 子图5: LSTM 选择率
    ax5 = axes[1, 1]
    lstm_ratios = [r['lstm_ratio'] * 100 for r in threshold_results]
    
    ax5.plot(thresholds, lstm_ratios, 'o-', linewidth=2, markersize=3, color='purple')
    ax5.axhline(27.5, color='green', linestyle='--', linewidth=2, 
               label='Oracle LSTM ratio (27.5%)')
    ax5.plot(thresholds[best_idx], lstm_ratios[best_idx], 'r*', markersize=20,
            label=f'Best ({lstm_ratios[best_idx]:.1f}%)')
    ax5.set_xlabel('Threshold', fontsize=12)
    ax5.set_ylabel('LSTM Selection Ratio (%)', fontsize=12)
    ax5.set_title('LSTM Selection vs Threshold', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 子图6: Oracle 实现率
    ax6 = axes[1, 2]
    oracle_pcts = [r['oracle_realization_pct'] for r in threshold_results]
    
    ax6.plot(thresholds, oracle_pcts, 'o-', linewidth=2, markersize=3, color='coral')
    ax6.axhline(0, color='black', linestyle='-', linewidth=1)
    ax6.plot(thresholds[best_idx], oracle_pcts[best_idx], 'r*', markersize=20,
            label=f'Best ({oracle_pcts[best_idx]:.1f}%)')
    ax6.set_xlabel('Threshold', fontsize=12)
    ax6.set_ylabel('Oracle Realization (%)', fontsize=12)
    ax6.set_title('Oracle Realization vs Threshold', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ranking_gate_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to {output_dir}/ranking_gate_analysis.png")
    plt.close()


# ===========================
# 主入口
# ===========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, 
                       default='./eval_out/fusion_stats_with_meta_features.json')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='./eval_out')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}\n")
    
    # 加载数据
    print(f"📂 Loading data from {args.data}...")
    with open(args.data, 'r') as f:
        all_samples = json.load(f)
    print(f"✅ Loaded {len(all_samples)} samples\n")
    
    # 数据划分
    train_samples, test_samples = train_test_split(all_samples, test_size=0.2, random_state=42)
    train_samples, val_samples = train_test_split(train_samples, test_size=0.15, random_state=42)
    
    print(f"📊 Dataset split:")
    print(f"   Train: {len(train_samples)}")
    print(f"   Val:   {len(val_samples)}")
    print(f"   Test:  {len(test_samples)}")
    
    # 创建数据集
    train_dataset = RankingDataset(train_samples)
    val_dataset = RankingDataset(val_samples)
    
    norm_stats = train_dataset.get_normalization_stats()
    val_dataset.mean, val_dataset.std = norm_stats['mean'], norm_stats['std']
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    model = RankingModel(input_dim=len(train_dataset.feature_keys)).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model architecture:")
    print(f"   Input dim:    {len(train_dataset.feature_keys)}")
    print(f"   Hidden dims:  [128, 64, 32]")
    print(f"   Total params: {total_params:,}")
    
    # 训练
    history = train_ranking_model(model, train_loader, val_loader, 
                                  args.epochs, args.lr, device)
    
    # 阈值搜索
    print("="*80)
    print("🔍 Searching Optimal Threshold")
    print("="*80)
    
    best_result, threshold_results = search_optimal_threshold(
        model, all_samples, norm_stats, device
    )
    
    # 用最优阈值评估
    print("\n" + "="*80)
    print("🎯 Final Evaluation with Optimal Threshold")
    print("="*80 + "\n")
    
    final_results, all_scores, all_labels = evaluate_ranking_gate(
        model, all_samples, norm_stats, best_result['threshold'], device
    )
    
    print(f"📊 Final Results:")
    print(f"   FDE:                {final_results['fde']:.4f} m")
    print(f"   vs GameFormer:      {final_results['fde_improvement_vs_gmf']:.4f} m ({final_results['fde_improvement_vs_gmf']/final_results['gameformer_baseline_fde']*100:+.2f}%)")
    print(f"   Oracle Realization: {final_results['oracle_realization_pct']:.1f}%")
    print(f"   Accuracy:           {final_results['accuracy']:.4f}")
    print(f"   LSTM ratio:         {final_results['lstm_ratio']*100:.1f}%")
    print(f"   Regret:             {final_results['regret']:.4f} m\n")
    
    # 可视化
    plot_ranking_results(history, threshold_results, all_scores, all_labels,
                        best_result, args.output_dir)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'normalization': norm_stats,
        'optimal_threshold': best_result['threshold'],
        'input_dim': len(train_dataset.feature_keys),
        'hidden_dims': [128, 64, 32],
        'dropout': 0.3
    }, f"{args.output_dir}/ranking_gate.pth")
    
    # 保存结果
    with open(f"{args.output_dir}/ranking_gate_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✅ Model and results saved to {args.output_dir}/\n")
    
    print("="*80)
    print("✅ RANKING GATE TRAINING COMPLETED!")
    print("="*80)

