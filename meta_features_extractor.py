"""
元特征提取器 (Meta-Features Extractor)
从模型内部信号提取不确定性、稳定性等元特征
用于更精准的门控决策
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy


# ===========================
# 1. MC Dropout 不确定性估计
# ===========================
def enable_dropout(model):
    """递归开启所有 Dropout 层（用于 MC Dropout）"""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_forward_lstm(model, ego_past, K=8):
    """
    LSTM-KF 的 MC Dropout 前向传播
    
    Args:
        model: LSTM 模型
        ego_past: [B, T, D] 输入历史轨迹
        K: MC 采样次数
    
    Returns:
        mean_pred: [B, T_future, 2] 平均预测
        total_var: [B] 整体轨迹方差
        final_var: [B] 末帧位置方差
    """
    model.eval()
    enable_dropout(model)  # 强制开启 dropout
    
    outputs = []
    with torch.no_grad():
        for _ in range(K):
            pred = model(ego_past)  # [B, T_future, 2]
            outputs.append(pred.detach())
    
    # [K, B, T_future, 2]
    outputs_stack = torch.stack(outputs, dim=0)
    
    # 平均预测
    mean_pred = outputs_stack.mean(dim=0)
    
    # 整体方差：所有时间步和坐标的方差均值
    total_var = outputs_stack.var(dim=0).mean(dim=(1, 2))  # [B]
    
    # 末帧方差：最后一个时间步的位置方差
    final_var = outputs_stack[:, :, -1, :].var(dim=0).norm(dim=-1)  # [B]
    
    model.train()  # 恢复训练模式
    
    return mean_pred, total_var.cpu().numpy(), final_var.cpu().numpy()


def mc_forward_scene_conditioned(model, inputs, K=8):
    """
    Scene-Conditioned GameFormer 的 MC Dropout 前向传播
    """
    model.eval()
    enable_dropout(model)

    ego_plans = []
    with torch.no_grad():
        for _ in range(K):
            inputs_clone = {k: v.clone() for k, v in inputs.items()}
            _, ego_plan = model(inputs_clone)
            ego_plans.append(ego_plan[..., :2].detach())

    plans_stack = torch.stack(ego_plans, dim=0)
    mean_pred = plans_stack.mean(dim=0)
    total_var = plans_stack.var(dim=0).mean(dim=(1, 2))
    final_var = plans_stack[:, :, -1, :].var(dim=0).norm(dim=-1)

    model.train()
    return mean_pred, total_var.cpu().numpy(), final_var.cpu().numpy()


def mc_forward_gameformer(model, inputs, K=8):
    """
    GameFormer 的 MC Dropout 前向传播
    
    Args:
        model: GameFormer 模型
        inputs: dict - GameFormer 输入字典
        K: MC 采样次数
    
    Returns:
        mean_pred: [B, T_future, 2] 平均预测（ego plan）
        total_var: [B] 整体轨迹方差
        final_var: [B] 末帧位置方差
    """
    model.eval()
    enable_dropout(model)
    
    ego_plans = []
    with torch.no_grad():
        for _ in range(K):
            level_k_outputs, ego_plan = model(inputs)
            ego_plans.append(ego_plan[..., :2].detach())  # 只取 xy 坐标
    
    # [K, B, T_future, 2]
    ego_plans_stack = torch.stack(ego_plans, dim=0)
    
    mean_pred = ego_plans_stack.mean(dim=0)
    total_var = ego_plans_stack.var(dim=0).mean(dim=(1, 2))
    final_var = ego_plans_stack[:, :, -1, :].var(dim=0).norm(dim=-1)
    
    model.train()
    
    return mean_pred, total_var.cpu().numpy(), final_var.cpu().numpy()


# ===========================
# 2. 输入扰动稳定性
# ===========================
def input_perturbation_stability(model, input_data, model_type='lstm', n_perturb=3, noise_scale=0.1):
    """
    测试模型对输入扰动的稳定性
    
    Args:
        model: 模型
        input_data: 输入数据
        model_type: 'lstm' 或 'gameformer'
        n_perturb: 扰动次数
        noise_scale: 噪声尺度
    
    Returns:
        stability_score: [B] 输出漂移的平均范数
    """
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        # 原始预测
        if model_type == 'lstm':
            pred_original = model(input_data)
        elif model_type == 'transformer':
            pred_original = model(input_data)
        elif model_type == 'scene_conditioned':
            _, pred_original = model(input_data)
            pred_original = pred_original[..., :2]
        else:  # gameformer
            _, pred_original = model(input_data)
            pred_original = pred_original[..., :2]
        
        # 扰动预测
        for _ in range(n_perturb):
            if model_type == 'lstm':
                perturbed = input_data + torch.randn_like(input_data) * noise_scale
                pred_perturb = model(perturbed)
            elif model_type == 'transformer':
                perturbed = input_data + torch.randn_like(input_data) * noise_scale
                pred_perturb = model(perturbed)
            elif model_type == 'scene_conditioned':
                inputs_perturb = copy.deepcopy(input_data)
                inputs_perturb['ego_agent_past'] = inputs_perturb['ego_agent_past'] + \
                    torch.randn_like(inputs_perturb['ego_agent_past']) * noise_scale
                _, pred_perturb = model(inputs_perturb)
                pred_perturb = pred_perturb[..., :2]
            else:
                inputs_perturb = copy.deepcopy(input_data)
                inputs_perturb['ego_agent_past'] = inputs_perturb['ego_agent_past'] + \
                    torch.randn_like(inputs_perturb['ego_agent_past']) * noise_scale
                _, pred_perturb = model(inputs_perturb)
                pred_perturb = pred_perturb[..., :2]
            
            # 计算与原始预测的差异
            diff = (pred_perturb - pred_original).norm(dim=-1).mean(dim=1)  # [B]
            predictions.append(diff)
    
    # 平均漂移
    stability_score = torch.stack(predictions, dim=0).mean(dim=0).cpu().numpy()
    
    return stability_score


# ===========================
# 3. 物理违规检测
# ===========================
def detect_physics_violations(trajectory, dt=0.1, max_lateral_accel=5.0, max_curvature=0.5):
    """
    检测轨迹的物理违规
    
    Args:
        trajectory: [B, T, 2] 预测轨迹
        dt: 时间步长
        max_lateral_accel: 最大横向加速度 (m/s²)
        max_curvature: 最大曲率 (1/m)
    
    Returns:
        violation_ratio: [B] 违规帧的比例
        metrics: dict - 详细指标
    """
    B, T, _ = trajectory.shape
    
    # 计算速度
    vel = (trajectory[:, 1:] - trajectory[:, :-1]) / dt  # [B, T-1, 2]
    speed = vel.norm(dim=-1)  # [B, T-1]
    
    # 计算加速度
    accel = (vel[:, 1:] - vel[:, :-1]) / dt  # [B, T-2, 2]
    accel_norm = accel.norm(dim=-1)  # [B, T-2]
    
    # 计算曲率（简化：使用三点法）
    violations = []
    
    for b in range(B):
        traj_b = trajectory[b].cpu().numpy()  # [T, 2]
        
        violation_count = 0
        total_checks = 0
        
        # 检查曲率
        for t in range(1, T - 1):
            p1 = traj_b[t - 1]
            p2 = traj_b[t]
            p3 = traj_b[t + 1]
            
            # 三点计算曲率
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                # 转角
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                dot = np.dot(v1, v2)
                angle = np.arctan2(cross, dot)
                
                # 曲率估计
                curvature = abs(angle) / (dt * np.linalg.norm(v1))
                
                if curvature > max_curvature:
                    violation_count += 1
                
                total_checks += 1
        
        # 检查横向加速度（简化：使用总加速度）
        if len(accel_norm[b]) > 0:
            high_accel = (accel_norm[b] > max_lateral_accel).sum().item()
            violation_count += high_accel
            total_checks += len(accel_norm[b])
        
        violation_ratio = violation_count / max(total_checks, 1)
        violations.append(violation_ratio)
    
    return np.array(violations)


# ===========================
# 4. 注意力熵（仅 GameFormer）
# ===========================
def extract_attention_entropy(model, inputs):
    """
    提取 GameFormer 注意力机制的熵
    
    注意：需要修改 GameFormer 模型以返回注意力权重
    这里提供接口，实际使用时需要适配模型
    
    Returns:
        attention_entropy: [B] 平均注意力熵
    """
    # TODO: 需要修改 GameFormer 代码以暴露注意力权重
    # 暂时返回占位符
    B = inputs['ego_agent_past'].shape[0]
    return np.zeros(B)


# ===========================
# 5. 综合元特征提取
# ===========================
def extract_meta_features(lstm_model, scene_model, gmf_model, batch_data, device='cuda', K=8):
    """
    提取所有元特征
    
    Args:
        lstm_model: LSTM-KF 模型
        scene_model: Scene-Conditioned GameFormer 模型
        gmf_model: GameFormer 模型
        batch_data: 一个批次的数据
        device: 设备
        K: MC 采样次数
    
    Returns:
        meta_features: dict - 包含所有元特征
    """
    ego, neighbors, lanes, crosswalks, routes, ego_future, neighbors_future = batch_data
    
    # 转换为 tensor
    ego = torch.tensor(ego, dtype=torch.float32).unsqueeze(0).to(device)
    neighbors = torch.tensor(neighbors, dtype=torch.float32).unsqueeze(0).to(device)
    lanes = torch.tensor(lanes, dtype=torch.float32).unsqueeze(0).to(device)
    crosswalks = torch.tensor(crosswalks, dtype=torch.float32).unsqueeze(0).to(device)
    routes = torch.tensor(routes, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Pad GameFormer 输入
    from eval_models import pad_or_trim
    lanes = pad_or_trim(lanes, 50, dim=1)
    crosswalks = pad_or_trim(crosswalks, 30, dim=1)
    routes = pad_or_trim(routes, 50, dim=1)
    
    gmf_inputs = {
        'ego_agent_past': ego,
        'neighbor_agents_past': neighbors,
        'map_lanes': lanes,
        'map_crosswalks': crosswalks,
        'route_lanes': routes
    }
    
    # ========== LSTM 元特征 ==========
    lstm_input = ego[..., :4]
    
    # MC Dropout 不确定性
    _, lstm_total_var, lstm_final_var = mc_forward_lstm(lstm_model, lstm_input, K=K)
    
    # 输入扰动稳定性
    lstm_stability = input_perturbation_stability(lstm_model, lstm_input, model_type='lstm')
    
    # 物理违规检测
    lstm_pred = lstm_model(lstm_input).detach()
    lstm_violations = detect_physics_violations(lstm_pred)

    # ========== Scene-Conditioned 元特征 ==========
    scene_inputs = {k: v.clone() for k, v in gmf_inputs.items()}
    _, scene_total_var, scene_final_var = mc_forward_scene_conditioned(scene_model, scene_inputs, K=K)
    scene_stability = input_perturbation_stability(scene_model, scene_inputs, model_type='scene_conditioned')
    scene_forward_inputs = {k: v.clone() for k, v in gmf_inputs.items()}
    _, scene_plan = scene_model(scene_forward_inputs)
    scene_pred = scene_plan[..., :2].detach()
    scene_violations = detect_physics_violations(scene_pred)
    
    # ========== GameFormer 元特征 ==========
    # MC Dropout 不确定性
    _, gmf_total_var, gmf_final_var = mc_forward_gameformer(gmf_model, gmf_inputs, K=K)
    
    # 输入扰动稳定性
    gmf_stability = input_perturbation_stability(gmf_model, gmf_inputs, model_type='gameformer')
    
    # 物理违规检测
    _, gmf_pred = gmf_model(gmf_inputs)
    gmf_pred_xy = gmf_pred[..., :2].detach()
    gmf_violations = detect_physics_violations(gmf_pred_xy)
    
    # 组装元特征（处理 LSTM 不确定性为 0 的情况）
    lstm_unc_total = float(lstm_total_var[0])
    lstm_unc_final = float(lstm_final_var[0])
    scene_unc_total = float(scene_total_var[0])
    scene_unc_final = float(scene_final_var[0])
    gmf_unc_total = float(gmf_total_var[0])
    gmf_unc_final = float(gmf_final_var[0])
    
    # 如果 LSTM 不确定性为 0，使用 GMF 的绝对值代替比值
    if lstm_unc_total < 1e-8:
        uncertainty_ratio = gmf_unc_total  # 直接用 GMF 不确定性
    else:
        uncertainty_ratio = gmf_unc_total / lstm_unc_total
    
    meta_features = {
        'lstm_uncertainty_total': lstm_unc_total,
        'lstm_uncertainty_final': lstm_unc_final,
        'lstm_stability': float(lstm_stability[0]),
        'lstm_physics_violation': float(lstm_violations[0]),
        
        'gmf_uncertainty_total': gmf_unc_total,
        'gmf_uncertainty_final': gmf_unc_final,
        'gmf_stability': float(gmf_stability[0]),
        'gmf_physics_violation': float(gmf_violations[0]),

        'scene_conditioned_uncertainty_total': scene_unc_total,
        'scene_conditioned_uncertainty_final': scene_unc_final,
        'scene_conditioned_stability': float(scene_stability[0]),
        'scene_conditioned_physics_violation': float(scene_violations[0]),
        
        # 相对元特征（鲁棒处理）
        'uncertainty_ratio': float(uncertainty_ratio),
        'stability_ratio': float(gmf_stability[0] / (lstm_stability[0] + 1e-6)),
        'violation_diff': float(gmf_violations[0] - lstm_violations[0]),
        
        # 新增：绝对差异特征
        'stability_diff': float(gmf_stability[0] - lstm_stability[0]),
        'scene_conditioned_uncertainty_ratio_lstm': float(scene_unc_total / (lstm_unc_total + 1e-6)),
        'scene_conditioned_uncertainty_ratio_gmf': float(scene_unc_total / (gmf_unc_total + 1e-6)),
        'scene_conditioned_stability_ratio_lstm': float(scene_stability[0] / (lstm_stability[0] + 1e-6)),
        'scene_conditioned_stability_ratio_gmf': float(scene_stability[0] / (gmf_stability[0] + 1e-6)),
        'scene_conditioned_violation_diff_lstm': float(scene_violations[0] - lstm_violations[0]),
        'scene_conditioned_violation_diff_gmf': float(scene_violations[0] - gmf_violations[0]),
    }
    
    return meta_features


# ===========================
# 测试代码
# ===========================
if __name__ == "__main__":
    print("🧪 Testing Meta-Features Extractor...")
    
    # 模拟数据
    B, T_past, T_future = 1, 21, 80
    
    # 模拟轨迹
    traj = torch.randn(B, T_future, 2)
    
    # 测试物理违规检测
    violations = detect_physics_violations(traj)
    print(f"\n✅ Physics Violation Detection:")
    print(f"   Violation ratio: {violations[0]:.2%}")
    
    print("\n✅ Meta-Features Extractor ready!")
    print("\n💡 Next step: Run extract_meta_features_dataset.py to extract features for all samples.")

