# Dynamic Model Selection for Trajectory Prediction

> 基于论文《Dynamic Model Selection for Trajectory Prediction via Pairwise Ranking and Meta-Features》构建的科研级开源实现。该仓库提供三专家动态门控、元特征抽取与 LLM 语义监督的完整代码路径，可在 nuPlan-mini (1,287 场景) 上复现实验结果。

## 项目概览
- **课题动机**：单一模型在长尾驾驶情境下容易失效，需在物理可行性与交互理解之间做出折衷。
- **方法核心**：构建 LSTM–KF（物理专家）、Scene-conditioned Transformer（长尾专家）与 GameFormer（SOTA 专家）的三专家集合，使用内部信号组成的元特征驱动 Pairwise Ranking 门控，再由 LLM 监督高风险场景。
- **主要成果**：nuPlan-mini 验证集上 FDE 从 2.835 m 降至 2.567 m（↓9.5%），实现 57.8% 的 Oracle Realization；在 open-loop 仿真中左转场景 FDE 约降低 10%。
- **关键脚本**：`meta_features_extractor.py`、`gate_ranking.py`、`llm_enhanced_gate.py`、`Planner/llm_gate_planner.py` 以及 `scripts/run_llm_gate.sh`。

## 核心贡献
1. **元特征驱动的门控表征**：首次系统引入模型内部信号（不确定性、输入稳定性、物理违规率）替代纯几何指标，显著提升特征–误差相关性。
2. **Pairwise Ranking 门控**：以 RankNet 风格的相对排序替代回归/分类，避免尺度敏感性，直接优化“哪位专家更优”的决策。
3. **LLM 语义监管**：通过 LLM Supervisor 对低置信度或语义高风险样本触发覆盖，34% 场景获得语义复核，提升极端情形可靠性。
4. **Oracle Gap 量化与闭环验证**：定义 Oracle Realization Rate (ORR) 衡量可实现的性能上限，并在 nuPlan open-loop 仿真中验证离线收益的可迁移性。

## 方法框架
### Tri-Expert Ensemble
- `train_predictor_lstm.py`：训练 Seq2Seq LSTM + Kalman Filter，保证动力学一致性（平均 FDE 8.12 m）。
- `train_scene_conditioned.py`：训练场景条件化 GameFormer，针对长尾交互（平均 FDE 7.07 m）。
- 预训练 GameFormer 作为通用专家（FDE 2.84 m），负责 68.7% 场景。

### Meta-Feature Pipeline
- `meta_features_extractor.py` 通过 **MC Dropout**、输入扰动与物理违规检测生成 20+ 个元特征（uncertainty/stability/violation 比例）。
- `extract_meta_features_dataset.py` 聚合几何指标与元特征，产出 `eval_out/fusion_stats_with_scene_conditioned.json` 供门控训练使用。

### Pairwise Ranking Gate
- `gate_ranking.py` 采用 RankNet 损失（±1 标签表示专家相对优劣），并自动标准化特征、输出最优阈值。
- 训练完成的检查点与统计写入 `eval_out/ranking_gate.pth` 和 `ranking_gate_results.json`，同时生成可视化图 `ranking_gate_analysis.png`。

### LLM-Enhanced 决策
- `llm_enhanced_gate.py` 将 TriExpert logits 与 LLM Supervisor 输出融合：当置信度低于 0.4 或语义提示“高风险”时触发 LLM 覆盖，必要时再调用物理验证器。
- `Planner/llm_gate_planner.py` 在 nuPlan 仿真中加载三专家 checkpoint、Ranking Gate 参数与语义词表 (`clusters_200/semantic_vocab.json`)，实现在线动态切换。

## 实验指标
| 模型 | FDE↓ (m) | ADE↓ (m) | Oracle Realization |
| --- | --- | --- | --- |
| LSTM–KF (Physics) | 8.117 ± 0.15 | 2.820 ± 0.09 | – |
| Transformer (Long-tail) | 7.066 ± 0.12 | 2.574 ± 0.08 | – |
| GameFormer (Baseline) | 2.835 ± 0.04 | 1.469 ± 0.02 | 0% |
| **LLM-Enhanced Gate (Ours)** | **2.567 ± 0.03** | **1.255 ± 0.02** | **57.8%** |

- **Open-loop 仿真**：左转场景 `c1bf7374695a5c47` FDE 从 18.17 m 降至 16.37 m；LLM 触发率约 34%。
- **专家选择占比**：GameFormer 68.7%，Scene-conditioned 25.3%，LSTM 6.0%（主要保障物理可行性）。

## 代码结构速览
```
multi_agent_package/
├── Planner/llm_gate_planner.py        # nuPlan planner 入口，封装三专家与 LLM 逻辑
├── GameFormer/                        # GameFormer & Scene-conditioned 训练代码
├── scripts/run_llm_gate.sh            # 复现 open-loop 仿真的示例脚本
├── meta_features_extractor.py         # 元特征抽取器
├── extract_meta_features_dataset.py   # 生成门控训练数据
├── gate_ranking.py                    # Pairwise Ranking gate 训练
├── llm_enhanced_gate.py               # 推理期 LLM 增强门控
├── training_log/, pretrained_models/  # 检查点与日志
└── my_paper.tex                       # 论文稿件
```

## 快速开始
### 环境配置
1. Python ≥ 3.10，PyTorch ≥ 2.1（需 CUDA 支持），安装 nuPlan-devkit 依赖。
2. 建议使用 Conda 环境 `nuplan`，并将 `GameFormer`, `nuplan-devkit` 路径加入 `PYTHONPATH`。
3. 额外依赖（见 `requirements.txt`）：
   ```bash
   pip install -r requirements.txt  # 包含 theseus-ai==0.1.3
   ```

### 数据与资源
- 下载 nuPlan-mini DB、地图与 devkit，目录示例：
  ```
  export NUPLAN_DATA_ROOT=/mnt/d/paper/nuplan
  export NUPLAN_MAPS_ROOT=/mnt/d/paper/nuplan/nuplan-maps-v1.0/maps
  export NUPLAN_EXP_ROOT=/home/hamster/GameFormer-Planner/testing_log
  ```
- 预训练模型：将 LSTM、Scene-conditioned、GameFormer checkpoint 分别放入 `training_log/` 与 `pretrained_models/`，路径在 `agent.md` 中列出。

### 训练流水线
1. **训练专家**  
   ```bash
   python train_predictor_lstm.py          # 生成 LSTM–KF
   python train_scene_conditioned.py       # 训练场景条件化模型
   # GameFormer 可复用官方 checkpoint
   ```
2. **抽取元特征**  
   ```bash
   python extract_meta_features_dataset.py \
     --gameformer_ckpt <path> \
     --scene_ckpt <path> \
     --lstm_ckpt <path> \
     --output ./eval_out/fusion_stats_with_scene_conditioned.json
   ```
3. **训练 Ranking Gate**  
   ```bash
   python gate_ranking.py \
     --data ./eval_out/fusion_stats_with_scene_conditioned.json \
     --epochs 50 --batch_size 64 --output_dir ./eval_out
   ```
   训练结束将获得 `ranking_gate.pth`、`ranking_gate_results.json` 与可视化图表。
4. **LLM 增强推理**  
   - 准备 `LLM_GATE_VOCAB`（语义词表）与 `LLM_GATE_CACHE`（缓存语义标签）。  
   - `llm_enhanced_gate.py` 可离线运行（禁用 LLM）或接入在线 LLM API。

### NuPlan Open-Loop 仿真
使用 `scripts/run_llm_gate.sh` 作为模板，关键环境变量：
```
export LLM_GATE_GAMEFORMER=.../model_epoch_18_valADE_1.7272.pth
export LLM_GATE_SCENE=.../SceneConditioned_v4/model_best_ADE_3.356.pth
export LLM_GATE_LSTM=.../LSTM_v2/best_seq2seq_lstm.pth
export LLM_GATE_RANKING=.../eval_out/tri_gate_v7.pth
export LLM_GATE_VOCAB=.../semantic_vocab.json
export LLM_GATE_CACHE=.../llm_semantic_cache.json
```
随后调用 `nuplan.planning.script.run_simulation`，选择 `planner=llm_gate_planner` 并设置场景过滤器即可复现论文实验。

### 评估与分析
- `compute_open_loop_metrics.py`、`compute_scene_conditioned_metrics.py`：解析 NuPlan JSON/msgpack 结果并输出 ADE/FDE。
- `gate_ranking.py` 会自动输出 ORR、专家选择比例、阈值扫描图，方便与论文指标对齐。
- `agent.md` 提供强制专家、左转场景等分析脚本，可用于再现附录实验。

## 研究复现提示
- **Oracle Realization**：`eval_out/ranking_gate_results.json` 中保存了 baseline/oracle/gate FDE，可直接计算 ORR。
- **LLM 触发策略**：`llm_enhanced_gate.py` 中的 `trigger_threshold`、`min_lstm_prob`、`min_scene_prob` 与 `LLM_GATE_FORCE_EXPERT` 环境变量可重现论文的覆盖策略。
- **长尾切片分析**：使用 `analysis_runs/*.msgpack.xz` 与 `compute_open_loop_metrics.py` 验证强制专家在左转、cut-in、遮挡情境下的表现。

## 引用
如果本项目或论文对您的研究有所帮助，请引用：
```bibtex
@article{lu2026dynamic,
  title   = {Dynamic Model Selection for Trajectory Prediction via Pairwise Ranking and Meta-Features},
  author  = {Lu, Bowen},
  journal = {AAAI Conference on Artificial Intelligence},
  year    = {2026}
}
```

## 交流
- 问题或特性请求请直接创建 GitHub Issue。
- 亦可通过论文作者邮箱 `bluu0021@student.monash.edu` 联系。

本 README 将随着实验更新持续维护，欢迎贡献改进门控、语义监督或 planner 集成的相关内容。
