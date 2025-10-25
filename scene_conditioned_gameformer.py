"""
场景条件GameFormer (Scene-Conditioned GameFormer)
==============================================

目标：在GameFormer基础上添加语义条件embedding，使模型能够利用场景语义信息

核心改进：
1. 添加语义embedding层（scene_type, scene_subtype, keywords）
2. 将语义特征融合到encoder输出
3. 在困难场景（intersection/cut_in）中提供额外指导

训练策略：
- 使用LLM标注的语义标签作为条件
- 在长尾场景上过采样（critical/high样本）
- 联合优化预测精度和语义对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from GameFormer.predictor import Encoder, Decoder, NeuralPlanner


class SemanticEmbedding(nn.Module):
    """
    语义条件embedding模块
    
    输入：
    - scene_type: 场景类型索引 [batch_size]
    - scene_keywords: 关键词多热编码 [batch_size, vocab_size]
    
    输出：
    - semantic_embedding: [batch_size, embed_dim]
    """
    
    def __init__(self, vocab_size, embed_dim=256, dropout=0.1):
        super(SemanticEmbedding, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # 场景类型embedding（类别变量）
        # 预定义的场景类型（与LLM标注对齐）
        self.scene_types = [
            'intersection', 'cut_in', 'merging', 'yielding', 
            'occlusion', 'congestion', 'high_speed', 'other'
        ]
        self.num_scene_types = len(self.scene_types)
        self.scene_type_embedding = nn.Embedding(self.num_scene_types, embed_dim // 2)
        
        # 关键词embedding（多热编码）
        # 使用线性层将关键词多热向量映射到embedding空间
        self.keyword_embedding = nn.Sequential(
            nn.Linear(vocab_size, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, scene_type_ids, keyword_vectors):
        """
        Args:
            scene_type_ids: [batch_size] 场景类型索引
            keyword_vectors: [batch_size, vocab_size] 关键词多热向量
        
        Returns:
            semantic_embedding: [batch_size, embed_dim]
        """
        # 场景类型embedding
        scene_type_emb = self.scene_type_embedding(scene_type_ids)  # [B, embed_dim//2]
        
        # 关键词embedding
        keyword_emb = self.keyword_embedding(keyword_vectors)  # [B, embed_dim//2]
        
        # 拼接并融合
        combined = torch.cat([scene_type_emb, keyword_emb], dim=-1)  # [B, embed_dim]
        semantic_embedding = self.fusion(combined)
        
        return semantic_embedding


class SceneConditionedEncoder(nn.Module):
    """
    场景条件编码器
    
    在原始Encoder基础上融合语义条件
    """
    
    def __init__(self, vocab_size, dim=256, layers=6, heads=8, dropout=0.1):
        super(SceneConditionedEncoder, self).__init__()
        
        # 原始GameFormer编码器
        self.base_encoder = Encoder(dim, layers, heads, dropout)
        
        # 语义embedding
        self.semantic_embedding = SemanticEmbedding(vocab_size, dim, dropout)
        
        # 条件融合层（将语义特征融合到编码中）
        self.condition_fusion = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.condition_norm = nn.LayerNorm(dim)
    
    def forward(self, inputs, scene_type_ids, keyword_vectors):
        """
        Args:
            inputs: 原始输入（与GameFormer相同）
            scene_type_ids: [batch_size] 场景类型索引
            keyword_vectors: [batch_size, vocab_size] 关键词多热向量
        
        Returns:
            encoder_outputs: 编码输出（与GameFormer格式相同，但融合了语义信息）
        """
        # 1. 基础编码
        encoder_outputs = self.base_encoder(inputs)
        encoding = encoder_outputs['encoding']  # [B, N, dim]
        mask = encoder_outputs['mask']  # [B, N]
        
        # 2. 语义embedding
        semantic_emb = self.semantic_embedding(scene_type_ids, keyword_vectors)  # [B, dim]
        semantic_emb = semantic_emb.unsqueeze(1)  # [B, 1, dim]
        
        # 3. 条件融合（使用cross-attention）
        # 将语义作为query，编码作为key/value
        conditioned_encoding, _ = self.condition_fusion(
            query=semantic_emb.expand(-1, encoding.shape[1], -1),  # [B, N, dim]
            key=encoding,
            value=encoding,
            key_padding_mask=mask
        )
        
        # 残差连接 + LayerNorm
        encoding = self.condition_norm(encoding + conditioned_encoding)
        
        # 更新输出
        encoder_outputs['encoding'] = encoding
        encoder_outputs['semantic_embedding'] = semantic_emb.squeeze(1)  # [B, dim]
        
        return encoder_outputs


class SceneConditionedGameFormer(nn.Module):
    """
    场景条件GameFormer
    
    架构：
    1. SceneConditionedEncoder：融合语义条件
    2. Decoder：与原始GameFormer相同
    3. NeuralPlanner：与原始GameFormer相同
    """
    
    def __init__(
        self,
        vocab_size,
        encoder_layers=6,
        decoder_levels=3,
        modalities=6,
        neighbors=10
    ):
        super(SceneConditionedGameFormer, self).__init__()
        
        self.vocab_size = vocab_size
        
        # 场景条件编码器
        self.encoder = SceneConditionedEncoder(
            vocab_size=vocab_size,
            layers=encoder_layers
        )
        
        # 与原始GameFormer共享的decoder和planner
        self.decoder = Decoder(neighbors, modalities, decoder_levels)
        self.planner = NeuralPlanner()
    
    def forward(self, inputs, scene_type_ids=None, keyword_vectors=None):
        """
        Args:
            inputs: 原始输入（与GameFormer相同）
            scene_type_ids: [batch_size] 场景类型索引（可选）
            keyword_vectors: [batch_size, vocab_size] 关键词多热向量（可选）
        
        Returns:
            decoder_outputs: 预测输出
            ego_plan: 自车轨迹
        """
        # 如果未提供语义条件，使用默认值（'other'场景，全零关键词）
        if scene_type_ids is None:
            batch_size = inputs['ego_agent_past'].shape[0]
            scene_type_ids = torch.full(
                (batch_size,), 
                7,  # 'other' 的索引
                dtype=torch.long,
                device=inputs['ego_agent_past'].device
            )
        
        if keyword_vectors is None:
            batch_size = inputs['ego_agent_past'].shape[0]
            keyword_vectors = torch.zeros(
                (batch_size, self.vocab_size),
                dtype=torch.float32,
                device=inputs['ego_agent_past'].device
            )
        
        # 编码（融合语义条件）
        encoder_outputs = self.encoder(inputs, scene_type_ids, keyword_vectors)
        
        # 解码（与原始GameFormer相同）
        route_lanes = encoder_outputs['route_lanes']
        initial_state = encoder_outputs['actors'][:, 0, -1]
        decoder_outputs, env_encoding = self.decoder(encoder_outputs)
        ego_plan = self.planner(env_encoding, route_lanes, initial_state)
        
        return decoder_outputs, ego_plan
    
    @staticmethod
    def from_pretrained(pretrained_path, vocab_size, **kwargs):
        """
        从预训练的GameFormer初始化
        
        策略：
        1. 加载GameFormer权重到encoder.base_encoder
        2. 随机初始化semantic_embedding和condition_fusion
        3. 冻结部分层（可选）
        
        Args:
            pretrained_path: 预训练模型路径
            vocab_size: 语义词表大小
            **kwargs: 其他参数
        
        Returns:
            model: 初始化好的模型
        """
        # 创建模型
        model = SceneConditionedGameFormer(vocab_size, **kwargs)
        
        # 加载预训练权重
        print(f"Loading pretrained GameFormer from {pretrained_path}...")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 提取GameFormer的权重
        gameformer_state_dict = checkpoint.get('model', checkpoint)
        
        # 加载到base_encoder, decoder, planner
        model_state_dict = model.state_dict()
        pretrained_dict = {}
        
        for k, v in gameformer_state_dict.items():
            # encoder.* -> encoder.base_encoder.* (only replace the first prefix)
            if k.startswith('encoder.'):
                new_k = k.replace('encoder.', 'encoder.base_encoder.', 1)
                if new_k in model_state_dict:
                    pretrained_dict[new_k] = v
            # decoder.*, planner.* 直接复制
            elif k.startswith('decoder.') or k.startswith('planner.'):
                if k in model_state_dict:
                    pretrained_dict[k] = v
        
        print(f"Loaded {len(pretrained_dict)}/{len(model_state_dict)} parameters from pretrained model")
        
        # 更新模型权重
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict)
        
        print("✅ Successfully initialized from pretrained GameFormer")
        print(f"   - Base encoder: {len([k for k in pretrained_dict if 'base_encoder' in k])} params")
        print(f"   - Decoder: {len([k for k in pretrained_dict if 'decoder' in k])} params")
        print(f"   - Planner: {len([k for k in pretrained_dict if 'planner' in k])} params")
        print(f"   - Semantic (random init): {len([k for k in model_state_dict if 'semantic' in k or 'condition' in k])} params")
        
        return model
    
    def freeze_base_encoder(self):
        """冻结基础编码器（仅训练语义部分）"""
        for param in self.encoder.base_encoder.parameters():
            param.requires_grad = False
        print("✅ Froze base encoder parameters")
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
        print("✅ Unfroze all parameters")


# ============================================================================
# 辅助函数：场景类型和关键词转换
# ============================================================================

def get_scene_type_mapping():
    """获取场景类型映射"""
    scene_types = [
        'intersection', 'cut_in', 'merging', 'yielding', 
        'occlusion', 'congestion', 'high_speed', 'other'
    ]
    return {s: i for i, s in enumerate(scene_types)}


def encode_scene_semantic(labels, vocab_dict):
    """
    将LLM标注转换为模型输入
    
    Args:
        labels: LLM标注结果列表（字典）
        vocab_dict: 语义词表 {keyword: id}
    
    Returns:
        scene_type_ids: [N] 场景类型索引
        keyword_vectors: [N, vocab_size] 关键词多热向量
    """
    scene_type_mapping = get_scene_type_mapping()
    vocab_size = len(vocab_dict)
    
    scene_type_ids = []
    keyword_vectors = []
    
    for label in labels:
        # 场景类型
        scene_type = label.get('scene_type', 'other')
        scene_type_id = scene_type_mapping.get(scene_type, 7)  # 默认'other'=7
        scene_type_ids.append(scene_type_id)
        
        # 关键词多热向量
        keywords = label.get('semantic_keywords', [])
        keyword_vec = torch.zeros(vocab_size)
        for kw in keywords:
            if kw in vocab_dict:
                keyword_vec[vocab_dict[kw]] = 1.0
        keyword_vectors.append(keyword_vec)
    
    scene_type_ids = torch.tensor(scene_type_ids, dtype=torch.long)
    keyword_vectors = torch.stack(keyword_vectors)
    
    return scene_type_ids, keyword_vectors


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import json
    
    print("="*80)
    print("Testing Scene-Conditioned GameFormer")
    print("="*80)
    
    # 1. 加载语义词表
    vocab_path = './eval_out/clusters/semantic_vocab.json'
    if Path(vocab_path).exists():
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
            vocab_size = vocab_data['vocab_size']
            vocab_dict = vocab_data['keyword_to_id']
        print(f"\n✅ Loaded semantic vocabulary: {vocab_size} keywords")
    else:
        print(f"\n⚠️  Vocabulary not found, using dummy vocab_size=100")
        vocab_size = 100
        vocab_dict = {}
    
    # 2. 创建模型
    print(f"\n🏗️  Creating Scene-Conditioned GameFormer...")
    model = SceneConditionedGameFormer(
        vocab_size=vocab_size,
        encoder_layers=6,
        decoder_levels=3,
        modalities=6,
        neighbors=10
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # 3. 测试前向传播
    print(f"\n🧪 Testing forward pass...")
    
    batch_size = 2
    
    # 构造dummy输入
    dummy_inputs = {
        'ego_agent_past': torch.randn(batch_size, 21, 7),
        'neighbor_agents_past': torch.randn(batch_size, 10, 21, 11),
        'map_lanes': torch.randn(batch_size, 100, 50, 7),
        'map_crosswalks': torch.randn(batch_size, 50, 30, 3),
        'route_lanes': torch.randn(batch_size, 10, 50, 3)
    }
    
    # 场景条件（模拟LLM标注）
    scene_type_ids = torch.tensor([0, 1])  # intersection, cut_in
    keyword_vectors = torch.randn(batch_size, vocab_size).sigmoid()  # 随机关键词
    
    # 前向传播
    with torch.no_grad():
        decoder_outputs, ego_plan = model(
            dummy_inputs,
            scene_type_ids=scene_type_ids,
            keyword_vectors=keyword_vectors
        )
    
    print(f"✅ Forward pass successful")
    print(f"   Ego plan shape: {ego_plan.shape}")  # [B, 80, 3]
    print(f"   Decoder outputs keys: {list(decoder_outputs.keys())}")
    
    # 4. 测试从预训练加载
    pretrained_path = './training_log/Exp1/model_epoch_17_valADE_1.97.pth'
    from pathlib import Path
    if Path(pretrained_path).exists():
        print(f"\n🔄 Testing loading from pretrained...")
        model_pretrained = SceneConditionedGameFormer.from_pretrained(
            pretrained_path,
            vocab_size=vocab_size
        )
        print(f"✅ Pretrained loading successful")
    else:
        print(f"\n⚠️  Pretrained model not found at {pretrained_path}")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)
