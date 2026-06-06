import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSFormer(nn.Module):
    """
    阶段 2 核心成果：高低频物理空间解耦多头 SFormer 包装器
    f_cut = 5.0 GHz -> 低频 833 点，高频 1667 点
    """
    def __init__(self, base_model, embedding_dim=128, num_channels=20):
        super(MultiHeadSFormer, self).__init__()
        self.base_model = base_model  # 你原有的 SFormer 骨干网络
        self.num_channels = num_channels
        
        # 移除或接管原有基模型的最后一层输出层（假设基模型原输出为 embedding_dim 维的特征）
        # 1. 低频回归头：专注拟合 0 ~ 5GHz 的密集谐振突变
        self.low_freq_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_channels * 833)
        )
        
        # 2. 高频回归头：专注拟合 5GHz ~ 15GHz 的单调介质损耗衰减
        self.high_freq_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_channels * 1667)
        )

    def forward(self, adj_matrices, node_features, global_features, pcb_ids, freqs):
        # 1. 调用你原本的 SFormer 提取器提取全局几何与网络特征
        # 注意：这里需要确保你的原 base_model 可以配置为只输出全连接前的特征向量（例如形状为 [Batch, embedding_dim]）
        shared_features = self.base_model.extract_features(adj_matrices, node_features, global_features, pcb_ids, freqs)
        
        batch_size = shared_features.size(0)
        
        # 2. 双头并行前向传播
        low_out = self.low_freq_head(shared_features).view(batch_size, self.num_channels, 833)
        high_out = self.high_freq_head(shared_features).view(batch_size, self.num_channels, 1667)
        
        # 3. 动态拼接成完整的 2500 维输出，用于兼容原有的评估指标与 plot.py 脚本
        full_prediction = torch.cat([low_out, high_out], dim=-1)
        
        return full_prediction, low_out, high_out

def compute_physics_informed_loss(low_pred, high_pred, low_target, high_target, alpha=1.0, beta=1.0, gamma=5.0):
    """
    阶段 2 核心成果：带 5.0 GHz 边界连续性惩罚的混合损失函数
    """
    # 1. 高低频段各自的标准回归 Loss (MSE)
    loss_low = F.mse_loss(low_pred, low_target)
    loss_high = F.mse_loss(high_pred, high_target)
    
    # 2. 物理交界衔接点连续性损失 (Boundary Continuity Loss)
    # 约束：低频段的最后一个频点(第832点) 必须等于 高频段的第一个频点(第833点)
    boundary_low_edge = low_pred[..., -1]
    boundary_high_edge = high_pred[..., 0]
    loss_boundary = F.l1_loss(boundary_low_edge, boundary_high_edge)
    
    # Total Loss 融合
    total_loss = alpha * loss_low + beta * loss_high + gamma * loss_boundary
    
    return total_loss, loss_low.item(), loss_high.item(), loss_boundary.item()