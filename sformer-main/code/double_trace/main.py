import os
import json
import math
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import skrf as rf
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset

"""Double-trace S-parameter prediction with dummy nodes.

本脚本实现：
1) 从双线几何与 S4P 构图（含 2 个 dummy 节点）；
2) 节点特征包含到对面线距离；
3) 回归 10 个 S 参数的实部与虚部，共 20 个通道；
4) 保存测试结果，逐样本绘图由 plot.py 统一完成。
"""

DATA_DIR = "D:/sformer-main-back up/sformer-main/data/double_trace/data"
SAVE_DIR = "D:/sformer-main-back up/sformer-main/data_gen/double_trace1"

TRACE_WIDTH = 0.055
DIELECTRIC_THICKNESS = 0.2032

# 10 个独立 S 参数（4 端口上三角）
S_PARAM_INDICES: List[Tuple[int, int]] = [
    (0, 0),  # S11
    (0, 1),  # S12
    (0, 2),  # S13
    (0, 3),  # S14
    (1, 1),  # S22
    (1, 2),  # S23
    (1, 3),  # S24
    (2, 2),  # S33
    (2, 3),  # S34
    (3, 3),  # S44
]
S_PARAM_NAMES = [f"S{i + 1}{j + 1}" for i, j in S_PARAM_INDICES]


def set_seed(random_seed: int) -> None:
    # 固定随机种子，保证课堂演示时结果可复现。
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.mlp = nn.Linear(in_features, out_features)

    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        # GCN 规范化邻接矩阵: D^(-1/2) A D^(-1/2)
        degree_inv_sqrt = torch.sum(adjacency_matrix, dim=-1) ** -0.5
        degree_matrix = torch.diag_embed(degree_inv_sqrt)
        normalized_adjacency = torch.bmm(torch.bmm(degree_matrix, adjacency_matrix), degree_matrix)
        aggregated_features = torch.bmm(normalized_adjacency, node_features)
        projected_features = self.mlp(aggregated_features)
        return F.relu(projected_features)


class DoubleTraceTransformer(nn.Module):
    def __init__(
        self,
        in_features: int,
        global_features_dim: int,
        hidden_features: int,
        out_features: int,
        num_nodes: int,
        num_layers: int = 6,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.gcn1 = SimpleGCNLayer(in_features, hidden_features)
        self.mlp_mid = nn.Sequential(
            nn.Linear(hidden_features + global_features_dim, hidden_features),
            nn.ReLU(),
        )
        self.gcn2 = SimpleGCNLayer(hidden_features, hidden_features)
        enc1_layer = TransformerEncoderLayer(
            hidden_features,
            nhead=4,
            dim_feedforward=hidden_features * 2,
            batch_first=True,
        )
        self.enc1 = TransformerEncoder(enc1_layer, num_layers=num_layers // 2)
        self.weight_gcn2 = nn.Parameter(torch.ones(1))
        self.weight_enc1 = nn.Parameter(torch.ones(1))
        self.mlp_fusion = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        )
        enc2_layer = TransformerEncoderLayer(
            hidden_features,
            nhead=4,
            dim_feedforward=hidden_features * 2,
            batch_first=True,
        )
        self.enc2 = TransformerEncoder(enc2_layer, num_layers=num_layers // 2)
        # ==================== 【修改前：原本的单一全连接层】 ====================
        # self.fc = nn.Linear(hidden_features * num_nodes, out_features)

        # ==================== 【修改后：阶段 2 解耦多头设计】 ====================
        # 阶段 2：高低频物理空间解耦双预测头 (5.0 GHz 分界点 -> 对应索引 800)
        self.fc_low = nn.Linear(hidden_features * num_nodes, 20 * 933)
        self.fc_high = nn.Linear(hidden_features * num_nodes, 20 * 1767)
    # 修改前
    # def forward(
    #    self,
    #    node_features: torch.Tensor,
    #    global_features: torch.Tensor,
    #    adjacency_matrix: torch.Tensor,
    # ) -> torch.Tensor:

    #  修改后的声明：
    def forward(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # <-- 改成支持返回包含3个Tensor的元组
        batch_size = node_features.size(0)

        # 1) 图卷积提取局部拓扑信息。
        local_graph_features = self.gcn1(node_features, adjacency_matrix)
        # 2) 将节点局部特征与全局工艺特征拼接。
        fused_input_features = torch.cat([local_graph_features, global_features], dim=-1)
        hidden_features_mid = self.mlp_mid(fused_input_features)

        # 3) GCN 与 Transformer 双分支编码并融合。
        graph_branch_features = self.gcn2(hidden_features_mid, adjacency_matrix)
        transformer_branch_features = self.enc1(hidden_features_mid)
        fused_branch_features = (
            self.weight_gcn2 * graph_branch_features
            + self.weight_enc1 * transformer_branch_features
        )

        # 4) 深层编码后拉平，回归全部频点上的输出通道。
        refined_features = self.mlp_fusion(fused_branch_features)
        encoded_features = self.enc2(refined_features)
        # ==================== 【修改前：原本的 forward 结尾】 ====================
        # flattened_features = encoded_features.contiguous().view(batch_size, -1)
        # regression_output = self.fc(flattened_features)
        # return regression_output.view(batch_size, 20, -1)

        # ==================== 【第一次修改后：阶段 2 多头输出与频段拼接】 ====================
        # flattened_features = encoded_features.contiguous().view(batch_size, -1)
        
        # # 阶段 2：分别通过低频头和高频头
        # out_low = self.fc_low(flattened_features).view(batch_size, 20, 833)
        # out_high = self.fc_high(flattened_features).view(batch_size, 20, 1667)
        
        # # 在最后一维（频率轴）拼接，完美兼容原项目后续的指标计算和 npz 保存
        # full_prediction = torch.cat([out_low, out_high], dim=-1)
        
        # # 同时返回完整拼接结果和两个分头，方便损失函数做边界约束
        # return full_prediction, out_low, out_high

        # ==================== 【第二次修改后：方案B 前向重叠拼接】 ====================
        flattened_features = encoded_features.contiguous().view(batch_size, -1)
        
        # 分别输出包含重叠区的两段预测
        out_low = self.fc_low(flattened_features).view(batch_size, 20, 933)
        out_high = self.fc_high(flattened_features).view(batch_size, 20, 1767)
        
        # 拼装完整的 2500 个频点
        seg1 = out_low[..., :733]                             # 0 ~ 800 点（纯低频）
        seg2 = (out_low[..., 733:833] + out_high[..., :100]) / 2.0  # 800 ~ 900 点（重叠区取双头平均值）
        seg3 = out_high[..., 100:]                            # 900 ~ 2500 点（纯高频）
        
        full_prediction = torch.cat([seg1, seg2, seg3], dim=-1)
        
        # 为了让损失函数能拿到未拼接的重叠部分计算约束，这里将 out_low 和 out_high 一并返回
        return full_prediction, out_low, out_high

# 修改前
# def physics_informed_loss(out_low, out_high, target, alpha=1.0, beta=1.0, gamma=10.0):
#     """
#     解耦多头物理边界连续性损失函数
#     gamma: 5.0 GHz 拼接点断层惩罚项权重 (值越大衔接越平滑)
#     """
#     # 1. 自动对真实标签（target）进行高低频切片
#     target_low = target[..., :833]    # [Batch, 20, 800]
#     target_high = target[..., 833:]   # [Batch, 20, 1600]
    
#     # 2. 计算高低频段各自的基础回归损失 (这里严格使用原作者指定的 Huber Loss 逻辑)
#     loss_low = huber_loss(target_low, out_low)
#     loss_high = huber_loss(target_high, out_high)
    
#     # 3. 核心物理约束：低频预测头的最后一个频点(第800点)，必须与高频预测头的第一点连续
#     boundary_low_edge = out_low[..., -1]
#     boundary_high_edge = out_high[..., 0]
#     loss_boundary = F.l1_loss(boundary_low_edge, boundary_high_edge)
    
#     # 加权融合
#     total_loss = alpha * loss_low + beta * loss_high + gamma * loss_boundary
#     return total_loss

# 第一次修改后
# def physics_informed_loss(full_prediction, out_low, out_high, target, gamma=0.1):
#     """
#     full_prediction: 已经在 forward 中通过重叠区均值融合拼好的 2500 点完整预测
#     out_low: 低频头单独输出的 933 点预测
#     out_high: 高频头单独输出的 1767 点预测
#     target: 真实的 2500 点 S 参数标签
#     gamma: 方案 A 的弱约束权重
#     """
#     huber_loss = nn.HuberLoss()
    
#     # 全频段大局观拟合损失 (0 ~ 15 GHz 一体化计算)
#     # 不管高低频怎么分工，最终拼出来的 full_prediction 必须和真实的 2500 点完全对齐
#     loss_main = huber_loss(target, full_prediction) 
    
#     # 100 点重叠交界区的协同连续性损失
#     # 强迫低频头的尾巴 [733:833] 和高频头的脑袋 [:100] 在空间上互相认同、平滑交接
#     overlap_low_edge = out_low[..., 733:833]   
#     overlap_high_edge = out_high[..., :100]    
#     loss_overlap_match = F.l1_loss(overlap_low_edge, overlap_high_edge)
    
#     # 融合最终损失
#     total_loss = loss_main + gamma * loss_overlap_match
#     return total_loss

# 第二次修改后
# def physics_informed_loss(full_prediction, out_low, out_high, target, gamma=0.05):
#     """
#     1. 换回原作者特有的狂暴自定义 huber_loss，强迫网络高频死磕真值，找回图二的精准度。
#     2. 将 gamma 微调至 0.05，既能保证 5GHz 完美闭合，又绝不抢高频拟合的风头。
#     """
#     # 换回原作者自带的 huber_loss 算大局观（真值在前，预测在后）
#     loss_main = huber_loss(target, full_prediction) 
    
#     # 重叠交界平滑约束
#     overlap_low_edge = out_low[..., 733:833]   
#     overlap_high_edge = out_high[..., :100]    
#     loss_overlap_match = F.l1_loss(overlap_low_edge, overlap_high_edge)
    
#     # 给高频段增加一点点小小的单独看管，防止它高频相位漂移
#     loss_high_fine_tune = huber_loss(target[..., 833:], out_high[..., 100:])
    
#     # 最终黄金比例加权
#     total_loss = loss_main + 0.2 * loss_high_fine_tune + gamma * loss_overlap_match
#     return total_loss

# 第三次修改后
# def physics_informed_loss(full_prediction, out_low, out_high, target, gamma=None):
#     # 保证融合后的 2500 点整体拟合（这里交界处因为前向平均，已经天然平滑）
#     loss_main = huber_loss(target, full_prediction) 
    
#     # 提取出重叠区 [733:833] 的【真实标签】
#     overlap_target = target[..., 733:833]
    
#     # 让低频尾巴和高频脑袋各自去死磕【真实标签】，而不是互相捆绑
#     loss_overlap_low = huber_loss(overlap_target, out_low[..., 733:833])
#     loss_overlap_high = huber_loss(overlap_target, out_high[..., :100])
    
#     # 加重对纯高频段 [833:] 的惩罚，把高频相位强行“钉”在真值上
#     loss_high_phase = huber_loss(target[..., 833:], out_high[..., 100:])
    
#     # 最终加权：全面摒弃互相牵制的 gamma 变量
#     # 给高频相位 0.5 的高权重，强迫它回归正确的相位坐标
#     total_loss = loss_main + 0.1 * (loss_overlap_low + loss_overlap_high) + 0.5 * loss_high_phase
    
#     return total_loss

# 第四次修改后
# def physics_informed_loss(full_prediction, out_low, out_high, target, gamma=0.05, lambda_diff=0.5):
#     # 基础物理空间拟合 (大局观：保证振幅对齐)
#     # 使用你原作者自带的 huber_loss (结合了 MAE 和 MSE，非常优秀)
#     loss_main = huber_loss(target, full_prediction) 
    
#     # ==================== 一阶导数趋势匹配 ====================
#     # 沿着频率轴 (dim=-1) 计算相邻频点的差值 (即离散导数/斜率)
#     dy_pred = full_prediction[..., 1:] - full_prediction[..., :-1]
#     dy_true = target[..., 1:] - target[..., :-1]
    
#     # 趋势匹配损失 (消除锯齿毛刺，强力锚定波峰波谷相位)
#     # 我们用 L1 Loss (绝对值误差) 来严厉惩罚斜率的不一致
#     loss_trend = torch.mean(torch.abs(dy_pred - dy_true))
#     # =====================================================================
    
#     # 缝合区平滑保障 (方案 B)
#     # 保持一个极弱的 gamma=0.05 约束，仅仅是为了向网络暗示 "你们俩在 5GHz 处有关联"
#     # 不会抢夺主干网络追踪高频相位的注意力
#     overlap_low_edge = out_low[..., 733:833]   
#     overlap_high_edge = out_high[..., :100]    
#     loss_overlap_match = F.l1_loss(overlap_low_edge, overlap_high_edge)
    
#     # 融合比例
#     total_loss = loss_main + lambda_diff * loss_trend + gamma * loss_overlap_match
    
#     return total_loss

# 第五次修改后
# def physics_informed_loss(full_prediction, out_low, out_high, target, gamma=0.05, lambda_diff=0.5):
#     # 基础物理空间拟合 (大局观，关注垂直数值误差)
#     loss_main = huber_loss(target, full_prediction) 
    
#     # 一阶导数趋势匹配 (关注局部平滑度，消除毛刺，保持波形连贯)
#     dy_pred = full_prediction[..., 1:] - full_prediction[..., :-1]
#     dy_true = target[..., 1:] - target[..., :-1]
#     loss_trend = torch.mean(torch.abs(dy_pred - dy_true))
    
#     # 缝合区弱约束 (方案 B 遗留的平滑过渡保障)
#     overlap_low_edge = out_low[..., 733:833]   
#     overlap_high_edge = out_high[..., :100]    
#     loss_overlap_match = F.l1_loss(overlap_low_edge, overlap_high_edge)
    
#     # ==================== 高频段余弦形貌锁相 ====================
#     # 提取高频段 (从 833 点开始，即约 5.0 GHz 以后的剧烈振荡区)
#     target_hf = target[..., 833:]
#     pred_hf = full_prediction[..., 833:]
    
#     # F.cosine_similarity 沿着频率维度(dim=-1)计算
#     # 完全同相 = 1，完全反相 = -1。我们希望它趋近于 1，所以损失设为 (1 - cos_sim)
#     cos_sim = F.cosine_similarity(pred_hf, target_hf, dim=-1)
#     loss_phase_lock = torch.mean(1.0 - cos_sim)
#     # ===========================================================================
    
#     # 最终融合：给锁相机制分配 0.15 的权重，强迫模型对齐波峰波谷
#     total_loss = loss_main + lambda_diff * loss_trend + gamma * loss_overlap_match + 0.15 * loss_phase_lock
    
#     return total_loss

# 第六次修改后
# def physics_informed_loss(full_prediction, out_low, out_high, target, gamma=0.05, lambda_diff=1.0, lambda_phase=1.5):
#     """
#     参数说明:
#     - gamma (0.05): 控制 5GHz 缝合处的平滑过渡。
#     - lambda_diff (1.0): 强力压制毛刺，保证曲线一阶连续。
#     - lambda_phase (1.5): 【策略1】暴力拉升相位锁定的权重，逼迫模型直面错位！
#     """
#     # 基础物理空间拟合 (大局观，关注垂直数值误差)
#     loss_main = huber_loss(target, full_prediction) 
    
#     # 一阶导数趋势匹配 (强力消除毛刺，保持波形极其连贯)
#     dy_pred = full_prediction[..., 1:] - full_prediction[..., :-1]
#     dy_true = target[..., 1:] - target[..., :-1]
#     loss_trend = torch.mean(torch.abs(dy_pred - dy_true))
    
#     # 缝合区弱约束 (维持双头架构在 5GHz 处的无缝交接)
#     overlap_low_edge = out_low[..., 733:833]   
#     overlap_high_edge = out_high[..., :100]    
#     loss_overlap_match = F.l1_loss(overlap_low_edge, overlap_high_edge)
    
#     # ==================== 高频局部窗口分块锁相 ====================
#     # 提取高频段 (从 833 点开始，共 1667 个频点)
#     target_hf = target[..., 833:]
#     pred_hf = full_prediction[..., 833:]
    
#     # 将 1667 个点切分为 4 个局部窗口 (每个窗口约 416 个点)
#     # 这样可以迫使网络在每一个局部区间内都必须对齐波峰，彻底杜绝全局平均导致的“梯度稀释”
#     num_chunks = 4
#     chunks_pred = torch.chunk(pred_hf, chunks=num_chunks, dim=-1)
#     chunks_true = torch.chunk(target_hf, chunks=num_chunks, dim=-1)
    
#     loss_phase_lock = 0.0
#     for cp, ct in zip(chunks_pred, chunks_true):
#         # 针对每一个局部小窗口独立计算余弦相似度
#         cos_sim_local = F.cosine_similarity(cp, ct, dim=-1)
#         # 累加每个窗口的惩罚项
#         loss_phase_lock += torch.mean(1.0 - cos_sim_local)
        
#     # 取各窗口的平均惩罚值
#     loss_phase_lock = loss_phase_lock / num_chunks
#     # ===================================================================================
    
#     # 终极融合：将 lambda_phase (默认 1.5) 乘上去，给出极其严厉的相位惩罚！
#     total_loss = loss_main + (lambda_diff * loss_trend) + (gamma * loss_overlap_match) + (lambda_phase * loss_phase_lock)
    
#     return total_loss

# 第七次修改后
import torch
import torch.nn.functional as F

def physics_informed_loss(full_prediction, out_low, out_high, target, gamma=0.15, lambda_diff=1.0, lambda_phase=1.5):
    """  
    参数:
    - full_prediction: 模型拼接后的完整输出 [batch_size, 20, 1667]
    - out_low: 低频头输出 [batch_size, 20, 833]
    - out_high: 高频头输出 [batch_size, 20, 934]
    - target: 仿真真值标签 [batch_size, 20, 1667]
    - gamma: 一阶差分损失的权重系数 (从0.05上调至0.15，加大压制纹波和超调的力度)
    - lambda_diff: 一阶差分项的总缩放因子
    - lambda_phase: 高频局部锁相与幅值约束的总缩放因子
    """
    device = full_prediction.device
    batch_size = full_prediction.size(0)
    
    # ========================================================================
    # 全频段基础数值对齐 (Huber Loss)
    # ========================================================================
    base_huber = F.huber_loss(full_prediction, target, delta=1.0)
    
    # ========================================================================
    # 双头重叠区域边界衔接损失 (防止低频和高频接头处断层)
    # ========================================================================
    overlap_low_edge = out_low[..., -100:]      # 低频头的后 100 个点
    overlap_high_edge = out_high[..., :100]     # 高频头的前 100 个点
    loss_overlap_match = F.l1_loss(overlap_low_edge, overlap_high_edge)
    
    # ========================================================================
    # 全频段一阶差分约束 (Trend Loss) -> 压制高频毛刺与不物理震荡
    # ========================================================================
    # 计算预测值和真值在频率轴上的相邻点差分（斜率）
    diff_pred = full_prediction[..., 1:] - full_prediction[..., :-1]
    diff_target = target[..., 1:] - target[..., :-1]
    # 惩罚预测斜率与真实斜率的不一致
    loss_trend = F.huber_loss(diff_pred, diff_target, delta=1.0)
    
    # ========================================================================
    # 高频段分窗口局部锁相 + 幅值标准差约束 -> 解决相位右移与波峰过高
    # ========================================================================
    # 提取高频段数据 (从第 833 个频点开始，共 1667 个点)
    target_hf = target[..., 833:]
    pred_hf = full_prediction[..., 833:]
    
    # 将高频段切分为 4 个局部小窗口 (每个窗口约 416 个频点)
    # 这样可以迫使网络在每一个局部区域内部都必须对齐波峰，杜绝全局平均导致的“梯度稀释”
    num_chunks = 4
    chunks_pred = torch.chunk(pred_hf, chunks=num_chunks, dim=-1)
    chunks_true = torch.chunk(target_hf, chunks=num_chunks, dim=-1)
    
    loss_phase_lock = 0.0
    
    for cp, ct in zip(chunks_pred, chunks_true):
        # 相位约束：针对每一个局部小窗口独立计算余弦相似度（管横轴波峰波谷位置对齐）
        # cos_sim_local 范围在 [-1, 1]，1.0 代表形状和相位完全对齐
        cos_sim_local = F.cosine_similarity(cp, ct, dim=-1)
        loss_phase_local = torch.mean(1.0 - cos_sim_local)
        
        # 幅值约束（本次修正核心）：计算局部窗口内的标准差 (代表曲线的波动幅度)
        # 用来疯狂惩罚模型为了对齐相位而把波峰波谷强行“撑大/超调”的投机行为
        std_pred = torch.std(cp, dim=-1)
        std_true = torch.std(ct, dim=-1)
        # 使用 L1 损失确保预测振幅的标准差与真实振幅高度一致（管纵轴波峰高度对齐）
        loss_amp_local = F.l1_loss(std_pred, std_true)
        
        # 局部融合：相位对齐 + 1.0 * 幅值约束 (给足拉力，把突出去的尖峰拍回来)
        loss_phase_lock += loss_phase_local + 1.0 * loss_amp_local
        
    # 取各窗口的平均惩罚值
    loss_phase_lock = loss_phase_lock / num_chunks

    # ========================================================================
    # 终极权衡：将四大金刚损失按各自的物理权重进行加权求和
    # ========================================================================
    total_loss = (
        1.0 * base_huber +                 # 数值大盘不能丢
        0.5 * loss_overlap_match +         # 接头处保持平滑
        gamma * lambda_diff * loss_trend +  # 压制毛刺纹波 (权重已调整)
        lambda_phase * loss_phase_lock     # 死磕高频相位 + 压制幅值超调
    )
    
    return total_loss

def distance_from_start(xs: NDArray[np.float32], ys: NDArray[np.float32]) -> NDArray[np.float32]:
    """计算每个点到起点的欧氏距离。"""
    if len(xs) == 0:
        return np.array([], dtype=np.float32)
    return np.sqrt((xs - xs[0]) ** 2 + (ys - ys[0]) ** 2).astype(np.float32)


def nearest_line_distance(
    src_x: NDArray[np.float32],
    src_y: NDArray[np.float32],
    dst_x: NDArray[np.float32],
    dst_y: NDArray[np.float32],
) -> NDArray[np.float32]:
    """src 每个节点到 dst 走线最近节点的欧氏距离。"""
    src_pts = np.stack([src_x, src_y], axis=1)
    dst_pts = np.stack([dst_x, dst_y], axis=1)
    diff = src_pts[:, None, :] - dst_pts[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2)).astype(np.float32)
    return np.min(dist, axis=1)


def build_dummy_nodes(
    x1: NDArray[np.float32],
    y1: NDArray[np.float32],
    x2: NDArray[np.float32],
    y2: NDArray[np.float32],
) -> Tuple[np.float32, np.float32, np.float32, np.float32]:
    """构建双线的两个 dummy 节点: 起点前方与终点后方。"""
    start_mid_x = np.float32((x1[0] + x2[0]) / 2.0)
    start_mid_y = np.float32((y1[0] + y2[0]) / 2.0)
    end_mid_x = np.float32((x1[-1] + x2[-1]) / 2.0)
    end_mid_y = np.float32((y1[-1] + y2[-1]) / 2.0)

    # 按双线起始/终止方向估计 dummy 在 x 方向的前后偏移。
    dir_start_x = np.float32(((x1[1] - x1[0]) + (x2[1] - x2[0])) / 2.0)
    dir_end_x = np.float32(((x1[-1] - x1[-2]) + (x2[-1] - x2[-2])) / 2.0)

    # 平行双线场景下，两条线段长度一致，偏移量取 trace1 的局部段长即可。
    start_len = np.float32(math.hypot(float(x1[1] - x1[0]), float(y1[1] - y1[0])))
    end_len = np.float32(math.hypot(float(x1[-1] - x1[-2]), float(y1[-1] - y1[-2])))
    start_offset = np.float32(max(10, 2 * float(start_len)))
    end_offset = np.float32(max(10, 2 * float(end_len)))

    start_sign = np.float32(1.0 if dir_start_x >= 0 else -1.0)
    end_sign = np.float32(1.0 if dir_end_x >= 0 else -1.0)

    dummy_start_x = np.float32(start_mid_x - start_sign * start_offset)
    dummy_start_y = np.float32(start_mid_y)
    dummy_end_x = np.float32(end_mid_x + end_sign * end_offset)
    dummy_end_y = np.float32(end_mid_y)
    return dummy_start_x, dummy_start_y, dummy_end_x, dummy_end_y


def build_graph_for_double_trace(
    trace1: pd.DataFrame,
    trace2: pd.DataFrame,
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """将双线样本映射为图数据。

    节点顺序约定:
    0 = dummy_start,
    1..n1 = trace1,
    n1+1..n1+n2 = trace2,
    last = dummy_end。
    """
    x1 = trace1["X"].to_numpy(dtype=np.float32)
    y1 = trace1["Y"].to_numpy(dtype=np.float32)
    x2 = trace2["X"].to_numpy(dtype=np.float32)
    y2 = trace2["Y"].to_numpy(dtype=np.float32)

    if len(x1) < 2 or len(x2) < 2:
        raise ValueError("Each trace must contain at least 2 nodes.")

    n1 = len(x1)
    n2 = len(x2)
    num_nodes = n1 + n2 + 2

    dummy_start_x, dummy_start_y, dummy_end_x, dummy_end_y = build_dummy_nodes(x1, y1, x2, y2)

    dist1 = distance_from_start(x1, y1)
    dist2 = distance_from_start(x2, y2)

    # 平行线场景下线间距为常值，统一使用起点间距。
    line_spacing = np.float32(math.hypot(float(x1[0] - x2[0]), float(y1[0] - y2[0])))
    to_other_1 = np.full((n1,), line_spacing, dtype=np.float32)
    to_other_2 = np.full((n2,), line_spacing, dtype=np.float32)

    # 平行双线总长度一致，选择 trace1 作为全局长度代表。
    total_length = np.float32(float(dist1[-1]))

    # 节点特征: [x, y, 沿线累计长度, 到对面线最近距离]
    node_features = np.zeros((num_nodes, 4), dtype=np.float32)

    # 0: dummy_start
    node_features[0] = np.array([dummy_start_x, dummy_start_y, 0.0, line_spacing], dtype=np.float32)

    # 1..n1: trace1
    node_features[1 : 1 + n1, 0] = x1
    node_features[1 : 1 + n1, 1] = y1
    node_features[1 : 1 + n1, 2] = dist1
    node_features[1 : 1 + n1, 3] = to_other_1

    # n1+1..n1+n2: trace2
    start_t2 = 1 + n1
    node_features[start_t2 : start_t2 + n2, 0] = x2
    node_features[start_t2 : start_t2 + n2, 1] = y2
    node_features[start_t2 : start_t2 + n2, 2] = dist2
    node_features[start_t2 : start_t2 + n2, 3] = to_other_2

    # last: dummy_end
    node_features[-1] = np.array([dummy_end_x, dummy_end_y, total_length, line_spacing], dtype=np.float32)

    # 全局特征：[线长, 线宽, 对地距离]
    global_features = np.zeros((num_nodes, 3), dtype=np.float32)
    global_features[:, 0] = total_length
    global_features[:, 1] = np.float32(TRACE_WIDTH)
    global_features[:, 2] = np.float32(DIELECTRIC_THICKNESS) # 双层板子的对地距离即为介质厚度。

    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    def add_edge(i: int, j: int, w: float) -> None:
        weight = np.float32(max(w, 1e-6))
        adjacency_matrix[i, j] = weight
        adjacency_matrix[j, i] = weight

    # 边类型1: trace1 内部相邻节点连边。
    for k in range(n1 - 1):
        idx_a = 1 + k
        idx_b = 1 + k + 1
        w = float(math.hypot(float(x1[k + 1] - x1[k]), float(y1[k + 1] - y1[k])))
        add_edge(idx_a, idx_b, w)

    # 边类型2: trace2 内部相邻节点连边。
    for k in range(n2 - 1):
        idx_a = start_t2 + k
        idx_b = start_t2 + k + 1
        w = float(math.hypot(float(x2[k + 1] - x2[k]), float(y2[k + 1] - y2[k])))
        add_edge(idx_a, idx_b, w)

    # 边类型3: 起点 dummy 与两条线起点连接。
    add_edge(0, 1, float(math.hypot(float(dummy_start_x - x1[0]), float(dummy_start_y - y1[0]))))
    add_edge(0, start_t2, float(math.hypot(float(dummy_start_x - x2[0]), float(dummy_start_y - y2[0]))))

    # 边类型4: 终点 dummy 与两条线终点连接。
    end_t1 = 1 + n1 - 1
    end_t2 = start_t2 + n2 - 1
    add_edge(num_nodes - 1, end_t1, float(math.hypot(float(dummy_end_x - x1[-1]), float(dummy_end_y - y1[-1]))))
    add_edge(num_nodes - 1, end_t2, float(math.hypot(float(dummy_end_x - x2[-1]), float(dummy_end_y - y2[-1]))))

    return adjacency_matrix, node_features, global_features


def extract_labels_from_s4p(s_parameters: NDArray[np.complex64]) -> NDArray[np.float32]:
    """从 4 端口 S 矩阵抽取 10 个独立参数的实部/虚部。

    返回: [20, freq_points]
    """
    channels: List[NDArray[np.float32]] = []
    for i, j in S_PARAM_INDICES:
        channels.append(np.real(s_parameters[:, i, j]).astype(np.float32))
        channels.append(np.imag(s_parameters[:, i, j]).astype(np.float32))
    return np.stack(channels, axis=0).astype(np.float32)


def load_double_trace_data(data_dir: str, max_samples: int | None = None):
    """加载几何与 S4P，构建训练/测试样本列表。"""
    traces_file = os.path.join(os.path.dirname(data_dir), "pcb_traces.txt")
    traces_df = pd.read_csv(traces_file, sep=",", header=0)

    grouped_by_pcb = traces_df.groupby("PCB_Index")
    pcb_ids = sorted([int(str(pcb_id)) for pcb_id in grouped_by_pcb.groups.keys()])
    if max_samples is not None:
        pcb_ids = pcb_ids[:max_samples]

    adjacency_matrices = []
    node_features_list = []
    global_features_list = []
    labels_list = []
    pcb_id_list = []
    freq_list = []

    for pcb_id in pcb_ids:
        pcb_group = grouped_by_pcb.get_group(pcb_id)
        t1 = pcb_group[pcb_group["Trace_Index"] == 0].sort_values("Node_Index")
        t2 = pcb_group[pcb_group["Trace_Index"] == 1].sort_values("Node_Index")
        if t1.empty or t2.empty:
            continue

        s4p_path = os.path.join(data_dir, f"{int(pcb_id)}.s4p")
        if not os.path.exists(s4p_path):
            continue

        adjacency_matrix, node_features, global_features = build_graph_for_double_trace(t1, t2)

        network = rf.Network(s4p_path)
        s_parameters = np.asarray(network.s).astype(np.complex64)
        labels = extract_labels_from_s4p(s_parameters)
        # 保留原始非线性频点，不做重采样。
        freq_ghz = (network.f.astype(np.float64) / 1e9).astype(np.float32)

        adjacency_matrices.append(adjacency_matrix)
        node_features_list.append(node_features)
        global_features_list.append(global_features)
        labels_list.append(labels)
        pcb_id_list.append(int(pcb_id))
        freq_list.append(freq_ghz)

    return adjacency_matrices, node_features_list, global_features_list, labels_list, pcb_id_list, freq_list


class DoubleTraceDataset(Dataset):
    def __init__(self, adj_matrices, node_features, global_features, labels, pcb_ids, freqs):
        self.adj_matrices = adj_matrices
        self.node_features = node_features
        self.global_features = global_features
        self.labels = labels
        self.pcb_ids = pcb_ids
        self.freqs = freqs

    def __len__(self):
        return len(self.adj_matrices)

    def __getitem__(self, idx):
        # 每个样本返回: 邻接矩阵、节点特征、全局特征、标签、PCB编号、频率轴。
        return (
            torch.tensor(self.adj_matrices[idx], dtype=torch.float32),
            torch.tensor(self.node_features[idx], dtype=torch.float32),
            torch.tensor(self.global_features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            self.pcb_ids[idx],
            torch.tensor(self.freqs[idx], dtype=torch.float32),
        )


def huber_loss(y_true: torch.Tensor, y_pred: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    # 线性+平方误差组合，兼顾抗异常值与收敛速度。
    mae = torch.mean(torch.abs(y_true - y_pred))
    mse = torch.mean((y_true - y_pred) ** 2)
    return (1 - alpha) * mae + alpha * mse


def build_channel_names() -> List[str]:
    # 生成 20 个输出通道名称，便于指标保存。
    names: List[str] = []
    for s_name in S_PARAM_NAMES:
        names.append(f"{s_name}_real")
        names.append(f"{s_name}_imag")
    return names


def calculate_metrics(y_true, y_pred):
    """计算整体指标和每个输出通道指标。"""
    y_true_array = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred_array = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred

    metrics = {
        "mse": mean_squared_error(y_true_array.flatten(), y_pred_array.flatten()),
        "rmse": np.sqrt(mean_squared_error(y_true_array.flatten(), y_pred_array.flatten())),
        "mae": mean_absolute_error(y_true_array.flatten(), y_pred_array.flatten()),
        "r2": r2_score(y_true_array.flatten(), y_pred_array.flatten()),
    }

    channel_names = build_channel_names()
    for i, channel_name in enumerate(channel_names):
        true_channel = y_true_array[:, i, :].flatten()
        pred_channel = y_pred_array[:, i, :].flatten()
        metrics[f"{channel_name}_mse"] = mean_squared_error(true_channel, pred_channel)
        metrics[f"{channel_name}_mae"] = mean_absolute_error(true_channel, pred_channel)
    return metrics


def calculate_sample_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """按样本计算总体误差指标。"""
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    sample_metrics = []
    for i in range(y_true_np.shape[0]):
        true_sample = y_true_np[i]
        pred_sample = y_pred_np[i]
        sample_metrics.append(
            {
                "mse": mean_squared_error(true_sample.flatten(), pred_sample.flatten()),
                "mae": mean_absolute_error(true_sample.flatten(), pred_sample.flatten()),
                "r2": r2_score(true_sample.flatten(), pred_sample.flatten()),
            }
        )
    return sample_metrics


def plot_training_curves(train_losses, valid_losses, save_dir):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(valid_losses, label="Valid")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()


def main():
    # 主流程: 数据加载 -> 训练验证 -> 测试保存 -> 指标统计。
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    early_stop_patience = 50

    (
        adj_matrices,
        node_features,
        global_features,
        labels,
        pcb_ids,
        freqs,
    ) = load_double_trace_data(DATA_DIR)

    if len(labels) == 0:
        raise RuntimeError("No valid samples found. Please check double-trace data directory and txt file.")

    dataset = DoubleTraceDataset(adj_matrices, node_features, global_features, labels, pcb_ids, freqs)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    test_loader = DataLoader(test_dataset, batch_size=4)

    model = DoubleTraceTransformer(
        in_features=4,
        global_features_dim=3,
        hidden_features=128,
        out_features=20 * labels[0].shape[1],
        num_nodes=adj_matrices[0].shape[0],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_losses = []
    val_losses = []
    best_loss = float("inf")
    epochs_no_improve = 0

    # 修改前
    # print("Starting training for double-trace model...")
    # for epoch in range(60):
    #     model.train()
    #     train_loss_sum = 0.0
    #     for adjacency, node_feat, global_feat, target, _, _ in train_loader:
    #         adjacency = adjacency.to(device)
    #         node_feat = node_feat.to(device)
    #         global_feat = global_feat.to(device)
    #         target = target.to(device)

    #         # 修改前
    #         # optimizer.zero_grad()
    #         # prediction = model(node_feat, global_feat, adjacency)
    #         # loss = huber_loss(target, prediction)
    #         # loss.backward()

    #         # # 第一次修改后
    #         # optimizer.zero_grad()
    #         # # 接收模型吐出的完整预测以及高低频两个分头 (严格遵循原代码的输入特征顺序)
    #         # full_prediction, out_low, out_high = model(node_feat, global_feat, adjacency)
            
    #         # # 引入带 5.0 GHz 连续性惩罚的新损失函数
    #         # loss = physics_informed_loss(full_prediction, out_low, out_high, target, gamma=0.1)
    #         # loss.backward()

    #         # 第二次修改后
    #         # 梯度清零
    #         optimizer.zero_grad()
            
    #         # 前向传播（严格遵循原代码的输入特征顺序）
    #         full_prediction, out_low, out_high = model(node_feat, global_feat, adjacency)
            
    #         # 传入重叠连续性损失函数
    #         loss = physics_informed_loss(full_prediction, out_low, out_high, target, gamma=0.05)
            
    #         # 完整的炼丹更新五部曲（找回丢失的参数更新与裁剪）
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #         optimizer.step()
            
    #         # 必须进行 Loss 值的状态累加，否则 Train 永远是 0
    #         train_loss_sum += float(loss.item())

    #     avg_train_loss = train_loss_sum / len(train_loader)
    #     train_losses.append(avg_train_loss)

    #     model.eval()
    #     val_loss_sum = 0.0
    #     with torch.no_grad():
    #         for adjacency, node_feat, global_feat, target, _, _ in val_loader:
    #             adjacency = adjacency.to(device)
    #             node_feat = node_feat.to(device)
    #             global_feat = global_feat.to(device)
    #             target = target.to(device)
    #             # 修改前
    #             # val_loss_sum += float(huber_loss(target, model(node_feat, global_feat, adjacency)).item())

    #             # 修改后
    #             pred_full, _, _ = model(node_feat, global_feat, adjacency)
    #             val_loss_sum += float(huber_loss(target, pred_full).item())

    #     avg_val_loss = val_loss_sum / len(val_loader)
    #     val_losses.append(avg_val_loss)

    #     if avg_val_loss < best_loss:
    #         best_loss = avg_val_loss
    #         epochs_no_improve = 0
    #         torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
    #     else:
    #         epochs_no_improve += 1

    #     if epoch % 1 == 0:
    #         print(f"Epoch {epoch}: Train {avg_train_loss:.6f}, Val {avg_val_loss:.6f}")
    #     if epochs_no_improve >= early_stop_patience:
    #         print(f"Early stopping at epoch {epoch}")
    #         break

    # plot_training_curves(train_losses, val_losses, SAVE_DIR)

    # print("\nStarting testing...")
    
    # 第一次修改后
    # ==================== 🔄【断点恢复配置区】====================
    RESUME_TRAINING = True  # 想接着跑就保持 True，想从头跑改成 False
    START_EPOCH = 0
    train_losses, val_losses = [], []
    checkpoint_path = os.path.join(SAVE_DIR, "latest_checkpoint.pth")
    best_loss = float("inf")  # 原代码中用于保存 best_model 的阈值

    if RESUME_TRAINING and os.path.exists(checkpoint_path):
        print(f"\n 发现未完成的训练断点，正在加载进度...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        START_EPOCH = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"读档成功！将从 Epoch {START_EPOCH} 继续向下训练。")
    else:
        print("\n 未发现断点，将从零开始全新训练。")
    # ============================================================

    print("Starting training for double-trace model...")
    
    # 初始化变量，防止一进去还没跑完第一个 batch 就按 Ctrl+C 导致没定义的边缘情况
    avg_train_loss = 0.0
    avg_val_loss = 0.0

    try:
        # 注意：起点改为了解耦的 START_EPOCH
        for epoch in range(START_EPOCH, 60):
            model.train()
            train_loss_sum = 0.0
            for adjacency, node_feat, global_feat, target, _, _ in train_loader:
                adjacency = adjacency.to(device)
                node_feat = node_feat.to(device)
                global_feat = global_feat.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                full_prediction, out_low, out_high = model(node_feat, global_feat, adjacency)
                
                # 传入带高频局部窗口锁相+一阶差分的物理损失函数
                loss = physics_informed_loss(full_prediction, out_low, out_high, target, gamma=0.05)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss_sum += float(loss.item())

            # 关键：先在 try 内部算完这一轮的平均 train loss
            avg_train_loss = train_loss_sum / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for adjacency, node_feat, global_feat, target, _, _ in val_loader:
                    adjacency = adjacency.to(device)
                    node_feat = node_feat.to(device)
                    global_feat = global_feat.to(device)
                    target = target.to(device)
                    
                    pred_full, _, _ = model(node_feat, global_feat, adjacency)
                    val_loss_sum += float(huber_loss(target, pred_full).item())

            # 关键：在 try 内部算完这一轮的平均 val loss
            avg_val_loss = val_loss_sum / len(val_loader)
            val_losses.append(avg_val_loss)

            # 验证集表现好，则保存最优模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))

            if epoch % 1 == 0:
                print(f"Epoch {epoch}: Train {avg_train_loss:.6f}, Val {avg_val_loss:.6f}")

    # ==================== 【捕获 Ctrl+C 中断信号并安全存档】====================
    except KeyboardInterrupt:
        print(f"\n 检测到主动终止！正在紧急为您封存当前 Epoch {epoch} 的训练断点...")
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_loss': best_loss
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f" 断点已安全写入: {checkpoint_path}")
        print("提示：再次运行脚本将无缝接着当前的 Epoch 继续训练。程序优雅退出。")
        return  # 退出 main 函数，不再向下执行测试集出图
    #============================================================================

    print("\nStarting testing...")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=device))
    model.eval()

    test_results_dir = os.path.join(SAVE_DIR, "test_results")
    metrics_dir = os.path.join(SAVE_DIR, "metrics")
    os.makedirs(test_results_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    all_preds = []
    all_labels = []
    all_pcb_ids = []

    with torch.no_grad():
        for adjacency, node_feat, global_feat, target, pcb_batch_ids, freq_batch in test_loader:
            adjacency = adjacency.to(device)
            node_feat = node_feat.to(device)
            global_feat = global_feat.to(device)
            target = target.to(device)
            # 修改前
            # prediction = model(node_feat, global_feat, adjacency)

            # 修改后
            prediction, _, _ = model(node_feat, global_feat, adjacency)

            for i in range(len(pcb_batch_ids)):
                pcb_id = int(pcb_batch_ids[i])
                sample_pred = prediction[i].cpu().numpy()  # [20, 2400]
                sample_target = target[i].cpu().numpy()    # [20, 2400]
                sample_freq = freq_batch[i].cpu().numpy()  # [2400]

                # 将 [20, 2400] 拆为 real/imag 两个 [2400, 10] 矩阵，方便绘图和分析。
                pred_real = np.stack([sample_pred[2 * k] for k in range(len(S_PARAM_NAMES))], axis=1)
                pred_imag = np.stack([sample_pred[2 * k + 1] for k in range(len(S_PARAM_NAMES))], axis=1)
                target_real = np.stack([sample_target[2 * k] for k in range(len(S_PARAM_NAMES))], axis=1)
                target_imag = np.stack([sample_target[2 * k + 1] for k in range(len(S_PARAM_NAMES))], axis=1)

                sample_save_dict = {
                    "pred_real": pred_real.astype(np.float32),        # [2400, 10]
                    "pred_imag": pred_imag.astype(np.float32),        # [2400, 10]
                    "target_real": target_real.astype(np.float32),    # [2400, 10]
                    "target_imag": target_imag.astype(np.float32),    # [2400, 10]
                    "freq_ghz": sample_freq.astype(np.float32),       # [2400]
                    "param_names": np.array(S_PARAM_NAMES),
                }
                np.savez(os.path.join(test_results_dir, f"pcb_{pcb_id}_raw_data.npz"), **sample_save_dict)

            all_preds.append(prediction.cpu())
            all_labels.append(target.cpu())
            all_pcb_ids.extend([int(x) for x in pcb_batch_ids])

    preds = torch.cat(all_preds)
    true = torch.cat(all_labels)

    overall_metrics = calculate_metrics(true, preds)
    with open(os.path.join(metrics_dir, "overall_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=4)

    sample_metrics = calculate_sample_metrics(true, preds)
    per_sample_df = pd.DataFrame(sample_metrics, index=all_pcb_ids)
    per_sample_df.to_csv(os.path.join(metrics_dir, "per_sample_metrics.csv"))

    print("Testing completed. Run code/double_trace/plot.py for per-sample plots.")


if __name__ == "__main__":
    main()
