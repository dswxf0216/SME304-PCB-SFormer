import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import skrf as rf
import pandas as pd
import tqdm
import math
import random
import time
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from numpy.typing import NDArray

"""Single-trace S-parameter prediction example.

该脚本演示：
1) 从轨迹几何与 S2P 文件构建图数据；
2) 使用 GCN + Transformer 模型回归 S 参数；
3) 输出测试集指标与可视化结果。
"""


# ---------- 设置随机种子 ----------
def set_seed(random_seed: int) -> None:
    """固定随机种子，保证结果可复现。"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

GLOBAL_SEED = 42
set_seed(GLOBAL_SEED)

# ---------- 模型定义 ----------
class SimpleGCNLayer(nn.Module):
    """最简图卷积层：A_hat X W，其中 A_hat 为归一化邻接矩阵。"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = nn.Linear(in_features, out_features)

    def forward(self, node_features, adjacency_matrix):
        # _, _, _ = adjacency_matrix.size()
        degree_inv_sqrt = torch.sum(adjacency_matrix, dim=-1) ** -0.5
        degree_matrix = torch.diag_embed(degree_inv_sqrt)
        normalized_adjacency = torch.bmm(torch.bmm(degree_matrix, adjacency_matrix), degree_matrix)
        aggregated_features = torch.bmm(normalized_adjacency, node_features)
        projected_features = self.mlp(aggregated_features)
        return F.relu(projected_features)

class SingleTraceTransformer(nn.Module):
    """结合 GCN 与 Transformer 的单根走线建模网络。"""

    def __init__(self, in_features, global_features_dim, hidden_features, out_features, num_nodes, num_layers=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.gcn1 = SimpleGCNLayer(in_features, hidden_features)
        self.mlp_mid = nn.Sequential(
            nn.Linear(hidden_features + global_features_dim, hidden_features),
            nn.ReLU()
        )
        self.gcn2 = SimpleGCNLayer(hidden_features, hidden_features)
        enc1_layer = TransformerEncoderLayer(hidden_features, nhead=4, dim_feedforward=hidden_features*2, batch_first=True)
        self.enc1 = TransformerEncoder(enc1_layer, num_layers=num_layers//2)
        self.weight_gcn2 = nn.Parameter(torch.ones(1))
        self.weight_enc1 = nn.Parameter(torch.ones(1))
        self.mlp_fusion = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU()
        )
        enc2_layer = TransformerEncoderLayer(hidden_features, nhead=4, dim_feedforward=hidden_features*2, batch_first=True)
        self.enc2 = TransformerEncoder(enc2_layer, num_layers=num_layers//2)
        self.fc = nn.Linear(hidden_features * num_nodes, out_features)

    def forward(self, node_features, global_features, adjacency_matrix):
        batch_size = node_features.size(0)

        # 步骤1：局部图结构编码
        local_graph_features = self.gcn1(node_features, adjacency_matrix)

        # 步骤2：融合每个节点对应的全局特征（长度、线宽、介质厚度）
        fused_input_features = torch.cat([local_graph_features, global_features], dim=-1)
        hidden_features_mid = self.mlp_mid(fused_input_features)

        # 步骤3：GCN 与 Transformer 双分支提特征，再做可学习加权融合
        graph_branch_features = self.gcn2(hidden_features_mid, adjacency_matrix)
        transformer_branch_features = self.enc1(hidden_features_mid)
        fused_branch_features = (
            self.weight_gcn2 * graph_branch_features
            + self.weight_enc1 * transformer_branch_features
        )

        # 步骤4：深层序列编码 -> 拉平 -> 回归输出
        refined_features = self.mlp_fusion(fused_branch_features)
        encoded_features = self.enc2(refined_features)
        flattened_features = encoded_features.contiguous().view(batch_size, -1)
        regression_output = self.fc(flattened_features)

        # 输出维度：[batch, 6, freq_points]
        return regression_output.view(batch_size, 6, -1)

# ---------- 数据加载函数 ----------
def load_novia_data(data_dir, max_samples=None):
    """加载无过孔（novia）数据，构建图输入与监督标签。"""

    traces_file = os.path.join(os.path.dirname(data_dir), "pcb_traces.txt")
    traces_df = pd.read_csv(traces_file, sep=',', header=0)
    grouped_by_pcb = traces_df.groupby('PCB_Index')
    pcb_ids = list(grouped_by_pcb.groups.keys())
    if max_samples is not None:
        pcb_ids = pcb_ids[:max_samples]

    adjacency_matrices, node_features_list, global_features_list, labels_list, pcb_id_list = [], [], [], [], []
    width, dielectric_thickness = 0.041, 0.635

    for pcb_id in pcb_ids:
        pcb_group = grouped_by_pcb.get_group(pcb_id).sort_values('Node_Index')
        xs: NDArray[np.float32] = pcb_group['X'].to_numpy(dtype=np.float32)
        ys: NDArray[np.float32] = pcb_group['Y'].to_numpy(dtype=np.float32)
        real_node_count = len(xs)

        # 为单线结构加入一个“假点”（dummy node）。
        # 使用首尾节点中点作为假点坐标，便于和原几何尺度保持一致。
        dummy_x = np.float32((xs[0] + xs[-1]) / 2.0)
        dummy_y = np.float32((ys[0] + ys[-1]) / 2.0 - 15.0)  # 在 Y 方向上偏移一定距离，形成一个明显的假点位置。
        xs_with_dummy = np.concatenate([xs, np.array([dummy_x], dtype=np.float32)])
        ys_with_dummy = np.concatenate([ys, np.array([dummy_y], dtype=np.float32)])

        num_nodes = len(xs_with_dummy)
        dist_from_start: NDArray[np.float32] = np.sqrt(
            (xs_with_dummy - xs[0])**2 + (ys_with_dummy - ys[0])**2
        ).astype(np.float32)
        node_features = np.stack((xs_with_dummy, ys_with_dummy, dist_from_start), axis=1).astype(np.float32)
        segment_lengths = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        total_length = np.sum(segment_lengths)

        global_features = np.zeros((num_nodes, 3), dtype=np.float32)
        global_features[:, 0], global_features[:, 1], global_features[:, 2] = total_length, width, dielectric_thickness

        # 单根走线节点按顺序相连：i <-> i+1。
        # 边权重使用相邻两点走线长度。
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i in range(real_node_count - 1):
            edge_length = segment_lengths[i].astype(np.float32)
            adjacency_matrix[i, i+1] = edge_length
            adjacency_matrix[i+1, i] = edge_length

        # 假点与首尾节点相连，使整条线形成一个闭合图结构。
        # 新增边权重同样采用欧氏距离。
        dummy_node_index = real_node_count
        edge_dummy_to_head = np.float32(np.sqrt((dummy_x - xs[0])**2 + (dummy_y - ys[0])**2))
        edge_dummy_to_tail = np.float32(np.sqrt((dummy_x - xs[-1])**2 + (dummy_y - ys[-1])**2))
        adjacency_matrix[dummy_node_index, 0] = edge_dummy_to_head
        adjacency_matrix[0, dummy_node_index] = edge_dummy_to_head
        adjacency_matrix[dummy_node_index, real_node_count - 1] = edge_dummy_to_tail
        adjacency_matrix[real_node_count - 1, dummy_node_index] = edge_dummy_to_tail

        s2p_path = os.path.join(data_dir, f"{pcb_id}.s2p")
        if not os.path.exists(s2p_path):
            continue

        network = rf.Network(s2p_path)
        s_parameters = np.asarray(network.s)

        # 标签包含 S11、S21、S22。
        # 0: S11_R, 1: S11_I, 2: S21_R, 3: S21_I, 4: S22_R, 5: S22_I
        labels = np.stack([
            np.real(s_parameters[:, 0, 0]), np.imag(s_parameters[:, 0, 0]),
            np.real(s_parameters[:, 1, 0]), np.imag(s_parameters[:, 1, 0]),
            np.real(s_parameters[:, 1, 1]), np.imag(s_parameters[:, 1, 1])
        ]).astype(np.float32) # [6, 961]

        adjacency_matrices.append(adjacency_matrix)
        node_features_list.append(node_features)
        global_features_list.append(global_features)
        labels_list.append(labels)
        pcb_id_list.append(pcb_id)

    return adjacency_matrices, node_features_list, global_features_list, labels_list, pcb_id_list

class SingleTraceDataset(Dataset):
    """将预处理后的 numpy 列表包装为 PyTorch Dataset。"""

    def __init__(self, adj_matrices, node_features, global_features, labels, pcb_ids):
        self.adj_matrices, self.node_features, self.global_features, self.labels, self.pcb_ids = adj_matrices, node_features, global_features, labels, pcb_ids
    def __len__(self): return len(self.adj_matrices)
    def __getitem__(self, idx):
        return (torch.tensor(self.adj_matrices[idx], dtype=torch.float32), 
                torch.tensor(self.node_features[idx], dtype=torch.float32), 
                torch.tensor(self.global_features[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32), 
                self.pcb_ids[idx])

# ---------- 评价指标与绘图函数 ----------
def huber_loss(y_true, y_pred, alpha=0.5):
    mae = torch.mean(torch.abs(y_true - y_pred))
    mse = torch.mean((y_true - y_pred) ** 2)
    return (1 - alpha) * mae + alpha * mse

def calculate_metrics(y_true, y_pred):
    """计算总体及分通道指标"""
    y_true_array = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred_array = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
    metrics = {
        'mse': mean_squared_error(y_true_array.flatten(), y_pred_array.flatten()),
        'rmse': np.sqrt(mean_squared_error(y_true_array.flatten(), y_pred_array.flatten())),
        'mae': mean_absolute_error(y_true_array.flatten(), y_pred_array.flatten()),
        'r2': r2_score(y_true_array.flatten(), y_pred_array.flatten())
    }
    s_params = ['S11_real', 'S11_imag', 'S21_real', 'S21_imag', 'S22_real', 'S22_imag']
    for i, param in enumerate(s_params):
        true_channel, pred_channel = y_true_array[:, i, :].flatten(), y_pred_array[:, i, :].flatten()
        metrics[f'{param}_mse'] = mean_squared_error(true_channel, pred_channel)
        metrics[f'{param}_mae'] = mean_absolute_error(true_channel, pred_channel)
    return metrics

def calculate_sample_metrics(y_true, y_pred):
    """计算单个样本指标"""
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    sample_metrics = []
    s_params = ['S11_real', 'S11_imag', 'S21_real', 'S21_imag', 'S22_real', 'S22_imag']
    for i in range(y_true_np.shape[0]):
        true_sample, pred_sample = y_true_np[i], y_pred_np[i]
        d = {
            'mse': mean_squared_error(true_sample.flatten(), pred_sample.flatten()),
            'mae': mean_absolute_error(true_sample.flatten(), pred_sample.flatten())
        }
        for j, param in enumerate(s_params):
            d[f'{param}_mse'] = mean_squared_error(true_sample[j], pred_sample[j])
        sample_metrics.append(d)
    return sample_metrics

def plot_pred_vs_truth(y_true, y_pred, sample_index, pcb_id, save_path):
    """
    绘制S参数对比图 (包含 S11, S21, S22 的实部与虚部)
    y_true/y_pred shape: [batch, 6, freq_len]
    """
    s_params = [
        'S11_real', 'S11_imag', 
        'S21_real', 'S21_imag', 
        'S22_real', 'S22_imag'
    ]
    
    freq = np.linspace(0.4, 10, y_true.shape[2]) # 频率范围与点数根据实际数据调整
    
    # 3 行 2 列布局。
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.flatten()
    
    for i, param in enumerate(s_params):
        ax = axes[i]
        # 提取对应通道的真实值和预测值。
        true_values = y_true[sample_index, i, :]
        pred_values = y_pred[sample_index, i, :]
        
        # 绘图并设置标题、坐标轴与网格。
        ax.plot(freq, true_values, color='blue', linewidth=2.5, label='True (Target)', alpha=0.6)
        ax.plot(freq, pred_values, color='red', linewidth=1.5, linestyle='--', label='Pred (Model)')
        
        ax.set_title(f'{param} (PCB {pcb_id})', fontweight='bold', fontsize=12)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 计算相关系数（Correlation）。
        corr = np.corrcoef(true_values, pred_values)[0, 1]
        ax.text(0.05, 0.95, f'Corr: {corr:.4f}', 
                transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=10, verticalalignment='top')
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_training_curves(train_losses, valid_losses, save_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(valid_losses, label='Valid')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()

# ---------- 主函数 ----------
def main():
    """训练模型并在测试集上评估与可视化。"""

    data_dir = "/home/dengnuo/share/sformer/data/single_trace/data"
    save_dir = "/home/dengnuo/share/sformer/results_single1"
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 早停参数。
    early_stop_patience = 50  # 连续50个epoch不进步则停止

    # 步骤1：数据准备。
    adj_matrices, node_features, global_features, labels, pcb_ids = load_novia_data(data_dir)
    dataset = SingleTraceDataset(adj_matrices, node_features, global_features, labels, pcb_ids)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # 步骤2：模型与优化器。
    model = SingleTraceTransformer(in_features=3, global_features_dim=3, hidden_features=64, 
                                   out_features=6 * labels[0].shape[1], # y_true/y_pred shape: [batch, 6, freq_len]
                                   num_nodes=adj_matrices[0].shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    train_losses, val_losses, best_loss = [], [], float('inf')
    epochs_no_improve = 0
    
    # 步骤3：训练与验证。
    print("Starting training...")
    for epoch in range(1000):
        model.train()
        train_loss_sum = 0

        for adjacency, node_feat, global_feat, target, _ in train_loader:
            adjacency = adjacency.to(device)
            node_feat = node_feat.to(device)
            global_feat = global_feat.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            prediction = model(node_feat, global_feat, adjacency)
            loss = huber_loss(target, prediction)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()
        
        avg_train_loss = train_loss_sum / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for adjacency, node_feat, global_feat, target, _ in val_loader:
                adjacency = adjacency.to(device)
                node_feat = node_feat.to(device)
                global_feat = global_feat.to(device)
                target = target.to(device)
                val_loss_sum += huber_loss(target, model(node_feat, global_feat, adjacency)).item()

        avg_val_loss = val_loss_sum / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train {avg_train_loss:.6f}, Val {avg_val_loss:.6f}")
        if epochs_no_improve >= early_stop_patience:
            print(f"触发早停，停止于 epoch {epoch}")
            break

    plot_training_curves(train_losses, val_losses, save_dir)
    
    # ---------- 测试阶段 ----------
    print("\nStarting testing & visualization...")
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    model.eval()
    test_results_dir = os.path.join(save_dir, 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)

    all_preds, all_labels, all_pcb_ids = [], [], []
    with torch.no_grad():
        for adjacency, node_feat, global_feat, target, pcb_batch_ids in test_loader:
            adjacency = adjacency.to(device)
            node_feat = node_feat.to(device)
            global_feat = global_feat.to(device)
            target = target.to(device)
            prediction = model(node_feat, global_feat, adjacency)

            for i in range(len(pcb_batch_ids)):
                pcb_id = pcb_batch_ids[i]
                sample_pred = prediction[i].cpu().numpy()   # [6, 961]
                sample_target = target[i].cpu().numpy()     # [6, 961]
                
                sample_save_dict = {
                    # Reflection 分组包含 S11 和 S22。
                    'reflection_pred_real': np.stack([sample_pred[0], sample_pred[4]], axis=1),   # [961, 2]
                    'reflection_pred_imag': np.stack([sample_pred[1], sample_pred[5]], axis=1),   # [961, 2]
                    'reflection_target_real': np.stack([sample_target[0], sample_target[4]], axis=1),
                    'reflection_target_imag': np.stack([sample_target[1], sample_target[5]], axis=1),
                    
                    # Transmission 分组包含 S21。
                    'transmission_pred_real': sample_pred[2:3, :].T,    # [961, 1]
                    'transmission_pred_imag': sample_pred[3:4, :].T,    # [961, 1]
                    'transmission_target_real': sample_target[2:3, :].T,
                    'transmission_target_imag': sample_target[3:4, :].T,
                }
                np.savez(os.path.join(test_results_dir, f"pcb_{pcb_id}_raw_data.npz"), **sample_save_dict)

            all_preds.append(prediction.cpu())
            all_labels.append(target.cpu())
            all_pcb_ids.extend(pcb_batch_ids)

    preds, true = torch.cat(all_preds), torch.cat(all_labels)
    
    # 步骤1：保存指标。
    metrics_dir = os.path.join(save_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    overall_metrics = calculate_metrics(true, preds)
    with open(os.path.join(metrics_dir, 'overall_metrics.json'), 'w') as f:
        json.dump(overall_metrics, f, indent=4)
    
    sample_metrics = calculate_sample_metrics(true, preds)
    per_sample_df = pd.DataFrame(sample_metrics, index=all_pcb_ids)

    per_sample_df.to_csv(os.path.join(metrics_dir, 'per_sample_metrics.csv'))

    # 步骤2：逐样本作图由 plot.py 统一处理。
    print("Test results saved. Please run plot.py to generate per-sample figures.")



if __name__ == "__main__":
    main()