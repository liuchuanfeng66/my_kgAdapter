import torch
import torch.nn as nn
import torch.nn.functional as F

class KGAdapterLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(KGAdapterLayer, self).__init__()
        self.hidden_dim = hidden_dim

        # 可训练线性变换矩阵
        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wo = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 三元组融合的MLP
        self.MLP_trip = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, node_reps, edge_reps, adjacency_list):
        """
        :param node_reps: 当前层的节点表示，大小为 (num_nodes, hidden_dim)
        :param edge_reps: 当前层的边表示，大小为 (num_edges, hidden_dim)
        :param adjacency_list: 相邻节点的索引和对应边的索引
        :return: 更新的节点表示和三元组表示
        """
        # 使用RGAT更新节点表示????
        updated_node_reps = []
        for i, (node, neighbors) in enumerate(adjacency_list.items()):
            # 获取该节点的所有邻居节点及对应的边
            neighbor_nodes = [node_reps[j] for j, _ in neighbors]
            neighbor_edges = [edge_reps[edge_idx] for _, edge_idx in neighbors]

            # 计算注意力系数
            attention_weights = []
            for j, (neighbor_node, edge_rep) in enumerate(zip(neighbor_nodes, neighbor_edges)):
                q = self.Wq(node_reps[i]) # h_ni * Wq
                k = self.Wk(neighbor_node) + edge_rep # h_nj * Wk + r_ij
                attention_score = torch.matmul(q, k.T) / (self.hidden_dim ** 0.5)
                attention_weights.append(attention_score)

            # Softmax归一化
            attention_weights = F.softmax(torch.stack(attention_weights), dim=0)

            # 更新节点表示
            weighted_sum = sum(alpha * (self.Wv(neighbor_node) + edge_rep) 
                               for alpha, neighbor_node, edge_rep in zip(attention_weights, neighbor_nodes, neighbor_edges))
            updated_node = self.layer_norm(node_reps[i] + self.Wo(weighted_sum))
            updated_node_reps.append(updated_node)

        updated_node_reps = torch.stack(updated_node_reps)

        # 三元组融合表示（relation-centered）
        triple_reps = []
        for (node_i, edge_idx, node_j) in adjacency_list:
            hi = node_reps[node_i]
            hj = node_reps[node_j]
            h_edge = edge_reps[edge_idx]

            h_triplet = torch.cat([hi, h_edge, hj], dim=-1)
            fused_triplet = self.MLP_trip(h_triplet)
            triple_reps.append(fused_triplet)

        triple_reps = torch.stack(triple_reps)
        return updated_node_reps, triple_reps