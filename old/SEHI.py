import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class SEHIEmbedding(nn.Module):
    def __init__(self, llm_model_name, kg_hidden_size):
        super(SEHIEmbedding, self).__init__()
        kg_hidden_middle_size = 512
        # 加载预训练模型的分词器和嵌入模型
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_model = AutoModel.from_pretrained(llm_model_name)
        self.llm_hidden_size = self.llm_model.config.hidden_size
        
        # 定义MLP用于将子词表示下采样到KG的隐藏尺寸
        self.mlp = nn.Sequential(
            nn.Linear(self.llm_hidden_size, kg_hidden_middle_size),
            nn.ReLU(),
            nn.Linear(kg_hidden_middle_size, kg_hidden_size)
            )
        
    def embed_node(self, node_name):
        """
        将节点名称编码为子词级别嵌入并汇总为多词级别表示
        :param node_name: str, 节点名称
        :return: tensor, 下采样后的多词级别嵌入表示，形状为 [kg_hidden_size]
        """
        # 1. 使用分词器对节点名称进行编码
        inputs = self.tokenizer(node_name, return_tensors='pt')
        
        # 2. 获取语言模型的子词级别嵌入表示
        with torch.no_grad():
            outputs = self.llm_model(**inputs)
            subword_embeddings = outputs.last_hidden_state  # 形状: [1, num_subwords, hidden_size]
        
        # 3. 汇总子词嵌入得到多词级别表示
        multiword_embedding = torch.sum(subword_embeddings, dim=1).squeeze(0)  # 形状: [hidden_size]
        
        # 4. 通过MLP下采样到KG的隐藏尺寸
        multiword_embedding = self.mlp(multiword_embedding)  # 形状: [kg_hidden_size]
        
        return multiword_embedding


class SEHI(nn.Module):
    def __init__(self, llm_model_name, kg_hidden_size, final_hidden_size):
        super(SEHI, self).__init__()
        
        # 加载预训练语言模型的分词器和嵌入模型
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_model = AutoModel.from_pretrained(llm_model_name)
        llm_hidden_size = self.llm_model.config.hidden_size
        mlp1_middle_size = 2048
        mlp2_middle_size = 2048
        mlp3_middle_size = 32 
        # 定义多层感知机用于维度调整
        # mlp1 4096 -> 1024
        self.mlp1 = nn.Sequential(
            nn.Linear(llm_hidden_size, mlp1_middle_size),
            nn.ReLU(),
            nn.Linear(mlp1_middle_size, kg_hidden_size)
        )
        # mlp2 4096 -> 1024
        self.mlp2 = nn.Sequential(
            nn.Linear(kg_hidden_size, mlp2_middle_size),
            nn.ReLU(),
            nn.Linear(mlp2_middle_size, kg_hidden_size)
        )
        # mlp3 1024 -> 64
        self.mlp3 = nn.Sequential(
            nn.Linear(kg_hidden_size, mlp3_middle_size),
            nn.ReLU(),
            nn.Linear(mlp3_middle_size, final_hidden_size)
        )
        # 可学习的权重因子 σ
        self.sigma = nn.Parameter(torch.rand(1))

    def encode_node(self, node_name):
        """
        对节点名称进行编码，得到子词级别的嵌入表示
        """
        # 对节点名称进行分词，并转为tensor
        inputs = self.tokenizer(node_name, return_tensors='pt')
        
        # 获取LLM的子词级别嵌入
        with torch.no_grad():  # 仅需获取嵌入，不进行梯度计算
            outputs = self.llm_model(**inputs)
            subword_embeddings = outputs.last_hidden_state  # 形状: [1, num_subwords, hidden_size]
        
        # 移除batch维度，得到子词嵌入矩阵
        subword_embeddings = subword_embeddings.squeeze(0)  # 形状: [num_subwords, hidden_size]
        
        return subword_embeddings

    def forward(self, node_name):
        """
        获取节点的最终嵌入表示
        """
        # 1. 编码节点名称为子词级别的表示
        node_subword_representations = self.encode_node(node_name)  # 形状: [num_subwords, hidden_size]
        
        # 2. 求和子词嵌入以获得多词级别的表示
        h_n_mw = torch.sum(node_subword_representations, dim=0)  # 形状: [hidden_size]
        
        # 3. 下采样到KG的隐藏尺寸
        h_n_mw_tilde = self.mlp1(h_n_mw)  # 形状: [kg_hidden_size] 1024
        
        # 4. 更新预训练的KG表示
        h_n_mw_double_tilde = self.mlp2(h_n_mw_tilde)  # 形状: [kg_hidden_size] 1024
        
        # 5. 门控机制融合表示，并通过MLP3进一步下采样
        h_n = self.mlp3(self.sigma * h_n_mw_tilde + (1 - self.sigma) * h_n_mw_double_tilde)  # 形状: [final_hidden_size] 64
        
        return h_n
