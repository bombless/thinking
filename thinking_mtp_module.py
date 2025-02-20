from torch import nn
import torch

from transformer_block import TransformerBlock
from transformer_block import embed_dim

n_layers = 6


class OutHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        return self.network(x)


class ThinkingMtpMainModel(nn.Module):

    def __init__(self, embedding_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(n_layers)])

    def forward(self, x, mask=None):
        print('ThinkingMtpMainModel input ' + str(x.shape))
        for block in self.blocks:
            x = block(x, mask)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        """
        RMSNorm 归一化层
        :param dim: 归一化的特征维度
        :param eps: 避免除零的数值稳定项
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数 γ

    def forward(self, x):
        """
        :param x: 形状 (..., dim)，最后一个维度是 `dim`
        :return: 归一化后的 x
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)  # 计算 RMS
        return (x / rms) * self.weight  # 归一化并缩放



class MTPLinearProjection(nn.Module):
    def __init__(self, d):
        super(MTPLinearProjection, self).__init__()
        # 定义投影矩阵 M_k，形状为 (d, 2d)
        self.projection = nn.Linear(2 * d, d)

    def forward(self, h_prev, emb):
        """
        Args:
            h_prev: 前一个深度的表示，形状为 (batch_size, seq_len, d)
            emb: 当前深度的嵌入，形状为 (batch_size, seq_len, d)
        Returns:
            h_proj: 投影后的表示，形状为 (batch_size, seq_len, d)
        """
        # 拼接 h_prev 和 emb，形状为 (batch_size, seq_len, 2d)
        concatenated = torch.cat([h_prev, emb], dim=-1)

        # 通过投影矩阵 M_k 进行线性投影，形状为 (batch_size, seq_len, d)
        h_proj = self.projection(concatenated)

        return h_proj

class ThinkingMtpModule(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(n_layers)])

        self.rms_norm_left = RMSNorm(1)
        self.rms_norm_right = RMSNorm(1)

        self.linear_projection = MTPLinearProjection(embed_dim)


    def forward(self, x, prev, mask=None):
        h = self.linear_projection(self.rms_norm_left(prev), self.rms_norm_right(x))
        for block in self.blocks:
            h = block(h, mask)
        return h
