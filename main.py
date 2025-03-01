from torch import optim

from thinking_mtp_module import OutHead, ThinkingMtpModule, ThinkingMtpMainModel
from transformer_block import embed_dim, TransformerBlock
from tokenizer import Encoder

from my_utils import get_vmem_str

import torch

import torch.nn as nn

import torch.nn.functional as F



n_thinking_blocks = 64
n_generating_layers = 6

context_size = 64

tok = Encoder()
tok.init()
vocab_size = tok.size()


def create_causal_mask(seq_len):
    # 创建一个下三角矩阵作为自回归掩码
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0) 

class ThinkingNetwork(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.out_head = OutHead()
        self.mtp_modules = nn.ModuleList([ThinkingMtpModule(embedding_layer) for _ in range(n_thinking_blocks - 1)])
        self.mtp_main_model = ThinkingMtpMainModel(embedding_layer)

    def forward(self, x, mask=None):
        print('ThinkingNetwork forward: ' + get_vmem_str())
        prev = self.mtp_main_model(x[0], mask)
        print('ThinkingNetwork forward: mtp_main_model calculated: ' + get_vmem_str())
        mtp_main_result = self.out_head(prev)
        result = [mtp_main_result]
        for line in x:
            line_result = []
            for i in range(n_thinking_blocks - 1):
                mod = self.mtp_modules[i]
                prev = mod(line, prev, mask)
                h = self.out_head(prev)
                line_result.append(h)
                print('ThinkingNetwork forward: mtp_modules: one mtp module calculated: ' + get_vmem_str())
            print('ThinkingNetwork forward: mtp_modules: one line calculated: ' + get_vmem_str())
            result.append(line_result)
        return result

class GeneratingNetwork(nn.Module):

    def __init__(self, embedding_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.mlp = nn.Linear(embed_dim, vocab_size)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(n_generating_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return self.mlp(x)

def apply_rope(x, freqs):
    """
    对输入张量 x 应用 RoPE 旋转。
    x: 形状为 (batch, seq_len, dim) 的张量
    freqs: 旋转频率，形状为 (seq_len, dim // 2)
    """
    # 将 x 的最后一维拆分为实部和虚部
    x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
    # 生成旋转矩阵
    freqs = freqs.unsqueeze(0)  # 增加 batch 维度
    rot_mat = torch.view_as_complex(torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1).cuda())
    # 应用旋转
    x_rotated = x_complex * rot_mat
    # 将结果转换回实数形式
    x_rotated = torch.view_as_real(x_rotated).flatten(-2)
    return x_rotated

def get_rope_freqs(seq_len, dim, base=10000.0):
    """
    生成 RoPE 旋转频率。
    seq_len: 序列长度
    dim: 特征维度
    base: RoPE 的基数
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # 外积，形状为 (seq_len, dim // 2)
    return freqs







class Model:

    def __init__(self):
        super().__init__()
        embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer = embedding_layer
        self.thinking_network = ThinkingNetwork(embedding_layer)
        self.generating_network = GeneratingNetwork(embedding_layer)
        
        # 将所有模型参数移动到GPU
        self.embedding_layer = self.embedding_layer.cuda()
        self.thinking_network = self.thinking_network.cuda()
        self.generating_network = self.generating_network.cuda()

    def get_thinking_input(self, emb):
        pad = tok.encode("<pad>")[0]
        pad = self.embedding_layer(torch.tensor(pad).cuda()).unsqueeze(0).unsqueeze(0)
        for _ in range(emb.shape[1], n_thinking_blocks):
            emb = torch.cat((emb, pad), dim=-2)
        freqs = get_rope_freqs(emb.shape[-2], embed_dim)

        result = []
        for i in range(n_thinking_blocks):
            row = emb.clone()[:,i:,:]
            # print('emb.shape ' + str(emb.shape))
            # print('row.shape ' + str(row.shape))
            # print('pad.shape ' + str(pad.shape))
            for _ in range(i):
                row = torch.cat((row, pad), dim=-2)
            rope_result = apply_rope(row, freqs)
            result.append(rope_result)
        return torch.stack(result, dim=0)

    def forward_thinking(self, network: ThinkingNetwork, tokens_data, mask=None):
        emb = self.embedding_layer(tokens_data)
        print('forward_thinking: emb calculated: ' + get_vmem_str())
        rope_result = self.get_thinking_input(emb)
        print('forward_thinking: got rope_result: ' + get_vmem_str())
        causal_mask = create_causal_mask(rope_result.size(-2)).to(rope_result.device)
        print('forward_thinking: got causal_mask: ' + get_vmem_str())
        return network(rope_result, causal_mask)

    def forward_generating(self, thinking_network: ThinkingNetwork, generating_network: GeneratingNetwork, tokens_data, mask=None):
        emb = self.embedding_layer(tokens_data)
        prompt_rope_result = self.get_thinking_input(emb)
        causal_mask = create_causal_mask(prompt_rope_result.size(-2)).to(prompt_rope_result.device)
        thinking_result = thinking_network(prompt_rope_result, causal_mask)

        combined = torch.cat((emb, torch.stack(thinking_result, dim=0)), dim=-2)
        freqs = get_rope_freqs(n_thinking_blocks, embed_dim)
        combined_rope_result = apply_rope(combined, freqs)
        return generating_network(combined_rope_result, mask)

    def get_thinking_training_data(self):
        prompt = tok.encode("9.11>9.9")
        thinking = tok.encode("9.11>9.9->0.11>0.9->11>90->0>79->false")
        tensor_thinking = torch.tensor(thinking).to('cuda')
        print(tensor_thinking)
        print('vocab_size ' + str(vocab_size))
        return torch.tensor(prompt).unsqueeze(0).to('cuda'), self.embedding_layer(tensor_thinking).unsqueeze(0)

    def train_thinking_step(self, inputs, label, optimizer, print_loss=False):
        optimizer.zero_grad()
        thinking_result = self.forward_thinking(self.thinking_network, inputs, True)
        print('memory allocated ' + get_vmem_str())
        result = torch.stack(thinking_result, dim=0)
        print('result ' + str(result.shape))
        print('label ' + str(label.shape))
        loss = F.cross_entropy(result, label)
        if print_loss:
            print(loss.item())
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

    def train_thinking(self, inputs, label):
        optimizer = optim.SGD(self.thinking_network.parameters(), lr=0.01)
        print('train_thinking: ' + get_vmem_str())
        for i in range(1000):
            print(f'train_thinking: step {i}: ' + get_vmem_str())
            print('train_thinking ' + str(inputs.shape))
            self.train_thinking_step(inputs, label, optimizer, i % 20 == 0)


def run():
    model = Model()
    inputs, label = model.get_thinking_training_data()
    model.train_thinking(inputs, label)

run()

