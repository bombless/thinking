import torch

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
    rot_mat = torch.view_as_complex(torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1))
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

batch = 1
len1 = 5
len2 = 95
dim = 8

x = torch.randn(batch, len1, dim)
y = torch.randn(batch, len2, dim)

freqs_full = get_rope_freqs(100, 8)

freqs1 = freqs_full[:5]
freqs2 = freqs_full[5:]


b1 = apply_rope(x, freqs1)
b2 = apply_rope(y, freqs2)

c = torch.cat((b1, b2), dim=1)

z = torch.cat((x, y), dim=1)
d = apply_rope(z, freqs_full)

print(c == d)

b1b = apply_rope(x, freqs_full)
print(b1b == b1)