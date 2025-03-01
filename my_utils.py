import torch


def get_vmem_str():
    n = torch.cuda.memory_allocated()
    n = n / 1024 / 1024
    if n < 1024:
        return f'{n:.1f} MB'
    return f'{n / 1024:.1f} GB'
