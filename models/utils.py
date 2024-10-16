import torch
from collections import OrderedDict


def model_size(model: torch.nn.Module) -> int:
    return sum([p.numel() for p in model.parameters()])


@torch.jit.script
def mask_tokens(x: torch.Tensor, mask_token: torch.Tensor, p: float) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape

    mask_pos = torch.rand(batch_size, seq_len, device=x.device) < p
    x[mask_pos] = mask_token.to(x.dtype)

    return x


@torch.jit.script
def topk_latent(z: torch.Tensor, k: int) -> torch.Tensor:
    v, idx = torch.topk(z, k)

    zk = torch.zeros_like(z)
    zk.scatter_(-1, idx, v)

    return zk


@torch.no_grad()
def ema_update(online: torch.nn.Module, target: torch.nn.Module, decay: float = 0.999):
    """
    ema update target model weights from online model using given decay.

    :param online: online model
    :param target: model to update
    :param decay: decay for ema
    :return: None
    """

    # 1. update parameters
    model_params = OrderedDict(online.named_parameters())
    target_params = OrderedDict(target.named_parameters())

    assert model_params.keys() == target_params.keys()

    for name, param in model_params.items():
        target_params[name].sub_((1.0 - decay) * (target_params[name] - param))

    # 2. copy buffers
    model_buffers = OrderedDict(online.named_buffers())
    target_buffers = OrderedDict(target.named_buffers())

    assert model_buffers.keys() == target_buffers.keys()

    for name, buffer in model_buffers.items():
        target_buffers[name].copy_(buffer)
