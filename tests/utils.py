"""Helper utilities for testing."""

from typing import Dict

import torch


def get_next_batch(dataloader):
    return next(tensors for tensors in dataloader)


def copy_module_state_dict(module) -> Dict[str, torch.Tensor]:
    copy = {}
    for name, param in module.state_dict().items():
        copy[name] = param.detach().cpu().clone()
    return copy
