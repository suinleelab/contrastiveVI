"""Helper utilities for testing."""


def get_next_batch(dataloader):
    return next(tensors for tensors in dataloader)
