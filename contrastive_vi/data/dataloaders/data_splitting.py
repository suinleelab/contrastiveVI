"""Utilities for splitting a dataset into training, validation, and test set."""

from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
from anndata import AnnData
from scvi import settings
from scvi.dataloaders._data_splitting import validate_data_split
from scvi.model._utils import parse_use_gpu_arg

from contrastive_vi.data.dataloaders.contrastive_dataloader import ContrastiveDataLoader


class ContrastiveDataSplitter(pl.LightningDataModule):
    """
    Create ContrastiveDataLoader for training, validation, and test set.

    Args:
    ----
        adata: AnnData object that has been registered via `setup_anndata`.
        background_indices: Indices for background samples in `adata`.
        target_indices: Indices for target samples in `adata`.
        train_size: Proportion of data to include in the training set.
        validation_size: Proportion of data to include in the validation set. The
            remaining proportion after `train_size` and `validation_size` is used for
            the test set.
        use_gpu: Use default GPU if available (if None or True); or index of GPU to
            use (if int); or name of GPU (if str, e.g., `'cuda:0'`); or use CPU
            (if False).
        **kwargs: Keyword args for data loader (`ContrastiveDataLoader`).
    """

    def __init__(
        self,
        adata: AnnData,
        background_indices: List[int],
        target_indices: List[int],
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        use_gpu: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.adata = adata
        self.background_indices = background_indices
        self.target_indices = target_indices
        self.train_size = train_size
        self.validation_size = validation_size
        self.use_gpu = use_gpu
        self.data_loader_kwargs = kwargs

        self.n_background = len(background_indices)
        self.n_target = len(target_indices)
        self.n_background_train, self.n_background_val = validate_data_split(
            len(self.background_indices), self.train_size, self.validation_size
        )
        self.n_target_train, self.n_target_val = validate_data_split(
            len(self.target_indices), self.train_size, self.validation_size
        )

    def setup(self, stage: Optional[str] = None):
        random_state = np.random.RandomState(seed=settings.seed)

        background_permutation = random_state.permutation(self.background_indices)
        n_background_train = self.n_background_train
        n_background_val = self.n_background_val
        self.background_val_idx = background_permutation[:n_background_val]
        self.background_train_idx = background_permutation[
            n_background_val : (n_background_val + n_background_train)
        ]
        self.background_test_idx = background_permutation[
            (n_background_val + n_background_train) :
        ]

        target_permutation = random_state.permutation(self.target_indices)
        n_target_train = self.n_target_train
        n_target_val = self.n_target_val
        self.target_val_idx = target_permutation[:n_target_val]
        self.target_train_idx = target_permutation[
            n_target_val : (n_target_val + n_target_train)
        ]
        self.target_test_idx = target_permutation[(n_target_val + n_target_train) :]

        self.train_idx = self.background_train_idx + self.target_train_idx
        self.val_idx = self.background_val_idx + self.target_val_idx
        self.test_idx = self.background_test_idx + self.target_test_idx

        gpus, self.device = parse_use_gpu_arg(self.use_gpu, return_device=True)
        self.pin_memory = (
            True if (settings.dl_pin_memory_gpu_training and gpus != 0) else False
        )

    def _get_contrastive_dataloader(
        self, background_indices: List[int], target_indices: List[int]
    ) -> ContrastiveDataLoader:
        return ContrastiveDataLoader(
            self.adata,
            background_indices,
            target_indices,
            shuffle=True,
            drop_last=3,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def train_dataloader(self) -> ContrastiveDataLoader:
        return self._get_contrastive_dataloader(
            self.background_train_idx, self.target_train_idx
        )

    def val_dataloader(self) -> ContrastiveDataLoader:
        if len(self.background_val_idx) > 0 and len(self.target_val_idx) > 0:
            return self._get_contrastive_dataloader(
                self.background_val_idx, self.target_val_idx
            )
        else:
            pass

    def test_dataloader(self) -> ContrastiveDataLoader:
        if len(self.background_test_idx) > 0 and len(self.target_test_idx) > 0:
            return self._get_contrastive_dataloader(
                self.background_test_idx, self.target_test_idx
            )
        else:
            pass
