"""Data loader for contrastive learning."""
from itertools import cycle
from typing import List, Optional, Union

from scvi.data import AnnDataManager
from scvi.dataloaders._concat_dataloader import ConcatDataLoader


class _ContrastiveIterator:
    """
    Iterator for background and target dataloader pairs as found in the contrastive
    analysis setting.

    Each iteration of this iterator returns a dictionary with two elements:
    "background", containing one batch of data from the background dataloader, and
    "target", containing one batch of data from the target dataloader.
    """

    def __init__(self, background, target):
        self.background = iter(background)
        self.target = iter(target)

    def __iter__(self):
        return self

    def __next__(self):
        bg_samples = next(self.background)
        tg_samples = next(self.target)
        return {"background": bg_samples, "target": tg_samples}


class ContrastiveDataLoader(ConcatDataLoader):
    """
    Data loader to load background and target data for contrastive learning.

    Each iteration of the data loader returns a dictionary containing background and
    target data points, indexed by "background" and "target", respectively.
    Args:
    ----
        adata_manager: AnnDataManager object that has been created via `setup_anndata`.
        background_indices: Indices for background samples in `adata`.
        target_indices: Indices for target samples in `adata`.
        shuffle: Whether the data should be shuffled.
        batch_size: Mini-batch size to load for background and target data.
        data_and_attributes: Dictionary with keys representing keys in data
            registry (`adata.uns["_scvi"]`) and value equal to desired numpy
            loading type (later made into torch tensor). If `None`, defaults to all
            registered data.
        drop_last: If int, drops the last batch if its length is less than
            `drop_last`. If `drop_last == True`, drops last non-full batch.
            If `drop_last == False`, iterate over all batches.
        **data_loader_kwargs: Keyword arguments for `torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        background_indices: List[int],
        target_indices: List[int],
        shuffle: bool = False,
        batch_size: int = 128,
        data_and_attributes: Optional[dict] = None,
        drop_last: Union[bool, int] = False,
        **data_loader_kwargs,
    ) -> None:
        super().__init__(
            adata_manager=adata_manager,
            indices_list=[background_indices, target_indices],
            shuffle=shuffle,
            batch_size=batch_size,
            data_and_attributes=data_and_attributes,
            drop_last=drop_last,
            **data_loader_kwargs,
        )
        self.background_indices = background_indices
        self.target_indices = target_indices

    def __iter__(self):
        """

        Iter method for conctrastive data loader.

        Will iter over the dataloader with the most data while cycling through
        the data in the other dataloaders.
        """

        iter_list = [
            cycle(dl) if dl != self.largest_dl else dl for dl in self.dataloaders
        ]

        return _ContrastiveIterator(background=iter_list[0], target=iter_list[1])
