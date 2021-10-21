"""Data loader for contrastive learning."""
from typing import List, Optional, Union

from anndata import AnnData
from scvi.dataloaders._concat_dataloader import ConcatDataLoader


class ContrastiveDataLoader(ConcatDataLoader):
    """
    Data loader to load background and target data for contrastive learning.

    Each iteration of the data loader returns a tuple, where the first element is the
    background data, and the second element is the target data.
    Args:
    ----
        adata: AnnData object that has been registered via `setup_anndata`.
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
        adata: AnnData,
        background_indices: List[int],
        target_indices: List[int],
        shuffle: bool = False,
        batch_size: int = 128,
        data_and_attributes: Optional[dict] = None,
        drop_last: Union[bool, int] = False,
        **data_loader_kwargs,
    ) -> None:
        super().__init__(
            adata=adata,
            indices_list=[background_indices, target_indices],
            shuffle=shuffle,
            batch_size=batch_size,
            data_and_attributes=data_and_attributes,
            drop_last=drop_last,
            **data_loader_kwargs,
        )
        self.background_indices = background_indices
        self.target_indices = target_indices
