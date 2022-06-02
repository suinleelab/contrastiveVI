import torch
from scvi import REGISTRY_KEYS

from contrastive_vi.data.dataloaders.contrastive_dataloader import ContrastiveDataLoader
from tests.utils import get_next_batch


class TestContrastiveDataLoader:
    def test_one_batch(
        self,
        mock_adata,
        mock_adata_background_indices,
        mock_adata_background_label,
        mock_adata_target_indices,
        mock_adata_target_label,
    ):
        batch_size = 32
        dataloader = ContrastiveDataLoader(
            mock_adata,
            mock_adata_background_indices,
            mock_adata_target_indices,
            batch_size=batch_size,
            shuffle=False,
        )
        batch = get_next_batch(dataloader)
        assert type(batch) == dict
        assert len(batch.keys()) == 2
        assert "background" in batch.keys()
        assert "target" in batch.keys()

        expected_background_data = torch.Tensor(
            mock_adata.layers["raw_counts"][mock_adata_background_indices, :][
                :batch_size, :
            ]
        )
        expected_target_data = torch.Tensor(
            mock_adata.layers["raw_counts"][mock_adata_target_indices, :][
                :batch_size, :
            ]
        )

        assert torch.equal(
            batch["background"][REGISTRY_KEYS.X_KEY], expected_background_data
        )
        assert torch.equal(batch["target"][REGISTRY_KEYS.X_KEY], expected_target_data)

        assert (
            batch["background"][REGISTRY_KEYS.BATCH_KEY] == mock_adata_background_label
        ).sum() == batch_size
        assert (
            batch["target"][REGISTRY_KEYS.BATCH_KEY] == mock_adata_target_label
        ).sum() == batch_size
