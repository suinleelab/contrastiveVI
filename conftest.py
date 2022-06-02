import pytest
import scvi
from scvi.model._utils import _init_library_size

from contrastive_vi.data.dataloaders.contrastive_dataloader import ContrastiveDataLoader
from contrastive_vi.model.contrastive_vi import ContrastiveVIModel
from tests.utils import get_next_batch


@pytest.fixture
def mock_adata():
    adata = scvi.data.synthetic_iid(n_batches=2)  # Same number of cells in each batch.
    # Make number of cells unequal across batches to test edge cases.
    adata = adata[:-3, :]
    adata.layers["raw_counts"] = adata.X.copy()
    ContrastiveVIModel.setup_anndata(
        adata=adata,
        batch_key="batch",
        labels_key="labels",
        layer="raw_counts",
    )
    return adata


@pytest.fixture
def mock_adata_manager(mock_adata):
    return ContrastiveVIModel._setup_adata_manager_store[mock_adata.uns["_scvi_uuid"]]


@pytest.fixture
def mock_library_log_means_and_vars(mock_adata_manager):
    return _init_library_size(mock_adata_manager, n_batch=2)


@pytest.fixture
def mock_library_log_means(mock_library_log_means_and_vars):
    return mock_library_log_means_and_vars[0]


@pytest.fixture
def mock_library_log_vars(mock_library_log_means_and_vars):
    return mock_library_log_means_and_vars[1]


@pytest.fixture
def mock_n_input(mock_adata):
    return mock_adata.X.shape[1]


@pytest.fixture
def mock_n_batch(mock_adata):
    return len(mock_adata.obs["batch"].unique())


@pytest.fixture
def mock_adata_background_indices(mock_adata):
    return (
        mock_adata.obs.index[(mock_adata.obs["batch"] == "batch_0")]
        .astype(int)
        .tolist()
    )


@pytest.fixture
def mock_adata_background_label(mock_adata):
    return 0


@pytest.fixture
def mock_adata_target_indices(mock_adata):
    return (
        mock_adata.obs.index[(mock_adata.obs["batch"] == "batch_1")]
        .astype(int)
        .tolist()
    )


@pytest.fixture
def mock_adata_target_label(mock_adata):
    return 1


@pytest.fixture
def mock_contrastive_dataloader(
    mock_adata, mock_adata_background_indices, mock_adata_target_indices
):
    return ContrastiveDataLoader(
        mock_adata,
        mock_adata_background_indices,
        mock_adata_target_indices,
        batch_size=32,
        shuffle=False,
    )


@pytest.fixture
def mock_contrastive_batch(mock_contrastive_dataloader):
    return get_next_batch(mock_contrastive_dataloader)
