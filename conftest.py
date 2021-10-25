import numpy as np
import pytest
import scvi

from contrastive_vi.data.dataloaders.contrastive_dataloader import ContrastiveDataLoader
from tests.utils import get_next_batch


@pytest.fixture
def mock_adata():
    adata = scvi.data.synthetic_iid(run_setup_anndata=False, n_batches=2)
    adata.layers["raw_counts"] = adata.X.copy()
    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key="batch",
        labels_key="labels",
        layer="raw_counts",
    )
    return adata


@pytest.fixture
def mock_library_log_means(mock_adata):
    library_log_means = []
    for batch in mock_adata.uns["_scvi"]["categorical_mappings"]["_scvi_batch"][
        "mapping"
    ]:
        library = (
            mock_adata[mock_adata.obs["batch"] == batch].layers["raw_counts"].sum(1)
        )
        library_log = np.log(library)
        library_log_means.append(library_log.mean())
    library_log_means = np.array(library_log_means)[np.newaxis, :]
    return library_log_means


@pytest.fixture
def mock_library_log_vars(mock_adata):
    library_log_vars = []
    for batch in mock_adata.uns["_scvi"]["categorical_mappings"]["_scvi_batch"][
        "mapping"
    ]:
        library = (
            mock_adata[mock_adata.obs["batch"] == batch].layers["raw_counts"].sum(1)
        )
        library_log = np.log(library)
        library_log_vars.append(library_log.var())
    library_log_vars = np.array(library_log_vars)[np.newaxis, :]
    return library_log_vars


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
    return np.where(
        mock_adata.uns["_scvi"]["categorical_mappings"]["_scvi_batch"]["mapping"]
        == "batch_0"
    )[0][0]


@pytest.fixture
def mock_adata_target_indices(mock_adata):
    return (
        mock_adata.obs.index[(mock_adata.obs["batch"] == "batch_1")]
        .astype(int)
        .tolist()
    )


@pytest.fixture
def mock_adata_target_label(mock_adata):
    return np.where(
        mock_adata.uns["_scvi"]["categorical_mappings"]["_scvi_batch"]["mapping"]
        == "batch_1"
    )[0][0]


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
