import numpy as np
import pytest
import scvi


@pytest.fixture
def mock_adata():
    adata = scvi.data.synthetic_iid(run_setup_anndata=False)
    adata.layers["raw_counts"] = adata.X.copy()
    adata.obs["my_categorical_covariate"] = ["A"] * 200 + ["B"] * 200
    adata.obs["my_continuous_covariate"] = np.random.randint(0, 100, 400)
    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key="batch",
        labels_key="labels",
        layer="raw_counts",
        categorical_covariate_keys=["my_categorical_covariate"],
        continuous_covariate_keys=["my_continuous_covariate"],
    )
    return adata


@pytest.fixture
def mock_adata_background_indices(mock_adata):
    return (
        mock_adata.obs.index[mock_adata.obs["my_categorical_covariate"] == "A"]
        .astype(int)
        .tolist()
    )


@pytest.fixture
def mock_adata_background_label(mock_adata):
    return np.where(
        mock_adata.uns["_scvi"]["extra_categoricals"]["mappings"][
            "my_categorical_covariate"
        ]
        == "A"
    )[0][0]


@pytest.fixture
def mock_adata_target_indices(mock_adata):
    return (
        mock_adata.obs.index[mock_adata.obs["my_categorical_covariate"] == "B"]
        .astype(int)
        .tolist()
    )


@pytest.fixture
def mock_adata_target_label(mock_adata):
    return np.where(
        mock_adata.uns["_scvi"]["extra_categoricals"]["mappings"][
            "my_categorical_covariate"
        ]
        == "B"
    )[0][0]
