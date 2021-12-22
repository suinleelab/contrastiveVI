import numpy as np
import pandas as pd
import pytest
import torch

from contrastive_vi.model.contrastive_vi import ContrastiveVIModel
from tests.utils import copy_module_state_dict


@pytest.fixture(
    params=[True, False], ids=["with_observed_lib_size", "without_observed_lib_size"]
)
def mock_contrastive_vi_model(
    mock_adata,
    mock_adata_background_indices,
    mock_adata_target_indices,
    mock_library_log_means,
    mock_library_log_vars,
    request,
):
    if request.param:
        return ContrastiveVIModel(
            mock_adata,
            n_hidden=16,
            n_background_latent=4,
            n_salient_latent=4,
            n_layers=2,
            use_observed_lib_size=True,
        )
    else:
        return ContrastiveVIModel(
            mock_adata,
            n_hidden=16,
            n_background_latent=4,
            n_salient_latent=4,
            n_layers=2,
            use_observed_lib_size=False,
        )


class TestContrastiveVIModel:
    def test_train(
        self,
        mock_contrastive_vi_model,
        mock_adata_background_indices,
        mock_adata_target_indices,
    ):
        init_state_dict = copy_module_state_dict(mock_contrastive_vi_model.module)
        mock_contrastive_vi_model.train(
            background_indices=mock_adata_background_indices,
            target_indices=mock_adata_target_indices,
            max_epochs=10,
            batch_size=20,  # Unequal final batches to test edge case.
            use_gpu=False,
        )
        trained_state_dict = copy_module_state_dict(mock_contrastive_vi_model.module)
        for param_key in mock_contrastive_vi_model.module.state_dict().keys():
            is_library_param = (
                param_key == "library_log_means" or param_key == "library_log_vars"
            )
            is_px_r_decoder_param = "px_r_decoder" in param_key
            is_l_encoder_param = "l_encoder" in param_key

            if (
                is_library_param
                or is_px_r_decoder_param
                or (
                    is_l_encoder_param
                    and mock_contrastive_vi_model.module.use_observed_lib_size
                )
            ):
                # There are three cases where parameters are not updated.
                # 1. Library means and vars are derived from input data and should
                # not be updated.
                # 2. In ContrastiveVIModel, dispersion is assumed to be gene-dependent
                # but not cell-dependent, so parameters in the dispersion (px_r)
                # decoder are not used and should not be updated.
                # 3. When observed library size is used, the library encoder is not
                # used and its parameters not updated.
                assert torch.equal(
                    init_state_dict[param_key], trained_state_dict[param_key]
                )
            else:
                # Other parameters should be updated after training.
                assert not torch.equal(
                    init_state_dict[param_key], trained_state_dict[param_key]
                )

    @pytest.mark.parametrize("representation_kind", ["background", "salient"])
    def test_get_latent_representation(
        self, mock_contrastive_vi_model, representation_kind
    ):
        n_cells = mock_contrastive_vi_model.adata.n_obs
        if representation_kind == "background":
            n_latent = mock_contrastive_vi_model.module.n_background_latent
        else:
            n_latent = mock_contrastive_vi_model.module.n_salient_latent
        representation = mock_contrastive_vi_model.get_latent_representation(
            representation_kind=representation_kind
        )
        assert representation.shape == (n_cells, n_latent)

    @pytest.mark.parametrize("representation_kind", ["background", "salient"])
    def test_get_normalized_expression(
        self, mock_contrastive_vi_model, representation_kind
    ):
        n_samples = 50
        n_cells = mock_contrastive_vi_model.adata.n_obs
        n_genes = mock_contrastive_vi_model.adata.n_vars
        one_sample_exprs = mock_contrastive_vi_model.get_normalized_expression(
            n_samples=1, return_numpy=True
        )
        one_sample_exprs = one_sample_exprs[representation_kind]
        assert type(one_sample_exprs) == np.ndarray
        assert one_sample_exprs.shape == (n_cells, n_genes)

        many_sample_exprs = mock_contrastive_vi_model.get_normalized_expression(
            n_samples=n_samples,
            return_mean=False,
        )
        many_sample_exprs = many_sample_exprs[representation_kind]
        assert type(many_sample_exprs) == np.ndarray
        assert many_sample_exprs.shape == (n_samples, n_cells, n_genes)

        exprs_df = mock_contrastive_vi_model.get_normalized_expression(
            n_samples=1,
            return_numpy=False,
        )
        exprs_df = exprs_df[representation_kind]
        assert type(exprs_df) == pd.DataFrame
        assert exprs_df.shape == (n_cells, n_genes)

    def test_get_salient_normalized_expression(self, mock_contrastive_vi_model):
        n_samples = 50
        n_cells = mock_contrastive_vi_model.adata.n_obs
        n_genes = mock_contrastive_vi_model.adata.n_vars

        one_sample_expr = mock_contrastive_vi_model.get_salient_normalized_expression(
            n_samples=1, return_numpy=True
        )
        assert type(one_sample_expr) == np.ndarray
        assert one_sample_expr.shape == (n_cells, n_genes)

        many_sample_expr = mock_contrastive_vi_model.get_salient_normalized_expression(
            n_samples=n_samples,
            return_mean=False,
        )
        assert type(many_sample_expr) == np.ndarray
        assert many_sample_expr.shape == (n_samples, n_cells, n_genes)

        expr_df = mock_contrastive_vi_model.get_salient_normalized_expression(
            n_samples=1,
            return_numpy=False,
        )
        assert type(expr_df) == pd.DataFrame
        assert expr_df.shape == (n_cells, n_genes)

    def test_get_normalized_expression_fold_change(self, mock_contrastive_vi_model):
        n_samples = 50
        n_cells = mock_contrastive_vi_model.adata.n_obs
        n_genes = mock_contrastive_vi_model.adata.n_vars
        one_sample_fc = mock_contrastive_vi_model.get_normalized_expression_fold_change(
            n_samples=1
        )
        assert one_sample_fc.shape == (n_cells, n_genes)
        many_sample_fc = (
            mock_contrastive_vi_model.get_normalized_expression_fold_change(
                n_samples=50
            )
        )
        assert many_sample_fc.shape == (n_samples, n_cells, n_genes)

    def test_differential_expression(self, mock_contrastive_vi_model):
        de_df = mock_contrastive_vi_model.differential_expression(
            groupby="labels",
            group1=["label_0"],
        )
        n_vars = mock_contrastive_vi_model.adata.n_vars
        assert type(de_df) == pd.DataFrame
        assert de_df.shape[0] == n_vars
