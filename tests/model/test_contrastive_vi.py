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
            n_latent=4,
            n_layers=2,
            use_observed_lib_size=True,
        )
    else:
        return ContrastiveVIModel(
            mock_adata,
            n_hidden=16,
            n_latent=4,
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
