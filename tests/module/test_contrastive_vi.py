import pytest
import torch

from contrastive_vi.module.contrastive_vi import ContrastiveVIModule

required_data_sources = ["background", "target"]
required_inference_input_keys = ["x", "batch_index"]
required_inference_output_keys = [
    "z",
    "qz_m",
    "qz_v",
    "s",
    "qs_m",
    "qs_v",
    "library",
    "ql_m",
    "ql_v",
]
required_generative_input_keys_from_concat_tensors = ["batch_index"]
required_generative_input_keys_from_inference_outputs = ["z", "s", "library"]
required_generative_output_keys = [
    "px_scale",
    "px_r",
    "px_rate",
    "px_dropout",
]


@pytest.fixture(
    params=[True, False], ids=["with_observed_lib_size", "without_observed_lib_size"]
)
def mock_contrastive_vi_module(
    mock_n_input, mock_n_batch, mock_library_log_means, mock_library_log_vars, request
):
    if request.param:
        return ContrastiveVIModule(
            n_input=mock_n_input,
            n_batch=mock_n_batch,
            n_hidden=10,
            n_latent=4,
            n_layers=2,
            use_observed_lib_size=True,
            library_log_means=None,
            library_log_vars=None,
        )
    else:
        return ContrastiveVIModule(
            n_input=mock_n_input,
            n_batch=mock_n_batch,
            n_hidden=10,
            n_latent=4,
            n_layers=2,
            use_observed_lib_size=False,
            library_log_means=mock_library_log_means,
            library_log_vars=mock_library_log_vars,
        )


class TestContrastiveVIModule:
    def test_get_inference_input_from_concat_tensors(
        self,
        mock_contrastive_vi_module,
        mock_contrastive_batch,
        mock_n_input,
    ):
        inference_input = (
            mock_contrastive_vi_module._get_inference_input_from_concat_tensors(
                mock_contrastive_batch, 0
            )
        )
        for key in required_inference_input_keys:
            assert key in inference_input.keys()
        x = inference_input["x"]
        batch_index = inference_input["batch_index"]
        batch_size = x.shape[0]
        assert x.shape == (batch_size, mock_n_input)
        assert batch_index.shape == (batch_size, 1)

    def test_get_inference_input(
        self,
        mock_contrastive_vi_module,
        mock_contrastive_batch,
        mock_adata_background_label,
        mock_adata_target_label,
    ):
        inference_input = mock_contrastive_vi_module._get_inference_input(
            mock_contrastive_batch
        )
        for data_source in required_data_sources:
            assert data_source in inference_input.keys()

        background_input = inference_input["background"]
        background_input_keys = background_input.keys()
        target_input = inference_input["target"]
        target_input_keys = target_input.keys()

        for key in required_inference_input_keys:
            assert key in background_input_keys
            assert key in target_input_keys

        # Check background vs. target labels are consistent.
        assert (
            background_input["batch_index"] != mock_adata_background_label
        ).sum() == 0
        assert (target_input["batch_index"] != mock_adata_target_label).sum() == 0

    def test_get_generative_input_from_concat_tensors(
        self,
        mock_contrastive_vi_module,
        mock_contrastive_batch,
        mock_n_input,
    ):
        generative_input = (
            mock_contrastive_vi_module._get_generative_input_from_concat_tensors(
                mock_contrastive_batch, 0
            )
        )
        for key in required_generative_input_keys_from_concat_tensors:
            assert key in generative_input.keys()
        batch_index = generative_input["batch_index"]
        assert batch_index.shape[1] == 1

    @pytest.mark.parametrize("n_samples", [1, 2])
    def test_generic_inference(
        self,
        mock_contrastive_vi_module,
        mock_contrastive_batch,
        n_samples,
    ):
        inference_input = (
            mock_contrastive_vi_module._get_inference_input_from_concat_tensors(
                mock_contrastive_batch, 0
            )
        )
        batch_size = inference_input["x"].shape[0]
        n_latent = mock_contrastive_vi_module.n_latent

        inference_outputs = mock_contrastive_vi_module._generic_inference(
            **inference_input, n_samples=n_samples
        )
        for key in required_inference_output_keys:
            assert key in inference_outputs.keys()

        if n_samples > 1:
            expected_latent_shape = (n_samples, batch_size, n_latent)
            expected_library_shape = (n_samples, batch_size, 1)
        else:
            expected_latent_shape = (batch_size, n_latent)
            expected_library_shape = (batch_size, 1)

        assert inference_outputs["z"].shape == expected_latent_shape
        assert inference_outputs["qz_m"].shape == expected_latent_shape
        assert inference_outputs["qz_v"].shape == expected_latent_shape
        assert inference_outputs["s"].shape == expected_latent_shape
        assert inference_outputs["qs_m"].shape == expected_latent_shape
        assert inference_outputs["qs_v"].shape == expected_latent_shape
        assert inference_outputs["library"].shape == expected_library_shape
        assert (
            inference_outputs["ql_m"] is None
            or inference_outputs["ql_m"].shape == expected_library_shape
        )
        assert (
            inference_outputs["ql_v"] is None
            or inference_outputs["ql_m"].shape == expected_library_shape
        )

    def test_inference(
        self,
        mock_contrastive_vi_module,
        mock_contrastive_batch,
    ):
        inference_input = mock_contrastive_vi_module._get_inference_input(
            mock_contrastive_batch
        )
        inference_outputs = mock_contrastive_vi_module.inference(**inference_input)
        for data_source in required_data_sources:
            assert data_source in inference_outputs.keys()
        background_s = inference_outputs["background"]["s"]

        # Background salient variables should be all zeros.
        assert torch.equal(background_s, torch.zeros_like(background_s))

    def test_get_generative_input_from_inference_outputs(
        self,
        mock_contrastive_vi_module,
        mock_contrastive_batch,
    ):
        inference_outputs = mock_contrastive_vi_module.inference(
            **mock_contrastive_vi_module._get_inference_input(mock_contrastive_batch)
        )
        generative_input = (
            mock_contrastive_vi_module._get_generative_input_from_inference_outputs(
                inference_outputs, required_data_sources[0]
            )
        )
        for key in required_generative_input_keys_from_inference_outputs:
            assert key in generative_input

        z = generative_input["z"]
        s = generative_input["s"]
        library = generative_input["library"]
        n_latent = mock_contrastive_vi_module.n_latent
        batch_size = z.shape[0]

        assert z.shape == (batch_size, n_latent)
        assert s.shape == (batch_size, n_latent)
        assert library.shape == (batch_size, 1)

    def test_get_generative_input(
        self,
        mock_contrastive_vi_module,
        mock_contrastive_batch,
        mock_adata_background_label,
        mock_adata_target_label,
    ):
        inference_outputs = mock_contrastive_vi_module.inference(
            **mock_contrastive_vi_module._get_inference_input(mock_contrastive_batch)
        )
        generative_input = mock_contrastive_vi_module._get_generative_input(
            mock_contrastive_batch, inference_outputs
        )
        for data_source in required_data_sources:
            assert data_source in generative_input.keys()
        background_generative_input = generative_input["background"]
        background_generative_input_keys = background_generative_input.keys()
        target_generative_input = generative_input["target"]
        target_generative_input_keys = target_generative_input.keys()
        for key in (
            required_generative_input_keys_from_concat_tensors
            + required_generative_input_keys_from_inference_outputs
        ):
            assert key in background_generative_input_keys
            assert key in target_generative_input_keys

        # Check background vs. target labels are consistent.
        assert (
            background_generative_input["batch_index"] != mock_adata_background_label
        ).sum() == 0
        assert (
            target_generative_input["batch_index"] != mock_adata_target_label
        ).sum() == 0

    @pytest.mark.parametrize("n_samples", [1, 2])
    def test_generic_generative(
        self,
        mock_contrastive_vi_module,
        mock_contrastive_batch,
        n_samples,
    ):
        inference_outputs = mock_contrastive_vi_module.inference(
            **mock_contrastive_vi_module._get_inference_input(mock_contrastive_batch),
            n_samples=n_samples,
        )
        generative_input = mock_contrastive_vi_module._get_generative_input(
            mock_contrastive_batch, inference_outputs
        )["background"]
        generative_outputs = mock_contrastive_vi_module._generic_generative(
            **generative_input
        )
        for key in required_generative_output_keys:
            assert key in generative_outputs.keys()
        px_scale = generative_outputs["px_scale"]
        px_r = generative_outputs["px_r"]
        px_rate = generative_outputs["px_rate"]
        px_dropout = generative_outputs["px_dropout"]
        batch_size = px_scale.shape[-2]
        n_input = mock_contrastive_vi_module.n_input

        if n_samples > 1:
            expected_shape = (n_samples, batch_size, n_input)
        else:
            expected_shape = (batch_size, n_input)

        assert px_scale.shape == expected_shape
        assert px_r.shape == (n_input,)  # One dispersion parameter per gene.
        assert px_rate.shape == expected_shape
        assert px_dropout.shape == expected_shape

    def test_generative(
        self,
        mock_contrastive_vi_module,
        mock_contrastive_batch,
    ):
        inference_outputs = mock_contrastive_vi_module.inference(
            **mock_contrastive_vi_module._get_inference_input(mock_contrastive_batch),
        )
        generative_input = mock_contrastive_vi_module._get_generative_input(
            mock_contrastive_batch, inference_outputs
        )
        generative_outputs = mock_contrastive_vi_module.generative(**generative_input)
        for data_source in required_data_sources:
            assert data_source in generative_outputs.keys()
