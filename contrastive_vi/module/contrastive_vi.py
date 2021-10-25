"""PyTorch module for Contrastive VI for single cell expression data."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scvi import _CONSTANTS
from scvi.module.base import BaseModuleClass, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, one_hot
from torch.distributions import Normal

torch.backends.cudnn.benchmark = True


class ContrastiveVIModule(BaseModuleClass):
    """
    PyTorch module for Contrastive VI (Variational Inference).

    Args:
    ----
        n_input: Number of input genes.
        n_batch: Number of batches. If 0, no batch effect correction is performed.
        n_hidden: Number of nodes per hidden layer.
        n_latent: Dimensionality of the latent space.
        n_layers: Number of hidden layers used for encoder and decoder NNs.
        dropout_rate: Dropout rate for neural networks.
        use_observed_lib_size: Use observed library size for RNA as scaling factor in
            mean of conditional distribution.
        library_log_means: 1 x n_batch array of means of the log library sizes.
            Parameterize prior on library size if not using observed library size.
        library_log_vars: 1 x n_batch array of variances of the log library sizes.
            Parameterize prior on library size if not using observed library size.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.latent_distribution = "normal"
        self.dispersion = "gene"
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        self.use_observed_lib_size = use_observed_lib_size

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )
            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        cat_list = [n_batch]
        # Background encoder.
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )
        # Salient encoder.
        self.s_encoder = Encoder(
            n_input,
            n_latent,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )
        # Library size encoder.
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )
        # Decoder from latent variable to distribution parameters in data space.
        n_input_decoder = n_latent * 2
        self.decoder = DecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
        )

    @auto_move_data
    def _compute_local_library_params(
        self, batch_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars

    @staticmethod
    def _get_inference_input_from_concat_tensors(
        concat_tensors: Tuple[Dict[str, torch.Tensor]], index: int
    ) -> Dict[str, torch.Tensor]:
        tensors = concat_tensors[index]
        x = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        input_dict = dict(x=x, batch_index=batch_index)
        return input_dict

    def _get_inference_input(
        self, concat_tensors: Tuple[Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        background = self._get_inference_input_from_concat_tensors(concat_tensors, 0)
        target = self._get_inference_input_from_concat_tensors(concat_tensors, 1)
        return dict(background=background, target=target)

    @staticmethod
    def _get_generative_input_from_concat_tensors(
        concat_tensors: Tuple[Dict[str, torch.Tensor]], index: int
    ) -> Dict[str, torch.Tensor]:
        tensors = concat_tensors[index]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        input_dict = dict(batch_index=batch_index)
        return input_dict

    @staticmethod
    def _get_generative_input_from_inference_outputs(
        inference_outputs: Dict[str, Dict[str, torch.Tensor]], data_source: str
    ) -> Dict[str, torch.Tensor]:
        z = inference_outputs[data_source]["z"]
        s = inference_outputs[data_source]["s"]
        library = inference_outputs[data_source]["library"]
        return dict(z=z, s=s, library=library)

    def _get_generative_input(
        self,
        concat_tensors: Tuple[Dict[str, torch.Tensor]],
        inference_outputs: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        background_tensor_input = self._get_generative_input_from_concat_tensors(
            concat_tensors, 0
        )
        background_inference_outputs = (
            self._get_generative_input_from_inference_outputs(
                inference_outputs, "background"
            )
        )
        target_tensor_input = self._get_generative_input_from_concat_tensors(
            concat_tensors, 1
        )
        target_inference_outputs = self._get_generative_input_from_inference_outputs(
            inference_outputs, "target"
        )
        background = {**background_tensor_input, **background_inference_outputs}
        target = {**target_tensor_input, **target_inference_outputs}
        return dict(background=background, target=target)

    @staticmethod
    def _reshape_tensor_for_samples(tensor: torch.Tensor, n_samples: int):
        return tensor.unsqueeze(0).expand((n_samples, tensor.size(0), tensor.size(1)))

    @auto_move_data
    def _generic_inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        n_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        x_ = torch.log(1 + x_)

        qz_m, qz_v, z = self.z_encoder(x_, batch_index)
        qs_m, qs_v, s = self.s_encoder(x_, batch_index)

        ql_m, ql_v = None, None
        if not self.use_observed_lib_size:
            ql_m, ql_v, library_encoded = self.l_encoder(x_, batch_index)
            library = library_encoded

        if n_samples > 1:
            qz_m = self._reshape_tensor_for_samples(qz_m, n_samples)
            qz_v = self._reshape_tensor_for_samples(qz_v, n_samples)
            z = self._reshape_tensor_for_samples(z, n_samples)
            qs_m = self._reshape_tensor_for_samples(qs_m, n_samples)
            qs_v = self._reshape_tensor_for_samples(qs_v, n_samples)
            s = self._reshape_tensor_for_samples(s, n_samples)

            if self.use_observed_lib_size:
                library = self._reshape_tensor_for_samples(library, n_samples)
            else:
                ql_m = self._reshape_tensor_for_samples(ql_m, n_samples)
                ql_v = self._reshape_tensor_for_samples(ql_v, n_samples)
                library = Normal(ql_m, ql_v.sqrt()).sample()

        outputs = dict(
            z=z,
            qz_m=qz_m,
            qz_v=qz_v,
            s=s,
            qs_m=qs_m,
            qs_v=qs_v,
            library=library,
            ql_m=ql_m,
            ql_v=ql_v,
        )
        return outputs

    @auto_move_data
    def inference(
        self,
        background: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        n_samples: int = 1,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        background_outputs = self._generic_inference(**background, n_samples=n_samples)
        background_outputs["s"] = torch.zeros_like(background_outputs["s"])
        target_outputs = self._generic_inference(**target, n_samples=n_samples)
        return dict(background=background_outputs, target=target_outputs)

    @auto_move_data
    def _generic_generative(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        latent = torch.cat([z, s], dim=-1)
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            latent,
            library,
            batch_index,
        )
        px_r = torch.exp(self.px_r)
        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )

    @auto_move_data
    def generative(
        self,
        background: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        background_outputs = self._generic_generative(**background)
        target_outputs = self._generic_generative(**target)
        return dict(background=background_outputs, target=target_outputs)
