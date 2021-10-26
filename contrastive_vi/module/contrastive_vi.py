"""PyTorch module for Contrastive VI for single cell expression data."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scvi import _CONSTANTS
from scvi.distributions import ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, one_hot
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

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
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        background_outputs = self._generic_generative(**background)
        target_outputs = self._generic_generative(**target)
        return dict(background=background_outputs, target=target_outputs)

    @staticmethod
    def reconstruction_loss(
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute likelihood loss for zero-inflated negative binomial distribution.

        Args:
        ----
            x: Input data.
            px_rate: Mean of distribution.
            px_r: Inverse dispersion.
            px_dropout: Logits scale of zero inflation probability.

        Returns
        -------
            Negative log likelihood (reconstruction loss) for each data point. If number
            of latent samples == 1, the tensor has shape `(batch_size, )`. If number
            of latent samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        recon_loss = (
            -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            .log_prob(x)
            .sum(dim=-1)
        )
        return recon_loss

    @staticmethod
    def latent_kl_divergence(
        variational_mean: torch.Tensor,
        variational_var: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between a variational posterior and prior Gaussian.
        Args:
        ----
            variational_mean: Mean of the variational posterior Gaussian.
            variational_var: Variance of the variational posterior Gaussian.
            prior_mean: Mean of the prior Gaussian.
            prior_var: Variance of the prior Gaussian.

        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        return kl(
            Normal(variational_mean, variational_var.sqrt()),
            Normal(prior_mean, prior_var.sqrt()),
        ).sum(dim=-1)

    def library_kl_divergence(
        self,
        batch_index: torch.Tensor,
        variational_library_mean: torch.Tensor,
        variational_library_var: torch.Tensor,
        library: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between library size variational posterior and prior.

        Both the variational posterior and prior are Log-Normal.
        Args:
        ----
            batch_index: Batch indices for batch-specific library size mean and
                variance.
            variational_library_mean: Mean of variational Log-Normal.
            variational_library_var: Variance of variational Log-Normal.
            library: Sampled library size.

        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        if not self.use_observed_lib_size:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_library = kl(
                Normal(variational_library_mean, variational_library_var.sqrt()),
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            )
        else:
            kl_library = torch.zeros_like(library)
        return kl_library.sum(dim=-1)

    def _generic_loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[str, torch.Tensor],
        generative_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        x = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        qs_m = inference_outputs["qs_m"]
        qs_v = inference_outputs["qs_v"]
        library = inference_outputs["library"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        prior_z_m = torch.zeros_like(qz_m)
        prior_z_v = torch.ones_like(qz_v)
        prior_s_m = torch.zeros_like(qs_m)
        prior_s_v = torch.ones_like(qs_v)

        recon_loss = self.reconstruction_loss(x, px_rate, px_r, px_dropout)
        kl_z = self.latent_kl_divergence(qz_m, qz_v, prior_z_m, prior_z_v)
        kl_s = self.latent_kl_divergence(qs_m, qs_v, prior_s_m, prior_s_v)
        kl_library = self.library_kl_divergence(batch_index, ql_m, ql_v, library)
        return dict(
            recon_loss=recon_loss,
            kl_z=kl_z,
            kl_s=kl_s,
            kl_library=kl_library,
        )

    def loss(
        self,
        concat_tensors: Tuple[Dict[str, torch.Tensor]],
        inference_outputs: Dict[str, Dict[str, torch.Tensor]],
        generative_outputs: Dict[str, Dict[str, torch.Tensor]],
        kl_weight: float = 1.0,
    ) -> LossRecorder:
        """
        Compute loss terms for contrastive-VI.
        Args:
        ----
            concat_tensors: Tuple of data mini-batch. The first element contains
                background data mini-batch. The second element contains target data
                mini-batch.
            inference_outputs: Dictionary of inference step outputs. The keys
                are "background" and "target" for the corresponding outputs.
            generative_outputs: Dictionary of generative step outputs. The keys
                are "background" and "target" for the corresponding outputs.
            kl_weight: Importance weight for KL divergence of background and salient
                latent variables, relative to KL divergence of library size.

        Returns
        -------
            An scvi.module.base.LossRecorder instance that records the following:
            loss: One-dimensional tensor for overall loss used for optimization.
            reconstruction_loss: Reconstruction loss with shape
                `(n_samples, batch_size)` if number of latent samples > 1, or
                `(batch_size, )` if number of latent samples == 1.
            kl_local: KL divergence term with shape
                `(n_samples, batch_size)` if number of latent samples > 1, or
                `(batch_size, )` if number of latent samples == 1.
            kl_global: One-dimensional tensor for global KL divergence term.
        """
        background_tensors = concat_tensors[0]
        target_tensors = concat_tensors[1]
        background_losses = self._generic_loss(
            background_tensors,
            inference_outputs["background"],
            generative_outputs["background"],
        )
        target_losses = self._generic_loss(
            target_tensors,
            inference_outputs["target"],
            generative_outputs["target"],
        )
        recon_loss = background_losses["recon_loss"] + target_losses["recon_loss"]
        kl_z = background_losses["kl_z"] + target_losses["kl_z"]
        kl_s = target_losses["kl_s"]
        kl_library = background_losses["kl_library"] + target_losses["kl_library"]

        kl_local_for_warmup = kl_z + kl_s
        kl_local_no_warmup = kl_library

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup
        loss = torch.mean(recon_loss + weighted_kl_local)
        kl_local = dict(
            kl_z=kl_z,
            kl_s=kl_s,
            kl_library=kl_library,
        )
        kl_global = torch.tensor(0.0)
        return LossRecorder(loss, recon_loss, kl_local, kl_global)

    @torch.no_grad()
    def sample(self):
        raise NotImplementedError

    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self):
        raise NotImplementedError
