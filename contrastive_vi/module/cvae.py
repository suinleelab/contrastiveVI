"""PyTorch module for Contrastive VI for single cell expression data."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi import _CONSTANTS
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import Decoder, Encoder
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from math import floor, ceil

torch.backends.cudnn.benchmark = True


class CVAEModule(BaseModuleClass):
    """
    PyTorch module for cVAE.

    Args:
    ----
        n_input: Number of input genes.
        n_batch: Number of batches. If 0, no batch effect correction is performed.
        n_hidden: Number of nodes per hidden layer.
        n_background_latent: Dimensionality of the background latent space.
        n_salient_latent: Dimensionality of the salient latent space.
        n_layers: Number of hidden layers used for encoder and decoder NNs.
        dropout_rate: Dropout rate for neural networks.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_background_latent: int = 10,
        n_salient_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.n_background_latent = n_background_latent
        self.n_salient_latent = n_salient_latent
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.latent_distribution = "normal"

        cat_list = [n_batch]
        # Background encoder.
        self.z_encoder = Encoder(
            n_input,
            n_background_latent,
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
            n_salient_latent,
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


        # Decoder from latent variable to distribution parameters in data space.
        total_latent_size = n_background_latent + n_salient_latent
        self.discriminator = nn.Linear(total_latent_size, 1)

        self.decoder = Decoder(
            total_latent_size,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
        )

    @staticmethod
    def _get_min_batch_size(concat_tensors: Dict[str, Dict[str, torch.Tensor]]) -> int:
        return min(
            concat_tensors["background"][_CONSTANTS.X_KEY].shape[0],
            concat_tensors["target"][_CONSTANTS.X_KEY].shape[0],
        )

    @staticmethod
    def _reduce_tensors_to_min_batch_size(
        tensors: Dict[str, torch.Tensor], min_batch_size: int
    ) -> None:
        for name, tensor in tensors.items():
            tensors[name] = tensor[:min_batch_size, :]

    @staticmethod
    def _get_inference_input_from_concat_tensors(
        concat_tensors: Dict[str, Dict[str, torch.Tensor]], index: str
    ) -> Dict[str, torch.Tensor]:
        tensors = concat_tensors[index]
        x = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        input_dict = dict(x=x, batch_index=batch_index)
        return input_dict

    def _get_inference_input(
        self, concat_tensors: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        background = self._get_inference_input_from_concat_tensors(concat_tensors, "background")
        target = self._get_inference_input_from_concat_tensors(concat_tensors, "target")
        # Ensure batch sizes are the same.
        min_batch_size = self._get_min_batch_size(concat_tensors)
        self._reduce_tensors_to_min_batch_size(background, min_batch_size)
        self._reduce_tensors_to_min_batch_size(target, min_batch_size)
        return dict(background=background, target=target)

    @staticmethod
    def _get_generative_input_from_concat_tensors(
        concat_tensors: Dict[str, Dict[str, torch.Tensor]], index: str
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
        return dict(z=z, s=s)

    def _get_generative_input(
        self,
        concat_tensors: Dict[str, Dict[str, torch.Tensor]],
        inference_outputs: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        background_tensor_input = self._get_generative_input_from_concat_tensors(
            concat_tensors, "background"
        )
        target_tensor_input = self._get_generative_input_from_concat_tensors(
            concat_tensors, "target"
        )
        # Ensure batch sizes are the same.
        min_batch_size = self._get_min_batch_size(concat_tensors)
        self._reduce_tensors_to_min_batch_size(background_tensor_input, min_batch_size)
        self._reduce_tensors_to_min_batch_size(target_tensor_input, min_batch_size)

        background_inference_outputs = (
            self._get_generative_input_from_inference_outputs(
                inference_outputs, "background"
            )
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

        qz_m, qz_v, z = self.z_encoder(x_, batch_index)
        qs_m, qs_v, s = self.s_encoder(x_, batch_index)

        if n_samples > 1:
            qz_m = self._reshape_tensor_for_samples(qz_m, n_samples)
            qz_v = self._reshape_tensor_for_samples(qz_v, n_samples)
            z = self._reshape_tensor_for_samples(z, n_samples)
            qs_m = self._reshape_tensor_for_samples(qs_m, n_samples)
            qs_v = self._reshape_tensor_for_samples(qs_v, n_samples)
            s = self._reshape_tensor_for_samples(s, n_samples)

        outputs = dict(
            z=z,
            qz_m=qz_m,
            qz_v=qz_v,
            s=s,
            qs_m=qs_m,
            qs_v=qs_v,
        )
        return outputs

    @auto_move_data
    def inference(
        self,
        background: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        n_samples: int = 1,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        background_batch_size = background["x"].shape[0]
        target_batch_size = target["x"].shape[0]
        inference_input = {}
        for key in background.keys():
            inference_input[key] = torch.cat([background[key], target[key]], dim=0)
        outputs = self._generic_inference(**inference_input, n_samples=n_samples)
        batch_size_dim = 0 if n_samples == 1 else 1
        background_outputs, target_outputs = {}, {}
        for key in outputs.keys():
            if outputs[key] is not None:
                background_tensor, target_tensor = torch.split(
                    outputs[key],
                    [background_batch_size, target_batch_size],
                    dim=batch_size_dim,
                )
            else:
                background_tensor, target_tensor = None, None
            background_outputs[key] = background_tensor
            target_outputs[key] = target_tensor
        background_outputs["s"] = torch.zeros_like(background_outputs["s"])
        return dict(background=background_outputs, target=target_outputs)

    @auto_move_data
    def _generic_generative(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        latent = torch.cat([z, s], dim=-1)
        p_m, p_v = self.decoder(
            latent,
            batch_index,
        )
        return dict(
            p_m=p_m, p_v=p_v
        )

    @auto_move_data
    def generative(
        self,
        background: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        latent_z_shape = background["z"].shape
        batch_size_dim = 0 if len(latent_z_shape) == 2 else 1
        background_batch_size = background["z"].shape[batch_size_dim]
        target_batch_size = target["z"].shape[batch_size_dim]
        generative_input = {}
        for key in ["z", "s"]:
            generative_input[key] = torch.cat(
                [background[key], target[key]], dim=batch_size_dim
            )
        generative_input["batch_index"] = torch.cat(
            [background["batch_index"], target["batch_index"]], dim=0
        )
        outputs = self._generic_generative(**generative_input)
        background_outputs, target_outputs = {}, {}
        for key in ["p_m", "p_v"]:
            background_tensor, target_tensor = torch.split(
                outputs[key],
                [background_batch_size, target_batch_size],
                dim=batch_size_dim,
            )
            background_outputs[key] = background_tensor
            target_outputs[key] = target_tensor
        return dict(background=background_outputs, target=target_outputs)

    @staticmethod
    def reconstruction_loss(
        x: torch.Tensor,
        p_m: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute likelihood loss for Gaussian distribution.

        Args:
        ----
            x: Input data.
            p_m: Reconstructed data from decoder network

        Returns
        -------
            Negative log likelihood (reconstruction loss) for each data point. If number
            of latent samples == 1, the tensor has shape `(batch_size, )`. If number
            of latent samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        recon_loss = F.mse_loss(p_m, x, reduction='none').sum(dim=-1)
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
        p_m = generative_outputs["p_m"]

        prior_z_m = torch.zeros_like(qz_m)
        prior_z_v = torch.ones_like(qz_v)
        prior_s_m = torch.zeros_like(qs_m)
        prior_s_v = torch.ones_like(qs_v)

        recon_loss = self.reconstruction_loss(x, p_m)
        kl_z = self.latent_kl_divergence(qz_m, qz_v, prior_z_m, prior_z_v)
        kl_s = self.latent_kl_divergence(qs_m, qs_v, prior_s_m, prior_s_v)
        return dict(
            recon_loss=recon_loss,
            kl_z=kl_z,
            kl_s=kl_s,
        )

    def loss(
        self,
        concat_tensors: Dict[str, Dict[str, torch.Tensor]],
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
        background_tensors = concat_tensors["background"]
        target_tensors = concat_tensors["target"]
        # Ensure batch sizes are the same.
        min_batch_size = self._get_min_batch_size(concat_tensors)
        self._reduce_tensors_to_min_batch_size(background_tensors, min_batch_size)
        self._reduce_tensors_to_min_batch_size(target_tensors, min_batch_size)

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

        z_tar = inference_outputs["target"]["qz_m"]
        s_tar = inference_outputs["target"]["qs_m"]

        if len(z_tar.shape) == 3:
            z_tar = z_tar.permute(1, 0, 2)  # Make mini-batch the zeroth index
            s_tar = s_tar.permute(1, 0, 2)

        z1, z2 = torch.chunk(z_tar, 2)
        s1, s2 = torch.chunk(s_tar, 2)

        # Need to make sure that all tensors have same number of samples.
        # (This is necessary e.g. if we have an odd batch size at the end of an epoch)
        size = min(len(z1), len(z2))
        z1, z2, s1, s2 = z1[:size], z2[:size], s1[:size], s2[:size]

        q = torch.cat([torch.cat([z1, s1], dim=-1), torch.cat([z2, s2], dim=-1)])
        q_bar = torch.cat(
            [torch.cat([z1, s2], dim=-1), torch.cat([z2, s1], dim=-1)]
        )

        q_bar_score = F.sigmoid(self.discriminator(q_bar))
        q_score = F.sigmoid(self.discriminator(q))

        tc_loss = torch.log(q_score / (1 - q_score))
        discriminator_loss = (- torch.log(q_score) - torch.log(1 - q_bar_score))

        loss = (
            torch.sum(recon_loss)
            + torch.sum(kl_z)
            + torch.sum(kl_s)
            + torch.sum(tc_loss)
            + torch.sum(discriminator_loss)
        )
        kl_local = dict(
            kl_z=kl_z,
            kl_s=kl_s,
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
