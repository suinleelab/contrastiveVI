"""PyTorch module for Contrastive VI for single cell expression data."""

from typing import Dict, Optional, Tuple, Union, Literal, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial, NegativeBinomialMixture
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderTOTALVI, EncoderTOTALVI, one_hot
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

torch.backends.cudnn.benchmark = True


class TotalContrastiveVIModule(BaseModuleClass):
    """
    PyTorch module for total-contrastiveVI (contrastive analysis for CITE-seq).

    Args:
    ----
        n_input_genes: Number of input genes.
        n_input_proteins: Number of input proteins.
        n_batch: Number of batches. If 0, no batch effect correction is performed.
        n_hidden: Number of nodes per hidden layer.
        n_background_latent: Dimensionality of the background latent space.
        n_salient_latent: Dimensionality of the salient latent space.
        n_layers: Number of hidden layers used for encoder and decoder NNs.
        dropout_rate: Dropout rate for neural networks.
        protein_batch_mask: Dictionary where each key is a batch code, and value is for
            each protein, whether it was observed or not.
        use_observed_lib_size: Use observed library size for RNA as scaling factor in
            mean of conditional distribution.
        library_log_means: 1 x n_batch array of means of the log library sizes.
            Parameterize prior on library size if not using observed library size.
        library_log_vars: 1 x n_batch array of variances of the log library sizes.
            Parameterize prior on library size if not using observed library size.
    """

    def __init__(
        self,
        n_input_genes: int,
        n_input_proteins: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_background_latent: int = 10,
        n_salient_latent: int = 10,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate_decoder: float = 0.2,
        dropout_rate_encoder: float = 0.2,
        gene_dispersion: str = "gene",
        protein_dispersion: str = "protein",
        log_variational: bool = True,
        gene_likelihood: str = "nb",
        latent_distribution: str = "normal",
        protein_batch_mask: Dict[Union[str, int], np.ndarray] = None,
        encode_covariates: bool = True,
        protein_background_prior_mean: Optional[np.ndarray] = None,
        protein_background_prior_scale: Optional[np.ndarray] = None,
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
    ) -> None:
        super().__init__()
        self.gene_dispersion = gene_dispersion
        self.n_background_latent = n_background_latent
        self.n_salient_latent = n_salient_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_input_genes = n_input_genes
        self.n_input_proteins = n_input_proteins
        self.protein_dispersion = protein_dispersion
        self.latent_distribution = latent_distribution
        self.protein_batch_mask = protein_batch_mask
        self.encode_covariates = encode_covariates
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_means is None:
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

        # parameters for prior on rate_back (background protein mean)
        if protein_background_prior_mean is None:
            if n_batch > 0:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(n_input_proteins, n_batch)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins, n_batch), -10, 1)
                )
            else:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(n_input_proteins)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins), -10, 1)
                )
        else:
            if protein_background_prior_mean.shape[1] == 1 and n_batch != 1:
                init_mean = protein_background_prior_mean.ravel()
                init_scale = protein_background_prior_scale.ravel()
            else:
                init_mean = protein_background_prior_mean
                init_scale = protein_background_prior_scale
            self.background_pro_alpha = torch.nn.Parameter(
                torch.from_numpy(init_mean.astype(np.float32))
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.log(torch.from_numpy(init_scale.astype(np.float32)))
            )

        if self.gene_dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))
        elif self.gene_dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_batch))
        elif self.gene_dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_labels))
        else:  # gene-cell
            pass

        if self.protein_dispersion == "protein":
            self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins))
        elif self.protein_dispersion == "protein-batch":
            self.py_r = torch.nn.Parameter(
                2 * torch.rand(self.n_input_proteins, n_batch)
            )
        elif self.protein_dispersion == "protein-label":
            self.py_r = torch.nn.Parameter(
                2 * torch.rand(self.n_input_proteins, n_labels)
            )
        else:  # protein-cell
            pass

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input = n_input_genes + self.n_input_proteins
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        self.z_encoder = EncoderTOTALVI(
            n_input_encoder,
            n_background_latent,
            n_layers=n_layers_encoder,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )

        self.s_encoder = EncoderTOTALVI(
            n_input_encoder,
            n_salient_latent,
            n_layers=n_layers_encoder,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )
        n_total_latent = n_background_latent + n_salient_latent
        self.decoder = DecoderTOTALVI(
            n_total_latent + n_continuous_cov,
            n_input_genes,
            self.n_input_proteins,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
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
    def _get_min_batch_size(concat_tensors: Dict[str, Dict[str, torch.Tensor]]) -> int:
        return min(
            concat_tensors["background"][REGISTRY_KEYS.X_KEY].shape[0],
            concat_tensors["target"][REGISTRY_KEYS.X_KEY].shape[0],
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
        x = tensors[REGISTRY_KEYS.X_KEY]
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        input_dict = dict(x=x, y=y, batch_index=batch_index)
        return input_dict

    def _get_inference_input(
        self, concat_tensors: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        background = self._get_inference_input_from_concat_tensors(
            concat_tensors, "background"
        )
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
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        input_dict = dict(batch_index=batch_index)
        return input_dict

    @staticmethod
    def _get_generative_input_from_inference_outputs(
        inference_outputs: Dict[str, Dict[str, torch.Tensor]], data_source: str
    ) -> Dict[str, torch.Tensor]:
        z = inference_outputs[data_source]["z"]
        s = inference_outputs[data_source]["s"]
        library_gene = inference_outputs[data_source]["library_gene"]
        return dict(z=z, s=s, library_gene=library_gene)

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
        y: torch.Tensor,
        batch_index: torch.Tensor,
        n_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        x_ = x
        y_ = y
        if self.use_observed_lib_size:
            library_gene = x.sum(1).unsqueeze(1)
        x_ = torch.log(1 + x_)
        y_ = torch.log(1 + y_)
        encoder_input = torch.cat((x_, y_), dim=-1)

        (
            qz_m,
            qz_v,
            ql_m,
            ql_v,
            background_latent,
            untran_background_latent,
        ) = self.z_encoder(encoder_input, batch_index)
        z = background_latent["z"]
        untran_z = untran_background_latent["z"]
        untran_l = untran_background_latent["l"]  # Library encoder used and updated.
        (qs_m, qs_v, _, _, salient_latent, untran_salient_latent) = self.s_encoder(
            encoder_input, batch_index
        )
        s = salient_latent["z"]
        untran_s = untran_salient_latent["z"]
        # Library encoder not used and not updated.

        if not self.use_observed_lib_size:
            library_gene = background_latent["l"]

        if n_samples > 1:
            qz_m = self._reshape_tensor_for_samples(qz_m, n_samples)
            qz_v = self._reshape_tensor_for_samples(qz_v, n_samples)
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

            qs_m = self._reshape_tensor_for_samples(qs_m, n_samples)
            qs_v = self._reshape_tensor_for_samples(qs_v, n_samples)
            untran_s = Normal(qs_m, qs_v.sqrt()).sample()
            s = self.s_encoder.z_transformation(untran_s)

            ql_m = self._reshape_tensor_for_samples(ql_m, n_samples)
            ql_v = self._reshape_tensor_for_samples(ql_v, n_samples)
            untran_l = Normal(ql_m, ql_v.sqrt()).sample()

            if self.use_observed_lib_size:
                library_gene = self._reshape_tensor_for_samples(library_gene, n_samples)
            else:
                library_gene = self.z_encoder.l_transformation(untran_l)

        if self.n_batch > 0:
            py_back_alpha_prior = F.linear(
                one_hot(batch_index, self.n_batch), self.background_pro_alpha
            )
            py_back_beta_prior = F.linear(
                one_hot(batch_index, self.n_batch),
                torch.exp(self.background_pro_log_beta),
            )
        else:
            py_back_alpha_prior = self.background_pro_alpha
            py_back_beta_prior = torch.exp(self.background_pro_log_beta)

        back_mean_prior = Normal(py_back_alpha_prior, py_back_beta_prior)

        outputs = dict(
            untran_z=untran_z,
            z=z,
            qz_m=qz_m,
            qz_v=qz_v,
            untran_s=untran_s,
            s=s,
            qs_m=qs_m,
            qs_v=qs_v,
            library_gene=library_gene,
            ql_m=ql_m,
            ql_v=ql_v,
            untran_l=untran_l,
            back_mean_prior=back_mean_prior
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
        target_outputs = self._generic_inference(**target, n_samples=n_samples)
        background_outputs["s"] = torch.zeros_like(background_outputs["s"])
        return dict(background=background_outputs, target=target_outputs)

    @auto_move_data
    def _generic_generative(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        library_gene: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        latent = torch.cat([z, s], dim=-1)
        px_, py_, log_pro_back_mean = self.decoder(
            latent,
            library_gene,
            batch_index,
        )
        px_r = torch.exp(self.px_r)
        py_r = torch.exp(self.py_r)
        px_["r"] = px_r
        py_["r"] = py_r
        return dict(px_=px_, py_=py_, log_pro_back_mean=log_pro_back_mean)

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
        for key in ["z", "s", "library_gene"]:
            generative_input[key] = torch.cat(
                [background[key], target[key]], dim=batch_size_dim
            )
        generative_input["batch_index"] = torch.cat(
            [background["batch_index"], target["batch_index"]], dim=0
        )
        outputs = self._generic_generative(**generative_input)

        # Split outputs into corresponding background and target set.
        background_outputs = {"px_": {}, "py_": {}}
        target_outputs = {"px_": {}, "py_": {}}
        for modality in ["px_", "py_"]:
            for key in outputs[modality].keys():
                if key == "r":
                    background_tensor = outputs[modality][key]
                    target_tensor = outputs[modality][key]
                else:
                    if outputs[modality][key] is not None:
                        background_tensor, target_tensor = torch.split(
                            outputs[modality][key],
                            [background_batch_size, target_batch_size],
                            dim=batch_size_dim,
                        )
                    else:
                        background_tensor, target_tensor = None, None
                background_outputs[modality][key] = background_tensor
                target_outputs[modality][key] = target_tensor

        if outputs["log_pro_back_mean"] is not None:
            background_tensor, target_tensor = torch.split(
                outputs["log_pro_back_mean"],
                [background_batch_size, target_batch_size],
                dim=batch_size_dim,
            )
        else:
            background_tensor, target_tensor = None, None
        background_outputs["log_pro_back_mean"] = background_tensor
        target_outputs["log_pro_back_mean"] = target_tensor

        return dict(background=background_outputs, target=target_outputs)

    @staticmethod
    def get_reconstruction_loss(
        x: torch.Tensor,
        y: torch.Tensor,
        px_dict: Dict[str, torch.Tensor],
        py_dict: Dict[str, torch.Tensor],
        pro_batch_mask_minibatch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss."""
        px_ = px_dict
        py_ = py_dict

        reconst_loss_gene = (
            -NegativeBinomial(mu=px_["rate"], theta=px_["r"]).log_prob(x).sum(dim=-1)
        )

        py_conditional = NegativeBinomialMixture(
            mu1=py_["rate_back"],
            mu2=py_["rate_fore"],
            theta1=py_["r"],
            mixture_logits=py_["mixing"],
        )
        reconst_loss_protein_full = -py_conditional.log_prob(y)
        if pro_batch_mask_minibatch is not None:
            temp_pro_loss_full = torch.zeros_like(reconst_loss_protein_full)
            temp_pro_loss_full.masked_scatter_(
                pro_batch_mask_minibatch.bool(), reconst_loss_protein_full
            )

            reconst_loss_protein = temp_pro_loss_full.sum(dim=-1)
        else:
            reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_gene, reconst_loss_protein

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

    def library_gene_kl_divergence(
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
        x = tensors[REGISTRY_KEYS.X_KEY]
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        qs_m = inference_outputs["qs_m"]
        qs_v = inference_outputs["qs_v"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        library_gene = inference_outputs["library_gene"]
        px_ = generative_outputs["px_"]
        py_ = generative_outputs["py_"]

        prior_z_m = torch.zeros_like(qz_m)
        prior_z_v = torch.ones_like(qz_v)
        prior_s_m = torch.zeros_like(qs_m)
        prior_s_v = torch.ones_like(qs_v)

        if self.protein_batch_mask is not None:
            pro_batch_mask_minibatch = torch.zeros_like(y)
            for b in torch.unique(batch_index):
                b_indices = (batch_index == b).reshape(-1)
                pro_batch_mask_minibatch[b_indices] = torch.tensor(
                    self.protein_batch_mask[b.item()].astype(np.float32),
                    device=y.device,
                )
        else:
            pro_batch_mask_minibatch = None
        reconst_loss_gene, reconst_loss_protein = self.get_reconstruction_loss(
            x, y, px_, py_, pro_batch_mask_minibatch
        )

        kl_z = self.latent_kl_divergence(qz_m, qz_v, prior_z_m, prior_z_v)
        kl_s = self.latent_kl_divergence(qs_m, qs_v, prior_s_m, prior_s_v)
        kl_library_gene = self.library_gene_kl_divergence(
            batch_index, ql_m, ql_v, library_gene
        )

        kl_div_back_pro_full = kl(
            Normal(py_["back_alpha"], py_["back_beta"]), inference_outputs["back_mean_prior"]
        )
        if pro_batch_mask_minibatch is not None:
            kl_div_back_pro = (pro_batch_mask_minibatch * kl_div_back_pro_full).sum(
                dim=-1
            )
        else:
            kl_div_back_pro = kl_div_back_pro_full.sum(dim=-1)

        return dict(
            reconst_loss_gene=reconst_loss_gene,
            reconst_loss_protein=reconst_loss_protein,
            kl_z=kl_z,
            kl_s=kl_s,
            kl_library_gene=kl_library_gene,
            kl_div_back_pro=kl_div_back_pro,
        )

    def loss(
        self,
        concat_tensors: Dict[str, Dict[str, torch.Tensor]],
        inference_outputs: Dict[str, Dict[str, torch.Tensor]],
        generative_outputs: Dict[str, Dict[str, torch.Tensor]],
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

        Returns
        -------
            An scvi.module.base.LossRecorder instance that records the losses.
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

        reconst_loss_gene = (
            background_losses["reconst_loss_gene"] + target_losses["reconst_loss_gene"]
        )
        reconst_loss_protein = (
            background_losses["reconst_loss_protein"]
            + target_losses["reconst_loss_protein"]
        )
        reconst_losses = dict(
            reconst_loss_gene=reconst_loss_gene,
            reconst_loss_protein=reconst_loss_protein,
        )

        kl_z = background_losses["kl_z"] + target_losses["kl_z"]
        kl_s = target_losses["kl_s"]
        kl_library_gene = (
            background_losses["kl_library_gene"] + target_losses["kl_library_gene"]
        )
        kl_div_back_pro = (
            background_losses["kl_div_back_pro"] + target_losses["kl_div_back_pro"]
        )

        loss = torch.mean(
            reconst_loss_gene
            + reconst_loss_protein
            + kl_z
            + kl_s
            + kl_library_gene
            + kl_div_back_pro
        )

        kl_local = dict(
            kl_z=kl_z,
            kl_s=kl_s,
            kl_library_gene=kl_library_gene,
            kl_div_back_pro=kl_div_back_pro,
        )
        kl_global = torch.tensor(0.0)
        return LossRecorder(loss, reconst_losses, kl_local, kl_global)

    @torch.no_grad()
    def sample_mean(
        self,
        tensors: Dict[str, torch.Tensor],
        data_source: str,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Sample posterior mean."""
        raise NotImplementedError

    @torch.no_grad()
    def sample(self):
        raise NotImplementedError

    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self):
        raise NotImplementedError
