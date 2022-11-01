"""Model class for contrastive-VI for single cell expression data."""

import logging
import warnings
from collections.abc import Iterable as IterableClass
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from scvi import REGISTRY_KEYS
from scvi.dataloaders import AnnDataLoader
from scvi._compat import Literal
from scvi._types import Number
from scvi._utils import _doc_params
from scvi.data import AnnDataManager
from scvi.data._utils import _check_nonnegative_integers
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ProteinObsmField,
)
from scvi.dataloaders import DataSplitter
from scvi.model._utils import (
    _get_batch_code_from_category,
    _init_library_size,
    cite_seq_raw_counts_properties,
)
from scvi.model.base._utils import _de_core
from scvi.utils._docstrings import setup_anndata_dsp

from contrastive_vi.model.base.training_mixin import ContrastiveTrainingMixin
from contrastive_vi.module.total_contrastive_vi import TotalContrastiveVIModule

from scvi.model.base import BaseModelClass

logger = logging.getLogger(__name__)
Number = Union[int, float]


class TotalContrastiveVIModel(ContrastiveTrainingMixin, BaseModelClass):
    """
    Model class for total-contrastiveVI.
    Args:
    ----
        adata: AnnData object that has been registered via
            `TotalContrastiveVIModel.setup_anndata`.
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
        empirical_protein_background_prior: Set the initialization of protein
            background prior empirically. This option fits a GMM for each of
            100 cells per batch and averages the distributions. Note that even with
            this option set to `True`, this only initializes a parameter that is
            learned during inference. If `False`, randomly initializes. The default
            (`None`), sets this to `True` if greater than 10 proteins are used.
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_background_latent: int = 10,
        n_salient_latent: int = 10,
        gene_dispersion: Literal[
            "gene", "gene-batch", "gene-label", "gene-cell"
        ] = "gene",
        protein_dispersion: Literal[
            "protein", "protein-batch", "protein-label"
        ] = "protein",
        gene_likelihood: Literal["zinb", "nb"] = "nb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        empirical_protein_background_prior: Optional[bool] = None,
        override_missing_proteins: bool = False,
        wasserstein_penalty: float = 0,
        **model_kwargs,
    ) -> None:
        super(TotalContrastiveVIModel, self).__init__(adata)

        self.protein_state_registry = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.PROTEIN_EXP_KEY
        )
        if (
            ProteinObsmField.PROTEIN_BATCH_MASK in self.protein_state_registry
            and not override_missing_proteins
        ):
            batch_mask = self.protein_state_registry.protein_batch_mask
            msg = (
                "Some proteins have all 0 counts in some batches. "
                + "These proteins will be treated as missing measurements; however, "
                + "this can occur due to experimental design/biology. "
                + "Reinitialize the model with `override_missing_proteins=True`,"
                + "to override this behavior."
            )
            warnings.warn(msg, UserWarning)
            self._use_adversarial_classifier = True
        else:
            batch_mask = None
            self._use_adversarial_classifier = False

        emp_prior = (
            empirical_protein_background_prior
            if empirical_protein_background_prior is not None
            else (self.summary_stats.n_proteins > 10)
        )
        if emp_prior:
            prior_mean, prior_scale = self._get_totalvi_protein_priors(adata)
        else:
            prior_mean, prior_scale = None, None

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)[
                CategoricalJointObsField.N_CATS_PER_KEY
            ]
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )

        n_batch = self.summary_stats.n_batch
        use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )

        self.module = TotalContrastiveVIModule(
            n_input_genes=self.summary_stats["n_vars"],
            n_input_proteins=self.summary_stats["n_proteins"],
            n_batch=n_batch,
            n_background_latent=n_background_latent,
            n_salient_latent=n_salient_latent,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            gene_dispersion=gene_dispersion,
            protein_dispersion=protein_dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            protein_batch_mask=batch_mask,
            protein_background_prior_mean=prior_mean,
            protein_background_prior_scale=prior_scale,
            use_size_factor_key=use_size_factor_key,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            wasserstein_penalty=wasserstein_penalty,
            **model_kwargs,
        )
        self._model_summary_string = "totalContrastiveVI."
        # Necessary line to get params to be used for saving and loading.
        self.init_params_ = self._get_init_params(locals())
        logger.info("The model has been initialized")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        protein_expression_obsm_key: str,
        protein_names_uns_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[AnnData]:
        """
        %(summary)s.
        Parameters
        ----------
        %(param_adata)s
        protein_expression_obsm_key
            key in `adata.obsm` for protein expression data.
        protein_names_uns_key
            key in `adata.uns` for protein names. If None, will use the column names of `adata.obsm[protein_expression_obsm_key]`
            if it is a DataFrame, else will assign sequential names to proteins.
        %(param_batch_key)s
        %(param_layer)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        %(param_copy)s
        Returns
        -------
        %(returns)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        batch_field = CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(
                REGISTRY_KEYS.LABELS_KEY, None
            ),  # Default labels field for compatibility with TOTALVAE
            batch_field,
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
            ProteinObsmField(
                REGISTRY_KEYS.PROTEIN_EXP_KEY,
                protein_expression_obsm_key,
                use_batch_mask=True,
                batch_key=batch_field.attr_key,
                colnames_uns_key=protein_names_uns_key,
                is_count_data=True,
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
        representation_kind: str = "salient",
    ) -> np.ndarray:
        """
        Return the background or salient latent representation for each cell.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        give_mean: Give mean of distribution or sample from it.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        representation_kind: Either "background" or "salient" for the corresponding
            representation kind.

        Returns
        -------
            A numpy array with shape `(n_cells, n_latent)`.
        """
        available_representation_kinds = ["background", "salient"]
        assert representation_kind in available_representation_kinds, (
            f"representation_kind = {representation_kind} is not one of"
            f" {available_representation_kinds}"
        )

        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )
        latent = []
        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            outputs = self.module._generic_inference(
                x=x, y=y, batch_index=batch_index, n_samples=1
            )

            if representation_kind == "background":
                latent_m = outputs["qz_m"]
                latent_sample = outputs["z"]
            else:
                latent_m = outputs["qs_m"]
                latent_sample = outputs["s"]

            if give_mean:
                latent_sample = latent_m

            latent += [latent_sample.detach().cpu()]
        return torch.cat(latent).numpy()

    @torch.no_grad()
    def get_normalized_expression(
        self,
        adata=None,
        indices=None,
        n_samples_overall: Optional[int] = None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        gene_list: Optional[Sequence[str]] = None,
        protein_list: Optional[Sequence[str]] = None,
        library_size: Optional[Union[float, Literal["latent"]]] = 1,
        n_samples: int = 1,
        sample_protein_mixing: bool = False,
        scale_protein: bool = False,
        include_protein_background: bool = False,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Dict[
        str, Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]
    ]:
        r"""
        Returns the normalized gene expression and protein expression.
        This is denoted as :math:`\rho_n` in the totalVI paper for genes, and TODO
        for proteins, :math:`(1-\pi_{nt})\alpha_{nt}\beta_{nt}`.
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to use in total
        transform_batch
            Batch to condition on.
            If transform_batch is:
            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        protein_list
            Return protein expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of
            relevant magnitude.
        n_samples
            Get sample scale from multiple samples.
        sample_protein_mixing
            Sample mixing bernoulli, setting background to zero
        scale_protein
            Make protein expression sum to 1
        include_protein_background
            Include background component for protein expression
        batch_size
            Minibatch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `np.ndarray` instead of a `pd.DataFrame`. Includes gene
            names as columns. If either n_samples=1 or return_mean=True, defaults to
            False. Otherwise, it defaults to True.
        Returns
        -------
        - **gene_normalized_expression** - normalized expression for RNA
        - **protein_normalized_expression** - normalized expression for proteins
        If ``n_samples`` > 1 and ``return_mean`` is False, then the shape is
        ``(samples, cells, genes)``. Otherwise, shape is ``(cells, genes)``.
        Return type is ``pd.DataFrame`` unless ``return_numpy`` is True.
        """
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        post = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]
        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = self.scvi_setup_dict_["protein_names"]
            protein_mask = [True if p in protein_list else False for p in all_proteins]
        if indices is None:
            indices = np.arange(adata.n_obs)

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is "
                    "False, returning np.ndarray"
                )
            return_numpy = True

        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        transform_batch = _get_batch_code_from_category(adata_manager, transform_batch)

        results = {}
        for expression_type in ["salient", "background"]:
            scale_list_gene = []
            scale_list_pro = []

            for tensors in post:
                x = tensors[REGISTRY_KEYS.X_KEY]
                y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
                batch_original = tensors[REGISTRY_KEYS.BATCH_KEY]
                px_scale = torch.zeros_like(x)
                py_scale = torch.zeros_like(y)
                if n_samples > 1:
                    px_scale = torch.stack(n_samples * [px_scale])
                    py_scale = torch.stack(n_samples * [py_scale])
                for b in transform_batch:
                    inference_outputs = self.module._generic_inference(
                        x=x, y=y, batch_index=batch_original, n_samples=n_samples
                    )

                    if expression_type == "salient":
                        s = inference_outputs["s"]
                    elif expression_type == "background":
                        s = torch.zeros_like(inference_outputs["s"])
                    else:
                        raise NotImplementedError("Invalid expression type provided")

                    generative_outputs = self.module._generic_generative(
                        z=inference_outputs["z"],
                        s=s,
                        library_gene=inference_outputs["library_gene"],
                        batch_index=b,
                    )

                    if library_size == "latent":
                        px_scale += generative_outputs["px_"]["rate"].cpu()
                    else:
                        px_scale += generative_outputs["px_"]["scale"].cpu()
                    px_scale = px_scale[..., gene_mask]

                    py_ = generative_outputs["py_"]
                    # probability of background
                    protein_mixing = 1 / (1 + torch.exp(-py_["mixing"].cpu()))
                    if sample_protein_mixing is True:
                        protein_mixing = torch.distributions.Bernoulli(
                            protein_mixing
                        ).sample()
                    protein_val = py_["rate_fore"].cpu() * (1 - protein_mixing)
                    if include_protein_background is True:
                        protein_val += py_["rate_back"].cpu() * protein_mixing

                    if scale_protein is True:
                        protein_val = torch.nn.functional.normalize(
                            protein_val, p=1, dim=-1
                        )
                    protein_val = protein_val[..., protein_mask]
                    py_scale += protein_val
                px_scale /= len(transform_batch)
                py_scale /= len(transform_batch)
                scale_list_gene.append(px_scale)
                scale_list_pro.append(py_scale)

            if n_samples > 1:
                # concatenate along batch dimension
                # -> result shape = (samples, cells, features)
                scale_list_gene = torch.cat(scale_list_gene, dim=1)
                scale_list_pro = torch.cat(scale_list_pro, dim=1)
                # (cells, features, samples)
                scale_list_gene = scale_list_gene.permute(1, 2, 0)
                scale_list_pro = scale_list_pro.permute(1, 2, 0)
            else:
                scale_list_gene = torch.cat(scale_list_gene, dim=0)
                scale_list_pro = torch.cat(scale_list_pro, dim=0)

            if return_mean is True and n_samples > 1:
                scale_list_gene = torch.mean(scale_list_gene, dim=-1)
                scale_list_pro = torch.mean(scale_list_pro, dim=-1)

            scale_list_gene = scale_list_gene.cpu().numpy()
            scale_list_pro = scale_list_pro.cpu().numpy()
            if return_numpy is None or return_numpy is False:
                gene_df = pd.DataFrame(
                    scale_list_gene,
                    columns=adata.var_names[gene_mask],
                    index=adata.obs_names[indices],
                )
                protein_names = self.protein_state_registry.column_names
                pro_df = pd.DataFrame(
                    scale_list_pro,
                    columns=protein_names[protein_mask],
                    index=adata.obs_names[indices],
                )

                results[expression_type] = (gene_df, pro_df)
            else:
                results[expression_type] = (scale_list_gene, scale_list_pro)
        return results

    @torch.no_grad()
    def get_specific_normalized_expression(
        self,
        adata=None,
        indices=None,
        n_samples_overall=None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        scale_protein=False,
        batch_size: Optional[int] = None,
        n_samples=1,
        sample_protein_mixing=False,
        include_protein_background=False,
        return_mean=True,
        return_numpy=True,
        expression_type: Optional[str] = None,
        indices_to_return_salient: Optional[Sequence[int]] = None,
    ):
        """
        Return normalized (decoded) gene and protein expression.

        Gene + protein expressions are decoded from either the background or salient
        latent space. One of `expression_type` or `indices_to_return_salient` should
        have an input argument.

        Args:
        ----
        adata:
            AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall: The number of random samples in `adata` to use.
        transform_batch:
            Batch to condition on.
            If transform_batch is:
            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        scale_protein: Make protein expression sum to 1
        batch_size:
            Minibatch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        sample_protein_mixing: Sample mixing bernoulli, setting background to zero
        include_protein_background: Include background component for protein expression
        return_mean: Whether to return the mean of the samples.
        return_numpy:
            Return a `np.ndarray` instead of a `pd.DataFrame`. Includes gene
            names as columns. If either n_samples=1 or return_mean=True, defaults to
            False. Otherwise, it defaults to True.
        expression_type: One of {"salient", "background"} to specify the type of
            normalized expression to return.
        indices_to_return_salient: If `indices` is a subset of
            `indices_to_return_salient`, normalized expressions derived from background
            and salient latent embeddings are returned. If `indices` is not `None` and
            is not a subset of `indices_to_return_salient`, normalized expressions
            derived only from background latent embeddings are returned.

        Returns
        -------
            If `n_samples` > 1 and `return_mean` is `False`, then the shape is
            `((samples, cells, genes), (samples, cells, proteins))`. Otherwise, shape
            is `((cells, genes), (cells, proteins))`. In this case, return type is
            Tuple[`pandas.DataFrame`] unless `return_numpy` is `True`.
        """
        is_expression_type_none = expression_type is None
        is_indices_to_return_salient_none = indices_to_return_salient is None
        if is_expression_type_none and is_indices_to_return_salient_none:
            raise ValueError(
                "Both expression_type and indices_to_return_salient are None! "
                "Exactly one of them needs to be supplied with an input argument."
            )
        elif (not is_expression_type_none) and (not is_indices_to_return_salient_none):
            raise ValueError(
                "Both expression_type and indices_to_return_salient have an input "
                "argument! Exactly one of them needs to be supplied with an input "
                "argument."
            )
        else:
            exprs = self.get_normalized_expression(
                adata=adata,
                indices=indices,
                n_samples_overall=n_samples_overall,
                transform_batch=transform_batch,
                return_numpy=return_numpy,
                return_mean=return_mean,
                n_samples=n_samples,
                batch_size=batch_size,
                scale_protein=scale_protein,
                sample_protein_mixing=sample_protein_mixing,
                include_protein_background=include_protein_background,
            )
            if not is_expression_type_none:
                return exprs[expression_type]
            else:
                if indices is None:
                    indices = np.arange(adata.n_obs)
                if set(indices).issubset(set(indices_to_return_salient)):
                    return exprs["salient"]
                else:
                    return exprs["background"]

    def _expression_for_de(
        self,
        adata=None,
        indices=None,
        n_samples_overall=None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        scale_protein=False,
        batch_size: Optional[int] = None,
        n_samples=1,
        sample_protein_mixing=False,
        include_protein_background=False,
        protein_prior_count=0.5,
        expression_type: Optional[str] = None,
        indices_to_return_salient: Optional[Sequence[int]] = None,
    ):
        rna, protein = self.get_specific_normalized_expression(
            adata=adata,
            indices=indices,
            n_samples_overall=n_samples_overall,
            transform_batch=transform_batch,
            return_numpy=True,
            n_samples=n_samples,
            batch_size=batch_size,
            scale_protein=scale_protein,
            sample_protein_mixing=sample_protein_mixing,
            include_protein_background=include_protein_background,
            expression_type=expression_type,
            indices_to_return_salient=indices_to_return_salient,
        )
        protein += protein_prior_count

        joint = np.concatenate([rna, protein], axis=1)
        return joint

    def differential_expression(
        self,
        adata: Optional[AnnData] = None,
        groupby: Optional[str] = None,
        group1: Optional[Iterable[str]] = None,
        group2: Optional[str] = None,
        idx1: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        idx2: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.25,
        batch_size: Optional[int] = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Optional[Iterable[str]] = None,
        batchid2: Optional[Iterable[str]] = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        protein_prior_count: float = 0.1,
        scale_protein: bool = False,
        sample_protein_mixing: bool = False,
        include_protein_background: bool = False,
        target_idx: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        A unified method for differential expression analysis.
        Implements `"vanilla"` DE [Lopez18]_ and `"change"` mode DE [Boyeau19]_.

        Args:
        ----
        protein_prior_count:
            Prior count added to protein expression before LFC computation
        scale_protein:
            Force protein values to sum to one in every single cell
            (post-hoc normalization).
        sample_protein_mixing:
            Sample the protein mixture component, i.e., use the parameter to sample a
            Bernoulli that determines if expression is from foreground/background.
        include_protein_background:
            Include the protein background component as part of the protein expression
        target_idx: If not `None`, a boolean or integer identifier should be used for
            cells in the contrastive target group. Normalized expression values derived
            from both salient and background latent embeddings are used when
            {group1, group2} is a subset of the target group, otherwise background
            normalized expression values are used.
        **kwargs:
            Keyword args for
            :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`

        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)
        col_names = np.concatenate(
            [
                np.asarray(adata.var_names),
                self.protein_state_registry.column_names,
            ]
        )

        if target_idx is not None:
            target_idx = np.array(target_idx)
            if target_idx.dtype is np.dtype("bool"):
                assert (
                    len(target_idx) == adata.n_obs
                ), "target_idx mask must be the same length as adata!"
                target_idx = np.arange(adata.n_obs)[target_idx]
            model_fn = partial(
                self._expression_for_de,
                scale_protein=scale_protein,
                sample_protein_mixing=sample_protein_mixing,
                include_protein_background=include_protein_background,
                protein_prior_count=protein_prior_count,
                batch_size=batch_size,
                expression_type=None,
                indices_to_return_salient=target_idx,
                n_samples=100,
            )
        else:
            model_fn = partial(
                self._expression_for_de,
                scale_protein=scale_protein,
                sample_protein_mixing=sample_protein_mixing,
                include_protein_background=include_protein_background,
                protein_prior_count=protein_prior_count,
                batch_size=batch_size,
                expression_type="salient",
                n_samples=100,
            )

        result = _de_core(
            self.get_anndata_manager(adata, required=True),
            model_fn,
            groupby,
            group1,
            group2,
            idx1,
            idx2,
            all_stats,
            cite_seq_raw_counts_properties,
            col_names,
            mode,
            batchid1,
            batchid2,
            delta,
            batch_correction,
            fdr_target,
            silent,
            **kwargs,
        )

        return result
