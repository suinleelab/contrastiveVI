"""Model class for contrastive-VI for single cell expression data."""

import logging
import warnings
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scvi import _CONSTANTS
from scvi.data._anndata import _setup_anndata
from scvi.dataloaders import AnnDataLoader
from scvi.model._utils import (
    _get_batch_code_from_category,
    _get_var_names_from_setup_anndata,
)
from scvi.model.base import BaseModelClass

from contrastive_vi.data.utils import get_library_log_means_and_vars
from contrastive_vi.model.base.training_mixin import ContrastiveTrainingMixin
from contrastive_vi.module.contrastive_vi import ContrastiveVIModule

logger = logging.getLogger(__name__)
Number = Union[int, float]


class ContrastiveVIModel(ContrastiveTrainingMixin, BaseModelClass):
    """
    Model class for contrastive-VI.
    Args:
    ----
        adata: AnnData object that has been registered via
            `ContrastiveVIModel.setup_anndata`.
        n_batch: Number of batches. If 0, no batch effect correction is performed.
        n_hidden: Number of nodes per hidden layer.
        n_latent: Dimensionality of the latent space.
        n_layers: Number of hidden layers used for encoder and decoder NNs.
        dropout_rate: Dropout rate for neural networks.
        use_observed_lib_size: Use observed library size for RNA as scaling factor in
            mean of conditional distribution.
        disentangle: Whether to disentangle the salient and background latent variables.
    """

    def __init__(
        self,
        adata: AnnData,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_background_latent: int = 10,
        n_salient_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_observed_lib_size: bool = True,
        disentangle: bool = False,
    ) -> None:
        super(ContrastiveVIModel, self).__init__(adata)
        # self.summary_stats from BaseModelClass gives info about anndata dimensions
        # and other tensor info.
        if use_observed_lib_size:
            library_log_means, library_log_vars = None, None
        else:
            library_log_means, library_log_vars = get_library_log_means_and_vars(adata)

        self.module = ContrastiveVIModule(
            n_input=self.summary_stats["n_vars"],
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_background_latent=n_background_latent,
            n_salient_latent=n_salient_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            disentangle=disentangle,
        )
        self._model_summary_string = "Contrastive-VI."
        # Necessary line to get params to be used for saving and loading.
        self.init_params_ = self._get_init_params(locals())
        logger.info("The model has been initialized")

    @staticmethod
    def setup_anndata(
        adata: AnnData,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        layer: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        copy: bool = False,
    ) -> Optional[AnnData]:
        """
        Set up AnnData instance for contrastive-VI model.

        Args:
        ----
            adata: AnnData object containing raw counts. Rows represent cells, columns
                represent features.
            batch_key: Key in `adata.obs` for batch information. Categories will
                automatically be converted into integer categories and saved to
                `adata.obs["_scvi_batch"]`. If None, assign the same batch to all the
                data.
            labels_key: Key in `adata.obs` for label information. Categories will
                automatically be converted into integer categories and saved to
                `adata.obs["_scvi_labels"]`. If None, assign the same label to all the
                data.
            layer: If not None, use this as the key in `adata.layers` for raw count
                data.
            categorical_covariate_keys: Keys in `adata.obs` corresponding to categorical
                data. Used in some models.
            continuous_covariate_keys: Keys in `adata.obs` corresponding to continuous
                data. Used in some models.
            copy: If True, a copy of `adata` is returned.

        Returns
        -------
            If `copy` is True, return the modified `adata` set up for contrastive-VI
            model, otherwise `adata` is modified in place.
        """
        return _setup_anndata(
            adata,
            batch_key=batch_key,
            labels_key=labels_key,
            layer=layer,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys,
            copy=copy,
        )

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
        representation_kind: Literal["background":"salient"] = "salient",
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
            x = tensors[_CONSTANTS.X_KEY]
            batch_index = tensors[_CONSTANTS.BATCH_KEY]
            outputs = self.module._generic_inference(
                x=x, batch_index=batch_index, n_samples=1
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

    def get_normalized_expression_fold_change(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        gene_list: Optional[Sequence[str]] = None,
        library_size: Union[float, Literal["latent"]] = 1,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return the normalized (decoded) gene expression.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch: Batch to condition on. If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list: Return frequencies of expression for a subset of genes. This can
            save memory when working with large datasets and few genes are of interest.
        library_size:  Scale the expression frequencies to a common library size. This
            allows gene expression levels to be interpreted on a common scale of
            relevant magnitude. If set to `"latent"`, use the latent library size.
        n_samples: Number of posterior samples to use for estimation.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.

        Returns
        -------
            If `n_samples` > 1, then the shape is `(samples, cells, genes)`. Otherwise,
            shape is `(cells, genes)`. Each element is fold change of salient normalized
            expression divided by background normalized expression.
        """
        exprs = self.get_normalized_expression(
            adata=adata,
            indices=indices,
            transform_batch=transform_batch,
            gene_list=gene_list,
            library_size=library_size,
            n_samples=n_samples,
            batch_size=batch_size,
            return_mean=False,
            return_numpy=True,
        )
        salient_exprs = exprs["salient"]
        background_exprs = exprs["background"]
        fold_change = salient_exprs / background_exprs
        return fold_change

    @torch.no_grad()
    def get_normalized_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        gene_list: Optional[Sequence[str]] = None,
        library_size: Union[float, Literal["latent"]] = 1,
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Return the normalized (decoded) gene expression.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch: Batch to condition on. If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list: Return frequencies of expression for a subset of genes. This can
            save memory when working with large datasets and few genes are of interest.
        library_size:  Scale the expression frequencies to a common library size. This
            allows gene expression levels to be interpreted on a common scale of
            relevant magnitude. If set to `"latent"`, use the latent library size.
        n_samples: Number of posterior samples to use for estimation.
        n_samples_overall: The number of random samples in `adata` to use.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        return_mean: Whether to return the mean of the samples.
        return_numpy: Return a `numpy.ndarray` instead of a `pandas.DataFrame`.
            DataFrame includes gene names as columns. If either `n_samples=1` or
            `return_mean=True`, defaults to `False`. Otherwise, it defaults to `True`.

        Returns
        -------
            A dictionary with keys "background" and "salient", with value as follows.
            If `n_samples` > 1 and `return_mean` is `False`, then the shape is
            `(samples, cells, genes)`. Otherwise, shape is `(cells, genes)`. In this
            case, return type is `pandas.DataFrame` unless `return_numpy` is `True`.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )

        transform_batch = _get_batch_code_from_category(adata, transform_batch)

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = _get_var_names_from_setup_anndata(adata)
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and"
                    " return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if library_size == "latent":
            generative_output_key = "px_rate"
            scaling = 1
        else:
            generative_output_key = "px_scale"
            scaling = library_size

        background_exprs = []
        salient_exprs = []
        for tensors in data_loader:
            x = tensors[_CONSTANTS.X_KEY]
            batch_index = tensors[_CONSTANTS.BATCH_KEY]
            background_per_batch_exprs = []
            salient_per_batch_exprs = []
            for batch in transform_batch:
                if batch is not None:
                    batch_index = torch.ones_like(batch_index) * batch
                inference_outputs = self.module._generic_inference(
                    x=x, batch_index=batch_index, n_samples=n_samples
                )
                z = inference_outputs["z"]
                s = inference_outputs["s"]
                library = inference_outputs["library"]
                background_generative_outputs = self.module._generic_generative(
                    z=z, s=torch.zeros_like(s), library=library, batch_index=batch_index
                )
                salient_generative_outputs = self.module._generic_generative(
                    z=z, s=s, library=library, batch_index=batch_index
                )
                background_outputs = self._preprocess_normalized_expression(
                    background_generative_outputs,
                    generative_output_key,
                    gene_mask,
                    scaling,
                )
                background_per_batch_exprs.append(background_outputs)
                salient_outputs = self._preprocess_normalized_expression(
                    salient_generative_outputs,
                    generative_output_key,
                    gene_mask,
                    scaling,
                )
                salient_per_batch_exprs.append(salient_outputs)

            background_per_batch_exprs = np.stack(
                background_per_batch_exprs
            )  # Shape is (len(transform_batch) x batch_size x n_var).
            salient_per_batch_exprs = np.stack(salient_per_batch_exprs)
            background_exprs += [background_per_batch_exprs.mean(0)]
            salient_exprs += [salient_per_batch_exprs.mean(0)]

        if n_samples > 1:
            # The -2 axis correspond to cells.
            background_exprs = np.concatenate(background_exprs, axis=-2)
            salient_exprs = np.concatenate(salient_exprs, axis=-2)
        else:
            background_exprs = np.concatenate(background_exprs, axis=0)
            salient_exprs = np.concatenate(salient_exprs, axis=0)
        if n_samples > 1 and return_mean:
            background_exprs = background_exprs.mean(0)
            salient_exprs = salient_exprs.mean(0)

        if return_numpy is None or return_numpy is False:
            genes = adata.var_names[gene_mask]
            samples = adata.obs_names[indices]
            background_exprs = pd.DataFrame(
                background_exprs, columns=genes, index=samples
            )
            salient_exprs = pd.DataFrame(salient_exprs, columns=genes, index=samples)
        return {"background": background_exprs, "salient": salient_exprs}

    @staticmethod
    @torch.no_grad()
    def _preprocess_normalized_expression(
        generative_outputs: Dict[str, torch.Tensor],
        generative_output_key: str,
        gene_mask: Union[list, slice],
        scaling: float,
    ) -> np.ndarray:
        output = generative_outputs[generative_output_key]
        output = output[..., gene_mask]
        output *= scaling
        output = output.cpu().numpy()
        return output
