"""Model class for contrastive-VI for single cell expression data."""

import logging
from typing import List, Optional

from anndata import AnnData
from scvi.data._anndata import _setup_anndata
from scvi.model.base import BaseModelClass

from contrastive_vi.data.utils import get_library_log_means_and_vars
from contrastive_vi.model.base.training_mixin import ContrastiveTrainingMixin
from contrastive_vi.module.contrastive_vi import ContrastiveVIModule

logger = logging.getLogger(__name__)


class ContrastiveVIModel(ContrastiveTrainingMixin, BaseModelClass):
    """
    Model class for contrastive-VI.
    Args:
    ----
        adata: AnnData object that has been registered via
            `ContrastiveVIModel.setup_anndata`.
        background_indices: Indices for background samples in `adata`.
        target_indices: Indices for target samples in `adata`.
        n_hidden: Number of nodes per hidden layer.
        n_latent: Dimensionality of the latent space.
        n_layers: Number of hidden layers used for encoder and decoder NNs.
        **model_kwargs: Additional kwargs for `ContrastiveVIModule`.
    """

    def __init__(
        self,
        adata: AnnData,
        background_indices: List[int],
        target_indices: List[int],
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_observed_lib_size: bool = True,
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
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
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
