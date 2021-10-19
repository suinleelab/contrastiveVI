import logging

from anndata import AnnData
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin

from ._scvi_module import SCVIModule

logger = logging.getLogger(__name__)


class SCVIModel(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Skeleton for an scvi-tools model.

    Please use this skeleton to create new models.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    **model_kwargs
        Keyword args for :class:`~mypackage.MyModule`
    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.data.setup_anndata(adata, batch_key="batch")
    >>> vae = mypackage.MyModel(adata)
    >>> vae.train()
    >>> adata.obsm["X_mymodel"] = vae.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        **model_kwargs,
    ):
        super(SCVIModel, self).__init__(adata)

        # self.summary_stats gives info about anndata dimensions and other tensor info

        self.module = SCVIModule(
            n_input=self.summary_stats["n_vars"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            **model_kwargs,
        )
        self._model_summary_string = "Overwrite this attribute"
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")
