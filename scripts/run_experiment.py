"""Run a model training experiment."""
import argparse
import os

import constants
import numpy as np
import pickle
import scanpy as sc
import torch
from pcpca import PCPCA
from sklearn.preprocessing import StandardScaler
from scvi._settings import settings
from scvi.model import SCVI

from contrastive_vi.model.contrastive_vi import ContrastiveVIModel

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    type=str,
    choices=constants.DATASET_LIST,
    help="Which dataset to use for the experiment.",
)
parser.add_argument(
    "method", type=str, choices=["contrastiveVI", "scVI", "PCPCA"], help="Which model to train"
)
parser.add_argument(
    "-use_gpu", action="store_true", help="Flag for enabling GPU usage."
)
parser.add_argument(
    "--n_genes",
    type=int,
    default=2000,
    help="Number of highly variable genes in dataset.",
)
parser.add_argument(
    "--gpu_num",
    type=int,
    help="If -use_gpu is enabled, controls which specific GPU to use for training.",
)
parser.add_argument(
    "--random_seeds",
    nargs="+",
    type=int,
    default=constants.DEFAULT_SEEDS,
    help="List of random seeds to use for experiments, with one model trained per "
    "seed.",
)

args = parser.parse_args()

adata = sc.read_h5ad(
    os.path.join(
        constants.DEFAULT_DATA_PATH,
        args.dataset,
        "preprocessed",
        f"adata_top_{args.n_genes}_genes.h5ad",
    )
)

if args.dataset == "zheng_2017":
    split_key = "condition"
    background_value = "healthy"
elif args.dataset == "haber_2017":
    split_key = "condition"
    background_value = "Control"
else:
    raise NotImplementedError("Dataset not yet implemented.")

if args.use_gpu:
    if args.gpu_num is not None:
        use_gpu = args.gpu_num
    else:
        use_gpu = True
else:
    use_gpu = False

deep_learning_models = ["scVI", "contrastiveVI"]

# For deep learning methods, we experiment with multiple random initializations
# to get error bars
if args.method in deep_learning_models:
    for seed in args.random_seeds:
        settings.seed = seed

        if args.method == "contrastiveVI":
            ContrastiveVIModel.setup_anndata(adata, layer="count")

            model = ContrastiveVIModel(adata)

            # np.where returns a list of indices, one for each dimension of the input array.
            # Since we have 1d arrays, we simply grab the first (and only) returned list.
            background_indices = np.where(adata.obs[split_key] == background_value)[0]
            target_indices = np.where(adata.obs[split_key] != background_value)[0]

            model.train(
                check_val_every_n_epoch=1,
                train_size=0.8,
                background_indices=background_indices,
                target_indices=target_indices,
                use_gpu=use_gpu,
                early_stopping=True,
                max_epochs=25,
            )

        elif args.method == "scVI":
            # We only train scVI with target samples
            target_adata = adata[adata.obs[split_key] != background_value].copy()

            SCVI.setup_anndata(target_adata, layer="count")

            model = SCVI(target_adata)

            model.train(
                check_val_every_n_epoch=1,
                train_size=0.8,
                use_gpu=use_gpu,
                early_stopping=True,
            )

        checkpoint_dir = os.path.join(
            constants.DEFAULT_RESULTS_PATH, args.dataset, args.method, str(seed)
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model, os.path.join(checkpoint_dir, "model.ckpt"))

elif args.method == "PCPCA":
    # In the original PCPCA paper they use raw count data and standardize it to 0-mean
    # unit variance, so we do the same thing here
    background_data = StandardScaler().fit_transform(
        adata[adata.obs[split_key] == background_value].layers['count'])
    target_data = StandardScaler().fit_transform(
        adata[adata.obs[split_key] != background_value].layers['count'])

    n, m = target_data.shape[1], background_data.shape[1]
    model = PCPCA(n_components=10, gamma=0.7)

    # The PCPCA package expects data to have rows be features and columns be samples
    # so we transpose the data here
    model.fit(target_data.transpose(), background_data.transpose())

    model_dir = os.path.join(constants.DEFAULT_RESULTS_PATH, args.dataset, args.method)
    os.makedirs(model_dir, exist_ok=True)
    pickle.dump(model, open(os.path.join(model_dir, "model.pkl"), 'wb'))

