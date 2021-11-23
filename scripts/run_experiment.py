"""Run a model training experiment."""
import argparse
import os
import pickle
import sys

import constants
import numpy as np
import scanpy as sc
import tensorflow as tf
import torch
from contrastive import CPCA
from cplvm import CPLVM
from pcpca import PCPCA
from scvi._settings import settings
from scvi.model import SCVI
from sklearn.preprocessing import StandardScaler

from contrastive_vi.model.contrastive_vi import ContrastiveVIModel
from contrastive_vi.model.cvae import CVAEModel

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    type=str,
    choices=constants.DATASET_LIST,
    help="Which dataset to use for the experiment.",
)
parser.add_argument(
    "method",
    type=str,
    choices=[
        "contrastiveVI",
        "TC_contrastiveVI",
        "scVI",
        "PCPCA",
        "cPCA",
        "CPLVM",
        "cVAE",
    ],
    help="Which model to train",
)
parser.add_argument(
    "-use_gpu", action="store_true", help="Flag for enabling GPU usage."
)
parser.add_argument(
    "--latent_size",
    type=int,
    default=10,
    help="Size of the model's latent space. For contrastive models, this is the size "
    "of the salient latent space. For non-contrastive, this is the size of the "
    "single latent space.",
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
print(f"Running {sys.argv[0]} with arguments")
for arg in vars(args):
    print(f"\t{arg}={getattr(args, arg)}")

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
elif args.dataset == "fasolino_2021":
    split_key = "disease_state"
    background_value = "Control"
elif args.dataset == "mcfarland_2020":
    split_key = "condition"
    background_value = "DMSO"
else:
    raise NotImplementedError("Dataset not yet implemented.")

torch_models = ["scVI", "contrastiveVI", "TC_contrastiveVI", "cVAE"]
tf_models = ["CPLVM"]
normalized_expressions = None

# For deep learning methods, we experiment with multiple random initializations
# to get error bars
if args.method in torch_models:
    if args.use_gpu:
        if args.gpu_num is not None:
            use_gpu = args.gpu_num
        else:
            use_gpu = True
    else:
        use_gpu = False

    for seed in args.random_seeds:
        settings.seed = seed

        if args.method == "contrastiveVI":
            ContrastiveVIModel.setup_anndata(adata, layer="count")
            model = ContrastiveVIModel(
                adata,
                disentangle=False,
                n_salient_latent=args.latent_size,
                n_background_latent=args.latent_size,
            )

            # np.where returns a list of indices, one for each dimension of the input
            # array. Since we have 1d arrays, we simply grab the first (and only)
            # returned list.
            background_indices = np.where(adata.obs[split_key] == background_value)[0]
            target_indices = np.where(adata.obs[split_key] != background_value)[0]
            model.train(
                check_val_every_n_epoch=1,
                train_size=0.8,
                background_indices=background_indices,
                target_indices=target_indices,
                use_gpu=use_gpu,
                early_stopping=True,
                max_epochs=500,
            )
            target_adata = adata[adata.obs[split_key] != background_value].copy()
            latent_representations = model.get_latent_representation(
                adata=target_adata, representation_kind="salient"
            )
            normalized_expressions = model.get_normalized_expression(
                adata=adata, n_samples=100
            )

        elif args.method == "TC_contrastiveVI":
            ContrastiveVIModel.setup_anndata(adata, layer="count")
            model = ContrastiveVIModel(
                adata,
                disentangle=True,
                n_salient_latent=args.latent_size,
                n_background_latent=args.latent_size,
            )

            # np.where returns a list of indices, one for each dimension of the input
            # array. Since we have 1d arrays, we simply grab the first (and only)
            # returned list.
            background_indices = np.where(adata.obs[split_key] == background_value)[0]
            target_indices = np.where(adata.obs[split_key] != background_value)[0]
            model.train(
                check_val_every_n_epoch=1,
                train_size=0.8,
                background_indices=background_indices,
                target_indices=target_indices,
                use_gpu=use_gpu,
                early_stopping=True,
                max_epochs=500,
            )
            target_adata = adata[adata.obs[split_key] != background_value].copy()
            latent_representations = model.get_latent_representation(
                adata=target_adata, representation_kind="salient"
            )
            normalized_expressions = model.get_normalized_expression(
                adata=adata, n_samples=100
            )

        elif args.method == "scVI":
            # We only train scVI with target samples
            target_adata = adata[adata.obs[split_key] != background_value].copy()
            SCVI.setup_anndata(target_adata, layer="count")
            model = SCVI(target_adata, n_latent=args.latent_size)
            model.train(
                check_val_every_n_epoch=1,
                train_size=0.8,
                use_gpu=use_gpu,
                early_stopping=True,
                max_epochs=500,
            )
            latent_representations = model.get_latent_representation(adata=target_adata)

        elif args.method == "cVAE":
            CVAEModel.setup_anndata(adata)
            model = CVAEModel(
                adata,
                n_salient_latent=args.latent_size,
                n_background_latent=args.latent_size,
            )

            # np.where returns a list of indices, one for each dimension of the input
            # array. Since we have 1d arrays, we simply grab the first (and only)
            # returned list.
            background_indices = np.where(adata.obs[split_key] == background_value)[0]
            target_indices = np.where(adata.obs[split_key] != background_value)[0]
            model.train(
                check_val_every_n_epoch=1,
                train_size=0.8,
                background_indices=background_indices,
                target_indices=target_indices,
                use_gpu=use_gpu,
                early_stopping=True,
                max_epochs=500,
            )
            target_adata = adata[adata.obs[split_key] != background_value].copy()
            latent_representations = model.get_latent_representation(
                adata=target_adata, representation_kind="salient"
            )

        results_dir = os.path.join(
            constants.DEFAULT_RESULTS_PATH,
            args.dataset,
            args.method,
            f"latent_{args.latent_size}",
            str(seed),
        )
        os.makedirs(results_dir, exist_ok=True)
        torch.save(
            model, os.path.join(results_dir, "model.ckpt"), pickle_protocol=4
        )  # Protocol version >= 4 is required to save large model files.
        np.save(
            arr=latent_representations,
            file=os.path.join(results_dir, "latent_representations.npy"),
        )
        if normalized_expressions is not None:
            background_normalized_expression = normalized_expressions["background"]
            salient_normalized_expression = normalized_expressions["salient"]
            np.save(
                arr=background_normalized_expression,
                file=os.path.join(results_dir, "background_normalized_expression.npy"),
            )
            np.save(
                arr=salient_normalized_expression,
                file=os.path.join(results_dir, "salient_normalized_expression.npy"),
            )

elif args.method in tf_models:
    if args.use_gpu:
        if args.gpu_num is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Use CPU.

    for seed in args.random_seeds:
        tf.random.set_seed(seed)

        if args.method == "CPLVM":
            background_data = (
                adata[adata.obs[split_key] == background_value]
                .layers["count"]
                .transpose()
            )
            target_data = (
                adata[adata.obs[split_key] != background_value]
                .layers["count"]
                .transpose()
            )
            model = CPLVM(k_shared=args.latent_size, k_foreground=args.latent_size)
            model_output = model.fit_model_vi(
                X=background_data,
                Y=target_data,
                compute_size_factors=True,
                is_H0=False,
                offset_term=True,
            )
            model_output = {key: tensor.numpy() for key, tensor in model_output.items()}
            latent_representations = model_output["qty_mean"].transpose()

            results_dir = os.path.join(
                constants.DEFAULT_RESULTS_PATH,
                args.dataset,
                args.method,
                f"latent_{args.latent_size}",
                str(seed),
            )

            os.makedirs(results_dir, exist_ok=True)
            pickle.dump(
                model_output, open(os.path.join(results_dir, "model.pkl"), "wb")
            )
            np.save(
                arr=latent_representations,
                file=os.path.join(results_dir, "latent_representations.npy"),
            )

elif args.method == "PCPCA":
    # In the original PCPCA paper they use raw count data and standardize it to 0-mean
    # unit variance, so we do the same thing here
    background_data = StandardScaler().fit_transform(
        adata[adata.obs[split_key] == background_value].layers["count"]
    )
    target_data = StandardScaler().fit_transform(
        adata[adata.obs[split_key] != background_value].layers["count"]
    )

    model = PCPCA(n_components=args.latent_size, gamma=0.7)
    # The PCPCA package expects data to have rows be features and columns be samples
    # so we transpose the data here
    model.fit(target_data.transpose(), background_data.transpose())

    # model.transform() returns a tuple of transformed target and background data (in
    # this order).
    latent_representations = model.transform(
        target_data.transpose(), background_data.transpose()
    )[0].transpose()

    results_dir = os.path.join(
        constants.DEFAULT_RESULTS_PATH,
        args.dataset,
        args.method,
        f"latent_{args.latent_size}",
    )

    os.makedirs(results_dir, exist_ok=True)
    pickle.dump(model, open(os.path.join(results_dir, "model.pkl"), "wb"))
    np.save(
        arr=latent_representations,
        file=os.path.join(results_dir, "latent_representations.npy"),
    )

elif args.method == "cPCA":
    background_data = adata[adata.obs[split_key] == background_value].X
    target_data = adata[adata.obs[split_key] != background_value].X

    model = CPCA(n_components=args.latent_size, standardize=True)
    model.fit(
        foreground=target_data,
        background=background_data,
        preprocess_with_pca_dim=args.n_genes,  # Avoid preprocessing with standard PCA.
    )

    # model.transform() returns a list of transformed data for varying alpha values.
    latent_representations = model.transform(target_data, n_alphas_to_return=1)[0]

    results_dir = os.path.join(
        constants.DEFAULT_RESULTS_PATH,
        args.dataset,
        args.method,
        f"latent_{args.latent_size}",
    )
    os.makedirs(results_dir, exist_ok=True)
    pickle.dump(model, open(os.path.join(results_dir, "model.pkl"), "wb"))
    np.save(
        arr=latent_representations,
        file=os.path.join(results_dir, "latent_representations.npy"),
    )

print("Done!")
