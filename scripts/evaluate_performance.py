"""Evaluate trained models."""
import os
from typing import Dict, Optional

import constants
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def evaluate_latent_representations(
    labels: np.ndarray,
    latent_representations: np.ndarray,
    clustering_seed: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate latent representations against ground truth labels"""
    latent_clusters = (
        KMeans(n_clusters=len(np.unique(labels)), random_state=clustering_seed)
        .fit(latent_representations)
        .labels_
    )

    silhouette = silhouette_score(latent_representations, labels)
    calinski_harabasz = calinski_harabasz_score(latent_representations, labels)
    davies_bouldin = davies_bouldin_score(latent_representations, labels)

    adjusted_random_index = adjusted_rand_score(labels, latent_clusters)
    adjusted_mutual_info = adjusted_mutual_info_score(labels, latent_clusters)

    return {
        "silhouette": silhouette,
        "calinski_harabasz": calinski_harabasz,
        "davies_bouldin": davies_bouldin,
        "adjusted_random_index": adjusted_random_index,
        "adjusted_mutual_info": adjusted_mutual_info,
    }


datasets = ["mcfarland_2020", "zheng_2017", "haber_2017", "fasolino_2021"]
latent_sizes = [2, 10, 32, 64]
dataset_split_lookup = constants.DATASET_SPLIT_LOOKUP

deterministic_methods = ["cPCA", "PCPCA"]
non_deterministic_methods = [
    "cVAE",
    "CPLVM",
    "CGLVM",
    "scVI",
    "contrastiveVI",
    "TC_contrastiveVI",
    "mmd_contrastiveVI",
]
methods = deterministic_methods + non_deterministic_methods

result_df_list = []

for dataset in datasets:
    print(f"Evaluating models with dataset {dataset}...")
    adata = sc.read_h5ad(
        os.path.join(
            constants.DEFAULT_DATA_PATH,
            f"{dataset}/preprocessed/adata_top_2000_genes.h5ad",
        )
    )
    split_key = dataset_split_lookup[dataset]["split_key"]
    background_value = dataset_split_lookup[dataset]["background_value"]
    label_key = dataset_split_lookup[dataset]["label_key"]
    target_labels = adata[adata.obs[split_key] != background_value].obs[label_key]
    target_labels = LabelEncoder().fit_transform(target_labels)

    for method in tqdm(methods):
        if method in deterministic_methods:
            method_seeds = [""]
        else:
            method_seeds = constants.DEFAULT_SEEDS
        for latent_size in latent_sizes:
            for method_seed in method_seeds:
                output_dir = os.path.join(
                    constants.DEFAULT_RESULTS_PATH,
                    dataset,
                    method,
                    f"latent_{latent_size}",
                    f"{method_seed}",
                )

                model_filepath = os.path.join(output_dir, "model.ckpt")
                if os.path.exists(model_filepath):
                    model = torch.load(model_filepath, map_location="cpu")
                    num_epochs = model.history["reconstruction_loss_train"].shape[0]
                else:
                    model = None
                    num_epochs = 0

                representation_filepath = os.path.join(
                    output_dir, "latent_representations.npy"
                )
                latent_representations = np.load(representation_filepath)
                metrics = evaluate_latent_representations(
                    target_labels,
                    latent_representations,
                    clustering_seed=123,
                )
                metrics = pd.DataFrame({key: [val] for key, val in metrics.items()})
                metrics["dataset"] = dataset
                metrics["method"] = method
                metrics["latent_size"] = latent_size
                metrics["seed"] = (
                    "Deterministic" if method in deterministic_methods else method_seed
                )
                metrics["num_epochs"] = num_epochs
                result_df_list.append(metrics)

result_df = pd.concat(result_df_list).reset_index(drop=True)
result_df.to_csv(
    os.path.join(constants.DEFAULT_RESULTS_PATH, "performance_summary.csv"),
    index=False,
)
