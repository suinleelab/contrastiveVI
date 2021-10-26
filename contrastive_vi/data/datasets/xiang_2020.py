"""Download, read, and preprocess Xiang et al. (2020) expression data.

Single-cell expression data from Xiang et al. A Single-Cell Transcriptional Roadmap
of the Mouse and Human Lymph Node Lymphatic Vasculature. Frontiers in Cardiovascular
Medicine (2020).
"""
import os
from typing import Tuple

import anndata
import pandas as pd
import scanpy as sc
from anndata import AnnData

from contrastive_vi.data.utils import (
    read_seurat_cell_metadata,
    read_seurat_feature_metadata,
    read_seurat_raw_counts,
)


def download_xiang_2020(output_path: str) -> None:
    """For this data, due to limitations with the Chan-Zuckerberg Biohub website,
    we can't download the data file programatically. Instead, this function redirects
    the user to the webpage where the file can be downloaded.

    Args:
        output_path: Path where raw data file should live.

    Returns:
        None. This function redirects the user to the Chan-Zuckerberg Biohub to download
        the data file if it doesn't already exist.
    """
    if not os.path.exists(os.path.join(output_path, "local_mouse.rds")):
        raise FileNotFoundError(
            "Mouse expression data file cannot be downloaded automatically."
            "Please download RDS file from "
            "https://cellxgene.cziscience.com/collections/9c8808ce-1138-4dbe-818c"
            f"-171cff10e650, place it in {output_path}, and rename it 'local_mouse.rds'"
            " to continue."
        )

    if not os.path.exists(os.path.join(output_path, "local_human.rds")):
        raise FileNotFoundError(
            "Human expression data file cannot be downloaded automatically."
            "Please download RDS file from "
            "https://cellxgene.cziscience.com/collections/9c8808ce-1138-4dbe-818c"
            f"-171cff10e650, place it in {output_path}, and rename it 'local_human.rds'"
            " to continue."
        )


def read_xiang_2020(file_directory: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the expression data for Xiang et al. 2020 in the given directory.

    Args:
        file_directory: Directory containing Xiang et al. 2020 data.

    Returns: Two data frames of raw count expression data. The first contains
    single-cell gene expression count data from mouse samples, with cell identification
    barcodes as column names and gene IDs as indices. The second contains count data
    from human samples with the same format.
    """
    mouse_df = read_seurat_raw_counts(os.path.join(file_directory, "local_mouse.rds"))
    human_df = read_seurat_raw_counts(os.path.join(file_directory, "local_human.rds"))

    return mouse_df, human_df


def preprocess_xiang_2020(download_path: str, n_top_genes: int) -> AnnData:
    """Preprocess expression data from Xiang et al., 2020.

    Args:
        download_path: Path containing the downloaded Xiang et al. 2020 data files.
        n_top_genes: Number of most variable genes to retain.

    Returns:
        An AnnData object containing single-cell expression data. The layer
        "count" contains the count data for the most variable genes. The X
        variable contains the total-count-normalized and log-transformed data
        for the most variable genes (a copy with all the genes is stored in
        .raw).
    """

    mouse_df, human_df = read_xiang_2020(download_path)
    mouse_df, human_df = mouse_df.transpose(), human_df.transpose()

    mouse_seurat_object_path = os.path.join(download_path, "local_mouse.rds")
    mouse_metadata_df = read_seurat_cell_metadata(mouse_seurat_object_path)
    mouse_feature_metadata_df = read_seurat_feature_metadata(mouse_seurat_object_path)
    mouse_adata = AnnData(
        X=mouse_df.values, obs=mouse_metadata_df, var=mouse_feature_metadata_df
    )

    # Since the human cell data only has healthy cells, we keep only healthy
    # cells for mouse data too
    mouse_adata = mouse_adata[mouse_adata.obs["disease"] == "normal"]

    # Mouse gene names were lowercase for some reason while human genes were uppercase,
    # so here we make the mouse gene names also uppercase
    mouse_adata.var["feature_name"] = mouse_adata.var["feature_name"].apply(
        lambda x: x.upper()
    )

    # The original index used mouse-specific names, while the 'feature_name'
    # slot in the gene metadata contained species-agnostic names. So that we can merge
    # mouse and human data, we switch the index to these species-agnostic names.
    mouse_adata.var.index = mouse_adata.var["feature_name"]

    human_seurat_object_path = os.path.join(download_path, "local_human.rds")
    human_metadata_df = read_seurat_cell_metadata(human_seurat_object_path)
    human_feature_metadata_df = read_seurat_feature_metadata(human_seurat_object_path)
    human_adata = AnnData(
        X=human_df.values, obs=human_metadata_df, var=human_feature_metadata_df
    )
    human_adata.var.index = human_adata.var["feature_name"]

    # Subset data to shared genes (homologs)
    shared_genes = list(
        set(human_adata.var["feature_name"].values).intersection(
            mouse_adata.var["feature_name"].values
        )
    )
    human_adata = human_adata[:, shared_genes]
    mouse_adata = mouse_adata[:, shared_genes]

    full_adata = anndata.concat([human_adata, mouse_adata])
    full_adata.layers["count"] = full_adata.X.copy()
    sc.pp.normalize_total(full_adata)
    sc.pp.log1p(full_adata)
    full_adata.raw = full_adata
    sc.pp.highly_variable_genes(
        full_adata,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
        layer="count",
        subset=True,
    )
    return full_adata
