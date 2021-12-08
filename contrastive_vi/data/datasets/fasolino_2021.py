"""
Download, read, and preprocess Fasolino et al. (2021) expression data.

Single-cell expression data from Fasolino et al. Multiomics single-cell analysis of
human pancreatic islets reveals novel cellular states in health and type 1 diabetes.
bioRxiv (2021).
"""
import os

import pandas as pd
import scanpy as sc
from anndata import AnnData

from contrastive_vi.data.utils import (
    read_seurat_cell_metadata,
    read_seurat_feature_metadata,
    read_seurat_raw_counts,
)


def download_fasolino_2021(output_path: str) -> None:
    """
    For this data, due to limitations with the Chan-Zuckerberg Biohub website,
    we can't download the data file programatically. Instead, this function redirects
    the user to the webpage where the file can be downloaded.

    Args:
    ----
        output_path: Path where raw data file should live.

    Returns
    -------
        None. This function redirects the user to the Chan-Zuckerberg Biohub to download
        the data file if it doesn't already exist.
    """
    if not os.path.exists(os.path.join(output_path, "local.rds")):
        raise FileNotFoundError(
            "File cannot be downloaded automatically. Please download"
            "RDS file from "
            "https://cellxgene.cziscience.com/collections/51544e44-293b-4c2b-8c26"
            f"-560678423380 and place it in {output_path} to continue."
        )


def read_fasolino_2021(file_directory: str) -> pd.DataFrame:
    """
    Read the expression data for Fasolino et al. 2021 in the given directory.

    Args:
    ----
        file_directory: Directory containing Fasolino et al. 2021 data.

    Returns
    -------
        A data frame containing single-cell gene expression count, with cell
        identification barcodes as column names and gene IDs as indices.
    """
    seurat_object_path = os.path.join(file_directory, "local.rds")
    return read_seurat_raw_counts(seurat_object_path)


def preprocess_fasolino_2021(download_path: str, n_top_genes: int) -> AnnData:
    """
    Preprocess expression data from Fasolino et al., 2021.

    Args:
    ----
        download_path: Path containing the downloaded Fasolino et al. 2021 data file.
        n_top_genes: Number of most variable genes to retain.

    Returns
    -------
        An AnnData object containing single-cell expression data. The layer
        "count" contains the count data for the most variable genes. The X
        variable contains the total-count-normalized and log-transformed data
        for the most variable genes (a copy with all the genes is stored in
        .raw).
    """

    df = read_fasolino_2021(download_path)
    df = df.transpose()

    seurat_object_path = os.path.join(download_path, "local.rds")
    cell_metadata_df = read_seurat_cell_metadata(seurat_object_path)
    feature_metadata_df = read_seurat_feature_metadata(seurat_object_path)

    adata = AnnData(X=df.values, obs=cell_metadata_df, var=feature_metadata_df)
    adata.layers["count"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", n_top_genes=n_top_genes, layer="count", subset=True
    )
    adata = adata[adata.layers["count"].sum(1) != 0]  # Remove cells with all zeros.
    return adata
