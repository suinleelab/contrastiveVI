"""
Download, read, and preprocess Blish et al. (2020) expression data.

Single-cell expression data from Blish et al. A single-cell atlas of the peripheral
immune response in patients with severe COVID-19. Nature Medicine (2020).
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


def download_blish_2020(output_path: str) -> None:
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
            "https://cellxgene.cziscience.com/collections"
            "/a72afd53-ab92-4511-88da-252fb0e26b9a and place it in"
            f"{output_path} to continue."
        )


def read_blish_2020(file_directory: str) -> pd.DataFrame:
    """
    Read the expression data for Blish et al. 2020 in the given directory.

    Args:
    ----
        file_directory: Directory containing Blish et al. 2020 data.

    Returns
    -------
        A data frame containing single-cell gene expression count, with cell
        identification barcodes as column names and gene IDs as indices.
    """
    # Load in required R packages to handle Seurat object file
    seurat_object_path = os.path.join(file_directory, "local.rds")
    return read_seurat_raw_counts(seurat_object_path)


def preprocess_blish_2020(download_path: str, n_top_genes: int) -> AnnData:
    """
    Preprocess expression data from Blish et al., 2020.

    Args:
    ----
        download_path: Path containing the downloaded Blish et al. 2020 data file.
        n_top_genes: Number of most variable genes to retain.

    Returns
    -------
        An AnnData object containing single-cell expression data. The layer
        "count" contains the count data for the most variable genes. The X
        variable contains the total-count-normalized and log-transformed data
        for the most variable genes (a copy with all the genes is stored in
        .raw).
    """

    df = read_blish_2020(download_path)
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
