"""
Download, read, and preprocess Papalexi et al. (2021) expression data.

Single-cell expression data from Papalexi et al. Characterizing the molecular regulation
of inhibitory immune checkpoints with multimodal single-cell screens. (Nature Genetics
2021)
"""
import os
import shutil

import constants
import pandas as pd
import scanpy as sc
from anndata import AnnData

from contrastive_vi.data.utils import download_binary_file


def download_papalexi_2021(output_path: str) -> None:
    """
    Download Papalexi et al. 2021 data from the hosting URLs.

    Args:
    ----
        output_path: Output path to store the downloaded and unzipped
        directories.

    Returns
    -------
        None. File directories are downloaded to output_path.
    """

    counts_data_url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE153056&format=file"
    )
    data_output_filename = os.path.join(output_path, "GSE153056_RAW.tar")
    download_binary_file(counts_data_url, data_output_filename)
    shutil.unpack_archive(data_output_filename, output_path)

    metadata_url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE153056&"
        "format=file&file=GSE153056_ECCITE_metadata.tsv.gz"
    )
    metadata_filename = os.path.join(output_path, metadata_url.split("=")[-1])
    download_binary_file(metadata_url, metadata_filename)


def read_papalexi_2021(file_directory: str) -> pd.DataFrame:
    """
    Read the expression data for Papalexi et al. 2021 in the given directory.

    Args:
    ----
        file_directory: Directory containing Papalexi et al. 2021 data.

    Returns
    -------
        A pandas dataframe, with each column representing a cell
        and each row representing a gene feature.
    """

    matrix = pd.read_csv(
        os.path.join(file_directory, "GSM4633614_ECCITE_cDNA_counts.tsv.gz"),
        sep="\t",
        index_col=0,
    )
    return matrix


def preprocess_papalexi_2021(download_path: str, n_top_genes: int) -> AnnData:
    """
    Preprocess expression data from Papalexi et al. 2021.

    Args:
    ----
        download_path: Path containing the downloaded Papalexi et al. 2021 data files.
        n_top_genes: Number of most variable genes to retain.

    Returns
    -------
        An AnnData object containing single-cell expression data. The layer
        "count" contains the count data for the most variable genes. The .X
        variable contains the normalized and log-transformed data for the most variable
        genes. A copy of data with all genes is stored in .raw.
    """

    df = read_papalexi_2021(download_path)

    # Switch dataframe from gene rows and cell columns to cell rows and gene columns
    df = df.transpose()

    metadata = pd.read_csv(
        os.path.join(download_path, "GSE153056_ECCITE_metadata.tsv.gz"),
        sep="\t",
        index_col=0,
    )

    # Note: By initializing the anndata object from a dataframe, variable names
    # are automatically stored in adata.var
    adata = AnnData(df)
    adata.obs = metadata

    # Protein measurements also collected as part of CITE-Seq
    protein_counts_df = pd.read_csv(
        os.path.join(download_path, "GSM4633615_ECCITE_ADT_counts.tsv.gz"),
        sep="\t",
        index_col=0,
    )

    # Switch dataframe from protein rows and cell columns to cell rows and protein
    # columns
    protein_counts_df = protein_counts_df.transpose()

    # Storing protein counts in an obsm field as expected by totalVI
    # (see https://docs.scvi-tools.org/en/stable/tutorials/notebooks/totalVI.html
    # for an example). Since `protein_counts_df` is annotated with protein names,
    # our obsm field will retain them as well.
    adata.obsm[constants.PROTEIN_EXPRESSION_KEY] = protein_counts_df

    adata.layers["count"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", n_top_genes=n_top_genes, layer="count", subset=True
    )
    adata = adata[adata.layers["count"].sum(1) != 0]  # Remove cells with all zeros.
    return adata
