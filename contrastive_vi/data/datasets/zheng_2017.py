"""Download, read, and preprocess Zheng et al. (2017) expression data.

Single-cell expression data from Zheng et al. Massively parallel digital
transcriptional profiling of single cells. Nature Communications (2017).
"""
import os
import shutil

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.io import mmread

from contrastive_vi.data.utils import download_binary_file


def download_zheng_2017(output_path: str) -> None:
    """Download Zheng et al. 2017 data from the hosting URLs.

    Args:
        output_path: Output path to store the downloaded and unzipped
        directories.

    Returns:
        None. File directories are downloaded and unzipped in output_path.
    """
    host = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
    host_directories = [
        (
            "aml027_post_transplant/"
            "aml027_post_transplant_filtered_gene_bc_matrices.tar.gz"
        ),
        (
            "aml027_pre_transplant/"
            "aml027_pre_transplant_filtered_gene_bc_matrices.tar.gz"
        ),
        (
            "aml035_post_transplant/"
            "aml035_post_transplant_filtered_gene_bc_matrices.tar.gz"
        ),
        (
            "aml035_pre_transplant/"
            "aml035_pre_transplant_filtered_gene_bc_matrices.tar.gz"
        ),
        (
            "frozen_bmmc_healthy_donor1/"
            "frozen_bmmc_healthy_donor1_filtered_gene_bc_matrices.tar.gz"
        ),
        (
            "frozen_bmmc_healthy_donor2/"
            "frozen_bmmc_healthy_donor2_filtered_gene_bc_matrices.tar.gz"
        ),
    ]
    urls = [host + host_directory for host_directory in host_directories]
    output_filenames = [
        os.path.join(output_path, url.split("/")[-1]) for url in urls
    ]
    for url, output_filename in zip(urls, output_filenames):
        download_binary_file(url, output_filename)
        output_dir = output_filename.replace(".tar.gz", "")
        shutil.unpack_archive(output_filename, output_dir)


def read_zheng_2017(file_directory: str) -> pd.DataFrame:
    """Read the expression data for in a downloaded file directory.

    Args:
        file_directory: A downloaded and unzipped file directory.

    Returns:
        A data frame containing single-cell gene expression count, with cell
        identification barcodes as column names and gene IDs as indices.
    """
    data = mmread(
        os.path.join(file_directory, "filtered_matrices_mex/hg19/matrix.mtx")
    ).toarray()
    genes = pd.read_table(
        os.path.join(file_directory, "filtered_matrices_mex/hg19/genes.tsv"),
        header=None,
    )
    barcodes = pd.read_table(
        os.path.join(
            file_directory, "filtered_matrices_mex/hg19/barcodes.tsv"
        ),
        header=None,
    )
    return pd.DataFrame(
        data, index=genes.iloc[:, 0].values, columns=barcodes.iloc[:, 0].values
    )


def preprocess_zheng_2017(download_path: str, n_top_genes: int) -> AnnData:
    """Preprocess expression data from Zheng et al. 2017.

    Args:
        download_path: Path containing the downloaded and unzipped file
            directories.
        n_top_genes: Number of most variable genes to retain.

    Returns:
        An AnnData object containing single-cell expression data. The layer
        "count" contains the count data for the most variable genes. The X
        variable contains the total-count-normalized and log-transformed data
        for the most variable genes (a copy with all the genes is stored in
        .raw).
    """
    file_directory_dict = {
        "aml027_pre_transplant": (
            "aml027_pre_transplant_filtered_gene_bc_matrices"
        ),
        "aml027_post_transplant": (
            "aml027_post_transplant_filtered_gene_bc_matrices"
        ),
        "aml035_pre_transplant": (
            "aml035_pre_transplant_filtered_gene_bc_matrices"
        ),
        "aml035_post_transplant": (
            "aml035_post_transplant_filtered_gene_bc_matrices"
        ),
        "donor1_healthy": (
            "frozen_bmmc_healthy_donor1_filtered_gene_bc_matrices"
        ),
        "donor2_healthy": (
            "frozen_bmmc_healthy_donor2_filtered_gene_bc_matrices"
        ),
    }
    df_dict = {
        sample_id: read_zheng_2017(os.path.join(download_path, file_directory))
        for sample_id, file_directory in file_directory_dict.items()
    }
    gene_set_list = []
    for sample_id, df in df_dict.items():
        df = df.iloc[:, np.sum(df.values, axis=0) != 0]
        df = df.iloc[np.sum(df.values, axis=1) != 0, :]
        df = df.transpose()
        gene_set_list.append(set(df.columns))
        patient_id, condition = sample_id.split("_", 1)
        df["patient_id"] = patient_id
        df["condition"] = condition
        df_dict[sample_id] = df
    shared_genes = list(set.intersection(*gene_set_list))
    data_list = []
    meta_data_list = []
    for df in df_dict.values():
        data_list.append(df[shared_genes])
        meta_data_list.append(df[["patient_id", "condition"]])
    data = pd.concat(data_list)
    meta_data = pd.concat(meta_data_list)
    adata = AnnData(
        X=data.reset_index(drop=True), obs=meta_data.reset_index(drop=True)
    )
    adata.layers["count"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
        layer="count",
        subset=True,
    )
    return adata
