"""Download, read, and preprocess Mcfarland et al. (2020) expression data.

Single-cell expression data from Mcfarland et al. Multiplexed single-cell transcriptional response profiling to define cancer vulnerabilities and therapeutic mechanism of action.
Nature Communications (2020).
"""
import os
import shutil
from typing import Tuple

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.io import mmread

from contrastive_vi.data.utils import download_binary_file


def download_mcfarland_2020(output_path: str) -> None:
    """Download Mcfarland et al. 2020 data from the hosting URLs.

    Args:
        output_path: Output path to store the downloaded and unzipped
        directories.

    Returns:
        None. File directories are downloaded and unzipped in output_path.
    """
    idasanutlin_url = "https://figshare.com/ndownloader/files/18716351"
    idasanutlin_output_filename = os.path.join(output_path, "idasanutlin.zip")

    download_binary_file(idasanutlin_url, idasanutlin_output_filename)
    idasanutlin_output_dir = idasanutlin_output_filename.replace(".zip", "")
    shutil.unpack_archive(idasanutlin_output_filename, idasanutlin_output_dir)

    dmso_url = "https://figshare.com/ndownloader/files/18716354"
    dmso_output_filename = os.path.join(output_path, "dmso.zip")

    download_binary_file(dmso_url, dmso_output_filename)
    dmso_output_dir = dmso_output_filename.replace(".zip", "")
    shutil.unpack_archive(dmso_output_filename, dmso_output_dir)


def _read_mixseq_df(directory: str) -> pd.DataFrame:
    data = mmread(os.path.join(directory, "matrix.mtx"))
    barcodes = pd.read_table(os.path.join(directory, "barcodes.tsv"), header=None)
    classifications = pd.read_csv(os.path.join(directory, "classifications.csv"))
    classifications["cell_line"] = np.array(
        [x.split("_")[0] for x in classifications.singlet_ID.values]
    )
    gene_names = pd.read_table(os.path.join(directory, "genes.tsv"), header=None)

    df = pd.DataFrame(
        data.toarray(),
        columns=barcodes.iloc[:, 0].values,
        index=gene_names.iloc[:, 0].values,
    )
    return df

def _get_tp53_mutation_status(directory: str) -> np.array:
    # Taken from https://cancerdatascience.org/blog/posts/mix-seq/
    TP53_WT = ['LNCAPCLONEFGC_PROSTATE', 'DKMG_CENTRAL_NERVOUS_SYSTEM',
    'NCIH226_LUNG', 'RCC10RGB_KIDNEY', 'SNU1079_BILIARY_TRACT',
    'CCFSTTG1_CENTRAL_NERVOUS_SYSTEM', 'COV434_OVARY']

    classifications = pd.read_csv(
        os.path.join(directory, "classifications.csv"))
    TP53_mutation_status = ["Wild Type" if x in TP53_WT else "Mutation" for
                                        x in classifications.singlet_ID.values]
    return np.array(TP53_mutation_status)


def read_mcfarland_2020(file_directory: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the expression data for Mcfarland et al. 2020 in the given directory.

    Args:
        file_directory: Directory containing Mcfarland et al. 2020 data.

    Returns: Two data frames of raw count expression data. The first contains
    single-cell gene expression count data from cancer cell lines exposed to idasanutlin
    with cell identification barcodes as column names and gene IDs as indices. The
    second contains count data with the same format from samples exposed to a control
    solution (DMSO).
    """
    idasanutlin_dir = os.path.join(file_directory, "idasanutlin", "Idasanutlin_24hr_expt1")
    idasanutlin_df = _read_mixseq_df(idasanutlin_dir)

    dmso_dir = os.path.join(file_directory, "dmso", "DMSO_24hr_expt1")
    dmso_df = _read_mixseq_df(dmso_dir)

    return idasanutlin_df, dmso_df


def preprocess_mcfarland_2020(download_path: str, n_top_genes: int) -> AnnData:
    """Preprocess expression data from Mcfarland et al., 2020.

    Args:
        download_path: Path containing the downloaded Mcfarland et al. 2020 data files.
        n_top_genes: Number of most variable genes to retain.

    Returns:
        An AnnData object containing single-cell expression data. The layer
        "count" contains the count data for the most variable genes. The X
        variable contains the total-count-normalized and log-transformed data
        for the most variable genes (a copy with all the genes is stored in
        .raw).
    """

    idasanutlin_df, dmso_df = read_mcfarland_2020(download_path)
    idasanutlin_df, dmso_df = idasanutlin_df.transpose(), dmso_df.transpose()

    idasanutlin_adata = AnnData(idasanutlin_df)
    idasanutlin_dir = os.path.join(download_path, "idasanutlin", "Idasanutlin_24hr_expt1")
    idasanutlin_adata.obs['TP53_mutation_status'] = _get_tp53_mutation_status(idasanutlin_dir)
    idasanutlin_adata.obs['condition'] = np.repeat("Idasanutlin", idasanutlin_adata.shape[0])

    dmso_adata = AnnData(dmso_df)
    dmso_dir = os.path.join(download_path, "dmso", "DMSO_24hr_expt1")
    dmso_adata.obs['TP53_mutation_status'] = _get_tp53_mutation_status(dmso_dir)
    dmso_adata.obs['condition'] = np.repeat("DMSO", dmso_adata.shape[0])

    full_adata = anndata.concat([idasanutlin_adata, dmso_adata])
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
